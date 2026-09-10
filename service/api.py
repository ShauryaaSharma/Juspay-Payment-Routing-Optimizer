"""HTTP service: the router on the hot path, the agent on the cold one.

Two endpoints with nothing in common operationally, and keeping that visible is
the point of this module.

`POST /route` runs per transaction. It must answer in well under a millisecond,
so it touches no network: the posterior is in process, constraints are served
from a locally-refreshed cache, and the outcome event is handed to a
fire-and-forget pipeline. Nothing it does can block on Redis, Kafka or
ClickHouse.

`POST /investigate` runs per incident -- dozens of times a day, not 350 million.
It calls a language model and takes seconds. It is in the same process here only
because this is a demonstration; in production it is a separate service with a
separate SLO, because an agent that is slow is fine and a router that is slow is
an outage.

Everything optional stays optional. With no Redis, no Kafka, no ClickHouse and
no API key, this still starts and still routes -- it just keeps its state in
memory and drops its events, which is exactly what the rest of the project does.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np

from src.agent.config import settings
from src.agent.constraints import ConstraintStore, from_diagnosis
from src.gateways import ISSUERS, Outcome, RoutingContext, default_fleet
from src.routers import ThompsonRouter

from . import metrics
from .events import ClickHouseSink, EventPipeline, KafkaSink, RoutingEvent
from .state import PosteriorSnapshot, RedisConstraintStore, connect

try:  # FastAPI is optional; importing this module must not require it.
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only without FastAPI
    raise ImportError(
        "the service layer needs FastAPI: pip install -r requirements-service.txt"
    ) from exc


class RouteRequest(BaseModel):
    issuer: str = Field(..., description="Issuing bank, e.g. HDFC.")
    amount_paise: int = Field(0, ge=0, description="Amount; recorded, not yet routed on.")
    transaction_id: str | None = None


class RouteResponse(BaseModel):
    transaction_id: str
    gateway: str
    propensity: float = Field(..., description="P(this gateway | context) under the live policy.")
    blocked: list[str] = Field(default_factory=list)
    decision_micros: float


class OutcomeReport(BaseModel):
    transaction_id: str
    gateway: str
    success: bool
    latency_ms: float = 0.0
    issuer: str = "UNKNOWN"
    propensity: float = 1.0


class RouterService:
    """Holds the live policy and everything attached to it."""

    def __init__(self) -> None:
        self.specs = default_fleet()
        self.names = [s.name for s in self.specs]
        self.router = ThompsonRouter(len(self.specs), gamma=0.999, seed=0)
        self.started = time.time()
        self.decisions = 0
        # Simulated minutes added on top of the wall clock. Production leaves
        # this at zero. The demo endpoints advance it, because they replay
        # hours of traffic in milliseconds and everything downstream -- the
        # investigation window, constraint TTLs, baseline-vs-current
        # comparisons -- is expressed in minutes and would otherwise collapse
        # onto a single tick.
        self.clock_offset = 0
        # FastAPI runs `def` (non-async) handlers in a threadpool, so requests
        # mutate this object concurrently. The posterior update, the five
        # history appends and the decision counter are each read-modify-write
        # sequences; interleaved, they can leave the history arrays at
        # different lengths and silently mis-pair ticks with outcomes. A lock
        # costs ~100ns against a 35us decision.
        self._lock = threading.Lock()
        # A bounded ring of routed transactions. The agent reads this; without
        # it /investigate has nothing to investigate. Capped because the hot
        # path must not grow memory without limit.
        self.history_limit = int(os.environ.get("ROUTER_HISTORY_LIMIT", "200000"))
        self._hist_tick: list[int] = []
        self._hist_gateway: list[int] = []
        self._hist_success: list[bool] = []
        self._hist_latency: list[float] = []
        self._hist_issuer: list[int] = []
        # A dedicated generator: propensities must never consume the router's
        # own randomness, or measuring changes what is measured.
        self.propensity_rng = np.random.default_rng(12345)
        # More samples means a finer propensity, which off-policy evaluation
        # divides by. 64 gives a resolution of ~1.5%; the floor matters more
        # than the precision, so this is tunable rather than fixed.
        self.propensity_samples = int(os.environ.get("ROUTER_PROPENSITY_SAMPLES", "64"))

        self.redis = connect(os.environ.get("REDIS_URL"))
        self.constraints: ConstraintStore = (
            RedisConstraintStore(self.names, self.redis)
            if self.redis is not None else ConstraintStore(self.names)
        )
        self.snapshot = PosteriorSnapshot(self.redis)
        self.restored = self.snapshot.restore(self.router)

        self.events = EventPipeline(
            KafkaSink(os.environ.get("KAFKA_BROKERS"),
                      os.environ.get("KAFKA_TOPIC", "routing.outcomes")),
            ClickHouseSink(os.environ.get("CLICKHOUSE_URL"),
                           os.environ.get("CLICKHOUSE_TABLE", "routing_outcomes")),
        )
        if self.events.clickhouse.enabled:
            self.events.clickhouse.ensure_schema()

    @property
    def tick(self) -> int:
        """Minutes since start, the clock constraints are expressed in."""
        return int((time.time() - self.started) / 60) + self.clock_offset

    def route(self, issuer: str) -> tuple[int, float, list[str], float]:
        context = RoutingContext(tick=self.tick, issuer=issuer)
        with metrics.decision_duration.time():
            started = time.perf_counter()
            blocked = self.constraints.blocked(context)
            with self._lock:
                # One batched draw serves the decision and its propensity.
                # This was select() followed by a separate
                # action_probabilities(), where the propensity pass measured
                # 2.1x the decision itself and -- worse -- sat outside the
                # timed region, so the latency reported back understated the
                # real cost of a request by roughly 3x. Both are inside now.
                gateway, probabilities = self.router.decide(
                    context.tick, context, blocked,
                    n_samples=self.propensity_samples, rng=self.propensity_rng,
                )
            elapsed = time.perf_counter() - started

        if blocked:
            metrics.constraint_blocks_total.inc(issuer=issuer)
        metrics.constraints_active.set(len(self.constraints.active(context.tick)))
        with self._lock:
            self.decisions += 1
        return gateway, float(probabilities[gateway]), [self.names[i] for i in blocked], elapsed

    def record(self, report: OutcomeReport) -> None:
        try:
            index = self.names.index(report.gateway)
        except ValueError:
            raise HTTPException(404, f"unknown gateway {report.gateway!r}")
        with self._lock:
            self.router.update(Outcome(
                gateway=index, success=report.success, latency_ms=report.latency_ms,
                tick=self.tick, issuer=report.issuer,
            ))
            self._append_history(index, report)
        metrics.decisions_total.inc(
            gateway=report.gateway, outcome="success" if report.success else "failure"
        )
        self.events.publish(RoutingEvent(
            transaction_id=report.transaction_id, tick=self.tick, issuer=report.issuer,
            gateway=report.gateway, success=report.success, latency_ms=report.latency_ms,
            propensity=report.propensity, policy="thompson-0.999",
        ))
        self.snapshot.maybe_save(self.router)

    def _append_history(self, gateway: int, report: "OutcomeReport") -> None:
        issuer = report.issuer.upper()
        self._hist_tick.append(self.tick)
        self._hist_gateway.append(gateway)
        self._hist_success.append(bool(report.success))
        self._hist_latency.append(float(report.latency_ms))
        self._hist_issuer.append(ISSUERS.index(issuer) if issuer in ISSUERS else 0)
        if len(self._hist_tick) > self.history_limit:
            drop = len(self._hist_tick) - self.history_limit
            del self._hist_tick[:drop], self._hist_gateway[:drop]
            del self._hist_success[:drop], self._hist_latency[:drop], self._hist_issuer[:drop]

    def history(self):
        """Routed transactions so far, in the shape TelemetryStore expects.

        Copied under the lock. A reader that snapshots these five lists while a
        writer is midway through appending would get arrays of different
        lengths, and RunResult would then silently mis-pair ticks with outcomes.
        """
        from src.metrics import RunResult

        with self._lock:
            ticks = list(self._hist_tick)
            gateways = list(self._hist_gateway)
            successes = list(self._hist_success)
            latencies = list(self._hist_latency)
            issuers = list(self._hist_issuer)
        n = len(ticks)
        if n == 0:
            return None
        return RunResult(
            router="live", chosen=np.array(gateways),
            success=np.array(successes), latency_ms=np.array(latencies),
            instant_regret=np.zeros(n), cost_bps=np.zeros(n), n_gateways=len(self.specs),
            tick=np.array(ticks), issuer=np.array(issuers),
        )


SERVICE: RouterService | None = None


def set_service(service: "RouterService") -> None:
    """Replace the live service. Used by /simulate/reset to start a demo over."""
    global SERVICE
    SERVICE = service


@asynccontextmanager
async def lifespan(app):
    global SERVICE
    SERVICE = RouterService()
    yield
    if SERVICE is not None:
        SERVICE.events.flush()
        SERVICE.snapshot.maybe_save(SERVICE.router, force=True)


app = FastAPI(
    title="Payment Routing Optimizer",
    description="Adaptive gateway routing with an LLM diagnostic agent.",
    lifespan=lifespan,
)

# Demo endpoints live under /simulate so it is obvious which routes exist for
# the walkthrough and which are the service itself.
from .demo import router as demo_router  # noqa: E402  (needs `app` defined first)

app.include_router(demo_router)

_UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/", include_in_schema=False)
def index():
    """Serve the walkthrough UI, or say plainly that it is missing.

    The UI is a single static file with no build step. If it is absent the
    service is unaffected -- everything it does is available over the API.
    """
    from fastapi.responses import FileResponse, JSONResponse

    page = os.path.join(_UI_DIR, "index.html")
    if not os.path.exists(page):
        return JSONResponse({
            "service": "payment-routing-optimizer",
            "ui": "not installed (service/static/index.html is missing)",
            "docs": "/docs",
        })
    return FileResponse(page)


def _service() -> RouterService:
    if SERVICE is None:  # pragma: no cover - only outside the lifespan
        raise HTTPException(503, "service not started")
    return SERVICE


@app.get("/health")
def health() -> dict[str, Any]:
    svc = _service()
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - svc.started, 1),
        "decisions": svc.decisions,
        "gateways": svc.names,
        "backends": {
            "redis": svc.redis is not None,
            "posterior_restored_from_snapshot": svc.restored,
            "kafka": svc.events.kafka.enabled,
            "clickhouse": svc.events.clickhouse.enabled,
        },
    }


@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest) -> RouteResponse:
    svc = _service()
    if request.issuer.upper() not in ISSUERS:
        raise HTTPException(400, f"unknown issuer; expected one of {list(ISSUERS)}")
    gateway, propensity, blocked, elapsed = svc.route(request.issuer.upper())
    return RouteResponse(
        transaction_id=request.transaction_id or uuid.uuid4().hex,
        gateway=svc.names[gateway],
        propensity=round(propensity, 4),
        blocked=blocked,
        decision_micros=round(elapsed * 1_000_000, 1),
    )


@app.post("/outcome")
def outcome(report: OutcomeReport) -> dict[str, str]:
    _service().record(report)
    return {"status": "recorded"}


@app.get("/constraints")
def constraints() -> dict[str, Any]:
    svc = _service()
    active = svc.constraints.active(svc.tick)
    return {
        "active": [
            {"gateway": c.gateway, "issuer": c.issuer, "expires_tick": c.expires_tick,
             "confidence": c.confidence, "reason": c.reason}
            for c in active
        ],
        "stats": svc.constraints.stats(),
    }


@app.post("/investigate")
def investigate(alert: str = "Conversion is down.", lookback_minutes: int = 60) -> dict[str, Any]:
    """Cold path. Runs the agent, and installs a constraint if it earns one.

    Separate from /route on purpose: this takes seconds and calls a model.
    """
    from src.agent import Investigator, InvestigatorConfig, MemoryStore
    from src.agent.llm import default_client
    from src.agent.telemetry import TelemetryStore
    from src.agent.traces import TraceStore

    svc = _service()
    history = svc.history()
    if history is None:
        raise HTTPException(
            409,
            "no telemetry yet: this endpoint reads routed transactions, and none "
            "have been recorded. POST /route and /outcome first.",
        )
    window = (max(0, svc.tick - lookback_minutes), svc.tick)
    traces = TraceStore(settings().traces)
    investigator = Investigator(
        store=TelemetryStore(history, svc.specs), client=default_client(),
        memory=MemoryStore(), config=InvestigatorConfig(), traces=traces,
    )
    try:
        with metrics.investigation_duration.time():
            result = investigator.investigate(alert, window)
    finally:
        # Langfuse batches in a background thread and Postgres holds a
        # connection; leaking one per investigation accumulates silently.
        traces.close()
    metrics.investigations_total.inc(stop_reason=result.stop_reason)
    metrics.investigation_cost.inc(
        result.usage.cost_usd(settings().model.input_usd_per_mtok,
                              settings().model.output_usd_per_mtok)
    )

    installed = None
    if result.diagnosis is not None:
        constraints_cfg = settings().constraints
        constraint = from_diagnosis(
            result.diagnosis, svc.tick, svc.names,
            ttl_minutes=constraints_cfg.ttl_minutes,
            min_confidence=constraints_cfg.min_confidence,
            canary_rate=constraints_cfg.canary_rate,
        )
        if constraint is not None:
            svc.constraints.add(constraint)
            installed = constraint.describe()
    return {
        "stop_reason": result.stop_reason,
        "diagnosis": result.diagnosis.to_dict() if result.diagnosis else None,
        "tool_calls": result.usage.tool_calls,
        "constraint_installed": installed,
    }


@app.get("/metrics")
def prometheus_metrics():
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")
