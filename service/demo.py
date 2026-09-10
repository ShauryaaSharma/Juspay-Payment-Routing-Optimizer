"""Demo endpoints: drive real traffic through the live router, on demand.

The rest of the service expects a caller to route a transaction and then report
what happened to it. That is the right shape for production and a poor shape for
a demonstration, because there is no caller.

These endpoints close that gap by playing both sides: they route through the
**actual** bandit, resolve the outcome against the **actual** simulator, and
feed it back through the **actual** update path. Nothing here is a mock or a
replay — the numbers the UI shows are the router's real numbers, and the agent
investigates telemetry those decisions actually produced.

`/simulate/incident` plants an issuer-scoped failure at runtime, which is what
makes the loop visible end to end: break PG-Delta for HDFC, watch conversion
fall, run an investigation, watch a scoped constraint appear and the traffic
move. That sequence is the whole project in about thirty seconds.

Kept in its own module and mounted under `/simulate` so it is obvious which
endpoints exist for the demo and which are the service.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import numpy as np

from src.gateways import ISSUER_MIX, ISSUERS, DegradationEvent

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError("the demo layer needs FastAPI") from exc

router = APIRouter(prefix="/simulate", tags=["demo"])

# A long horizon: an injected incident lasts until it is explicitly cleared.
FOREVER = 10**9

# Transactions per simulated minute, matching the offline simulator's implicit
# rate (100k transactions over seven days). A batch of 600 therefore spans an
# hour of simulated time, which is what gives the investigation window, the
# constraint TTLs and the diurnal pattern something to range over. Without it
# the whole demo lands on one tick and every windowed query comes back empty.
TX_PER_TICK = 10

# One demo action at a time. `clock_offset += 1` is a read-modify-write, and two
# batches interleaving would advance the simulated clock by less than the
# transactions they routed -- besides which a half-injected incident is not a
# thing anyone wants to look at. This is separate from RouterService's own lock,
# which route() and record() take for themselves; taking that one here would
# deadlock.
_DEMO_LOCK = threading.Lock()


class TrafficRequest(BaseModel):
    count: int = Field(200, ge=1, le=20000, description="Transactions to route.")
    issuer: str | None = Field(None, description="Force one issuer; default is the real mix.")


class IncidentRequest(BaseModel):
    gateway: str = Field(..., description="Gateway that starts failing.")
    issuer: str | None = Field(None, description="Scope to one issuer; null means all traffic.")
    severity: float = Field(
        0.08, gt=0.0, le=1.0,
        description="Success-rate multiplier. 0.08 means it declines ~92% of what it accepted.",
    )


def _service():
    from .api import SERVICE

    if SERVICE is None:  # pragma: no cover
        raise HTTPException(503, "service not started")
    return SERVICE


@router.post("/traffic")
def generate_traffic(request: TrafficRequest) -> dict[str, Any]:
    """Route `count` transactions and resolve each against the simulated fleet.

    Deliberately runs the real path: `router.decide` for the choice and its
    propensity, the real constraint layer, the real update. The only thing
    standing in for production is the gateway itself, which is what the
    simulator has always been.
    """
    with _DEMO_LOCK:
        return _generate_traffic(request)


def _generate_traffic(request: TrafficRequest) -> dict[str, Any]:
    svc = _service()
    if request.issuer and request.issuer.upper() not in ISSUERS:
        raise HTTPException(400, f"unknown issuer; expected one of {list(ISSUERS)}")

    rng = np.random.default_rng()
    issuer_indices = (
        np.full(request.count, ISSUERS.index(request.issuer.upper()))
        if request.issuer
        else rng.choice(len(ISSUERS), size=request.count, p=np.asarray(ISSUER_MIX))
    )
    draws = rng.random(request.count)

    # Advance the simulated clock with the batch so the transactions land on a
    # time axis rather than all at once.
    start_tick = svc.tick
    successes = 0
    per_gateway: dict[str, dict[str, int]] = {
        name: {"routed": 0, "succeeded": 0} for name in svc.names
    }
    per_issuer: dict[str, dict[str, int]] = {
        name: {"routed": 0, "succeeded": 0} for name in ISSUERS
    }
    blocked_decisions = 0
    started = time.perf_counter()

    for i in range(request.count):
        svc.clock_offset += (i + 1) // TX_PER_TICK - i // TX_PER_TICK
        issuer = ISSUERS[int(issuer_indices[i])]
        gateway_index, propensity, blocked, _ = svc.route(issuer)
        gateway = svc.names[gateway_index]
        if blocked:
            blocked_decisions += 1

        # Ground truth from the simulator -- the one thing a real deployment
        # would get from the gateway instead.
        true_sr = svc.specs[gateway_index].true_sr(svc.tick, issuer)
        success = bool(draws[i] < true_sr)

        from .api import OutcomeReport

        svc.record(OutcomeReport(
            transaction_id=uuid.uuid4().hex, gateway=gateway, success=success,
            latency_ms=float(svc.specs[gateway_index].true_latency_ms(svc.tick, issuer)),
            issuer=issuer, propensity=propensity,
        ))

        successes += success
        per_gateway[gateway]["routed"] += 1
        per_gateway[gateway]["succeeded"] += success
        per_issuer[issuer]["routed"] += 1
        per_issuer[issuer]["succeeded"] += success

    elapsed = time.perf_counter() - started
    return {
        "routed": request.count,
        "minutes_simulated": svc.tick - start_tick,
        "success_rate": round(successes / request.count, 4),
        "blocked_decisions": blocked_decisions,
        "wall_seconds": round(elapsed, 3),
        "per_gateway": [
            {
                "gateway": name,
                "routed": stats["routed"],
                "share": round(stats["routed"] / request.count, 4),
                "success_rate": (
                    round(stats["succeeded"] / stats["routed"], 4) if stats["routed"] else None
                ),
            }
            for name, stats in per_gateway.items()
        ],
        "per_issuer": [
            {
                "issuer": name,
                "routed": stats["routed"],
                "success_rate": (
                    round(stats["succeeded"] / stats["routed"], 4) if stats["routed"] else None
                ),
            }
            for name, stats in per_issuer.items()
        ],
    }


class AttemptRequest(BaseModel):
    gateway: str = Field(..., description="Gateway to attempt the payment on.")
    issuer: str = Field(..., description="Issuing bank of the card or account.")


@router.post("/attempt")
def attempt(request: AttemptRequest) -> dict[str, Any]:
    """Stand in for the acquiring network: attempt one payment, report what happened.

    This is the single seam between the demo and reality. A merchant backend
    calls `/route` to be told where to send a payment, calls *this* where it
    would call the actual gateway, and reports the result to `/outcome`. Every
    other part of that path is the real thing.

    It is exposed by the router only because the router owns the simulated
    fleet and the injected faults; nothing about routing depends on it, and in
    production it is the one endpoint that would not exist.

    No lock: this only reads, and `spec.events` is replaced by rebinding rather
    than mutated in place, so a concurrent injection swaps one list for another
    rather than leaving a torn one behind.
    """
    svc = _service()
    if request.gateway not in svc.names:
        raise HTTPException(404, f"unknown gateway {request.gateway!r}")
    issuer = request.issuer.upper()
    if issuer not in ISSUERS:
        raise HTTPException(400, f"unknown issuer; expected one of {list(ISSUERS)}")

    spec = svc.specs[svc.names.index(request.gateway)]
    tick = svc.tick
    success = bool(np.random.default_rng().random() < spec.true_sr(tick, issuer))
    return {
        "gateway": spec.name,
        "issuer": issuer,
        "success": success,
        "latency_ms": round(float(spec.true_latency_ms(tick, issuer)), 1),
        "failure_reason": None if success else "declined by issuing bank",
    }


@router.post("/incident")
def inject_incident(request: IncidentRequest) -> dict[str, Any]:
    """Break a gateway, optionally for one issuer only.

    An issuer-scoped incident is the interesting case: the gateway's *aggregate*
    success rate only sags, so it looks merely soft, and only segmentation
    reveals that one bank is being declined outright.
    """
    with _DEMO_LOCK:
        return _inject_incident(request)


def _inject_incident(request: IncidentRequest) -> dict[str, Any]:
    svc = _service()
    if request.gateway not in svc.names:
        raise HTTPException(404, f"unknown gateway {request.gateway!r}")
    issuer = request.issuer.upper() if request.issuer else None
    if issuer and issuer not in ISSUERS:
        raise HTTPException(400, f"unknown issuer; expected one of {list(ISSUERS)}")

    spec = svc.specs[svc.names.index(request.gateway)]
    spec.events = [
        event for event in spec.events if not str(event.label).startswith("demo:")
    ]
    spec.events.append(DegradationEvent(
        start=svc.tick, end=FOREVER, sr_multiplier=request.severity,
        issuer=issuer, label=f"demo:{request.gateway}:{issuer or 'ALL'}",
    ))
    return {
        "status": "incident injected",
        "gateway": request.gateway,
        "issuer": issuer,
        "severity": request.severity,
        "note": (
            "Aggregate success rate on this gateway will only sag, because the "
            "other issuers still convert. Segmenting is what reveals it."
            if issuer else
            "This gateway now fails for all traffic."
        ),
    }


@router.post("/incident/clear")
def clear_incidents() -> dict[str, Any]:
    """Heal every injected gateway. Constraints are left alone deliberately.

    A constraint outliving the fault it was written for is the real behaviour,
    not a bug to paper over -- it expires on its TTL, and the canary traffic is
    what lets the router notice the recovery in the meantime.
    """
    with _DEMO_LOCK:
        return _clear_incidents()


def _clear_incidents() -> dict[str, Any]:
    svc = _service()
    cleared = 0
    for spec in svc.specs:
        before = len(spec.events)
        spec.events = [
            event for event in spec.events if not str(event.label).startswith("demo:")
        ]
        cleared += before - len(spec.events)
    return {"status": "cleared", "incidents_removed": cleared}


@router.get("/state")
def state() -> dict[str, Any]:
    """Everything the UI needs in one call, so it polls once rather than five times."""
    svc = _service()
    tick = svc.tick
    posterior = svc.router.estimated_sr()
    active = svc.constraints.active(tick)

    blocked_by_gateway: dict[str, list[str]] = {name: [] for name in svc.names}
    for constraint in active:
        if constraint.gateway in blocked_by_gateway:
            blocked_by_gateway[constraint.gateway].append(constraint.issuer or "ALL")

    injected = [
        {
            "gateway": spec.name,
            "issuer": event.issuer or "ALL",
            "severity": event.sr_multiplier,
        }
        for spec in svc.specs
        for event in spec.events
        if str(event.label).startswith("demo:")
    ]

    history = svc.history()
    recent_sr = None
    if history is not None and history.success.size:
        window = history.success[-2000:]
        recent_sr = round(float(window.mean()), 4)

    return {
        "tick": tick,
        "decisions": svc.decisions,
        "recent_success_rate": recent_sr,
        "gateways": [
            {
                "name": name,
                "posterior_sr": round(float(posterior[i]), 4),
                "blocked_for": blocked_by_gateway[name],
                # What the UI shows as "broken" -- the ground truth the router
                # cannot see, revealed here only because this is a demo.
                "injected_fault": next(
                    (f for f in injected if f["gateway"] == name), None
                ),
            }
            for i, name in enumerate(svc.names)
        ],
        "issuers": list(ISSUERS),
        "constraints": [
            {
                "gateway": c.gateway, "issuer": c.issuer, "expires_tick": c.expires_tick,
                "confidence": c.confidence, "canary_rate": c.canary_rate,
                "reason": c.reason,
            }
            for c in active
        ],
        "constraint_stats": svc.constraints.stats(),
        "backends": {
            "redis": svc.redis is not None,
            "kafka": svc.events.kafka.enabled,
            "clickhouse": svc.events.clickhouse.enabled,
        },
    }


@router.post("/reset")
def reset() -> dict[str, Any]:
    """Fresh router, no constraints, no faults. For starting a demo over."""
    from .api import RouterService, set_service

    with _DEMO_LOCK:
        set_service(RouterService())
    return {"status": "reset"}
