"""Detect -> investigate -> constrain -> route, inside one simulation.

`run_once` is open-loop: the router's decisions never change the fleet and the
run is fully determined before it starts. This module closes that. The
simulation pauses at a detection tick, hands the agent the telemetry generated
so far, and installs whatever constraint the diagnosis justifies. Every
subsequent transaction is routed under it.

That ordering is the whole point, and it is what makes the measurement honest:
the agent sees only what had actually happened by the moment it was called, and
its output changes what happens next. No lookahead, no oracle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .agent.constraints import ConstrainedRouter, ConstraintStore, RoutingConstraint, from_diagnosis
from .agent.schemas import Diagnosis, InvestigationResult
from .agent.telemetry import TelemetryStore
from .gateways import ISSUER_MIX, ISSUERS, GatewayEnvironment, GatewaySpec, Outcome, RoutingContext
from .metrics import RunResult
from .simulator import SimConfig

# How far back the agent looks when it is woken. Short on purpose: the incident
# is recent, and a long window dilutes it with healthy pre-incident traffic.
LOOKBACK_MINUTES = 60


@dataclass
class ClosedLoopResult:
    run: RunResult
    constraints: list[RoutingConstraint] = field(default_factory=list)
    investigations: list[InvestigationResult] = field(default_factory=list)
    store_stats: dict[str, Any] = field(default_factory=dict)

    def sr_between(self, start_tick: int, end_tick: int, issuer: str | None = None) -> float | None:
        """Realised success rate over a tick window, optionally for one issuer."""
        tick = np.asarray(self.run.tick)
        mask = (tick >= start_tick) & (tick < end_tick)
        if issuer is not None:
            mask &= np.asarray(self.run.issuer) == ISSUERS.index(issuer)
        if mask.sum() == 0:
            return None
        return float(np.asarray(self.run.success)[mask].mean())

    def share_to(
        self, gateway_index: int, start_tick: int, end_tick: int, issuer: str | None = None
    ) -> float | None:
        tick = np.asarray(self.run.tick)
        mask = (tick >= start_tick) & (tick < end_tick)
        if issuer is not None:
            mask &= np.asarray(self.run.issuer) == ISSUERS.index(issuer)
        if mask.sum() == 0:
            return None
        return float((np.asarray(self.run.chosen)[mask] == gateway_index).mean())


def _partial_result(
    specs: list[GatewaySpec],
    chosen: np.ndarray,
    success: np.ndarray,
    latency: np.ndarray,
    ticks: np.ndarray,
    issuers: np.ndarray,
    upto: int,
) -> RunResult:
    """A RunResult over the first ``upto`` transactions.

    The agent is handed exactly this -- nothing after the detection point --
    so it cannot accidentally diagnose from the future.
    """
    return RunResult(
        router="partial",
        chosen=chosen[:upto],
        success=success[:upto],
        latency_ms=latency[:upto],
        instant_regret=np.zeros(upto),
        cost_bps=np.zeros(upto),
        n_gateways=len(specs),
        tick=ticks[:upto],
        issuer=issuers[:upto],
    )


def run_closed_loop(
    specs: list[GatewaySpec],
    config: SimConfig,
    router_factory: Callable[[int, int], Any],
    seed: int,
    detect_ticks: tuple[int, ...] = (),
    investigate: Callable[[TelemetryStore, tuple[int, int]], InvestigationResult] | None = None,
    ttl_minutes: int = 480,
    canary_rate: float = 0.02,
    alert_template: str = "Conversion is down; the router flagged a calibration break at minute {tick}.",
) -> ClosedLoopResult:
    """Run a simulation that investigates and constrains itself mid-flight.

    ``detect_ticks`` are the moments the controller wakes the agent. More than
    one is realistic: constraints expire, and if the underlying fault is still
    live the next investigation has to re-establish it.
    """
    env = GatewayEnvironment(specs, seed=seed)
    names = [s.name for s in specs]
    n, n_ticks = config.transactions, config.n_ticks

    sr = np.array(
        [[[s.true_sr(t, iss) for iss in ISSUERS] for t in range(n_ticks)] for s in specs]
    )
    base_latency = np.array([s.base_latency_ms for s in specs])
    jitter = np.array([s.latency_jitter_ms for s in specs])
    latency_mult = np.ones((len(specs), n_ticks))
    for i, spec in enumerate(specs):
        for event in spec.events:
            lo, hi = max(0, event.start), min(n_ticks, event.end)
            latency_mult[i, lo:hi] *= event.latency_multiplier
    costs = np.array([s.cost_bps for s in specs])

    ticks = (np.arange(n) * n_ticks // n).astype(int)
    draws = env.rng.random(n)
    noise = env.rng.standard_normal(n)
    issuers = np.random.default_rng(seed + 10_000).choice(
        len(ISSUERS), size=n, p=np.asarray(ISSUER_MIX)
    )

    store = ConstraintStore(names, seed=seed)
    router = ConstrainedRouter(router_factory(len(specs), seed), store)

    chosen = np.empty(n, dtype=int)
    success = np.empty(n, dtype=bool)
    latency = np.empty(n, dtype=float)

    pending = sorted(detect_ticks)
    constraints: list[RoutingConstraint] = []
    investigations: list[InvestigationResult] = []

    for i in range(n):
        t = int(ticks[i])

        # Wake the agent once the clock passes a detection point.
        while pending and t >= pending[0] and investigate is not None and i > 0:
            detect_at = pending.pop(0)
            window = (max(0, detect_at - LOOKBACK_MINUTES), detect_at)
            telemetry = TelemetryStore(
                _partial_result(specs, chosen, success, latency, ticks, issuers, i), specs
            )
            result = investigate(telemetry, window)
            investigations.append(result)
            if result.diagnosis is not None:
                constraint = from_diagnosis(
                    result.diagnosis, detect_at, names,
                    ttl_minutes=ttl_minutes, canary_rate=canary_rate,
                )
                if constraint is not None:
                    store.add(constraint)
                    constraints.append(constraint)

        context = RoutingContext(tick=t, issuer=ISSUERS[issuers[i]])
        g = router.select(t, context)
        ok = bool(draws[i] < sr[g, t, issuers[i]])
        lat = max(1.0, base_latency[g] * latency_mult[g, t] + jitter[g] * noise[i])

        chosen[i] = g
        success[i] = ok
        latency[i] = lat
        router.update(Outcome(
            gateway=g, success=ok, latency_ms=lat, tick=t,
            issuer=ISSUERS[issuers[i]],
        ))

    oracle = sr.max(axis=0)
    return ClosedLoopResult(
        run=RunResult(
            router=router.name,
            chosen=chosen, success=success, latency_ms=latency,
            instant_regret=oracle[ticks, issuers] - sr[chosen, ticks, issuers],
            cost_bps=costs[chosen], n_gateways=len(specs),
            tick=ticks, issuer=issuers,
        ),
        constraints=constraints,
        investigations=investigations,
        store_stats=store.stats(),
    )
