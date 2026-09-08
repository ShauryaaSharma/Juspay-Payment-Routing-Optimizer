"""Runs routers against the gateway fleet and collects per-transaction records.

Transactions and wall-clock are decoupled on purpose. Gateway health drifts on
a clock (ticks == simulated minutes), but a router's learning rate depends on
transaction *volume*. Holding the calendar fixed while varying volume is how
you find out whether a strategy works at 10k transactions/day or only at 10M.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .gateways import (
    ISSUER_MIX,
    ISSUERS,
    GatewayEnvironment,
    GatewaySpec,
    Outcome,
    TICKS_PER_DAY,
    default_fleet,
)
from .metrics import RunResult

RouterFactory = Callable[[int, int], "object"]  # (n_gateways, seed) -> Router


@dataclass
class SimConfig:
    days: int = 7
    transactions: int = 100_000
    seeds: Sequence[int] = (0, 1, 2)

    @property
    def n_ticks(self) -> int:
        return self.days * TICKS_PER_DAY


def run_once(
    router_factory: RouterFactory,
    specs: list[GatewaySpec],
    config: SimConfig,
    seed: int,
) -> RunResult:
    env = GatewayEnvironment(specs, seed=seed)
    router = router_factory(env.n_gateways, seed)

    n = config.transactions
    n_ticks = config.n_ticks
    n_issuers = len(ISSUERS)

    # Ground truth is a pure function of (tick, issuer), so evaluate it once
    # per cell rather than once per transaction.
    sr = np.array(
        [[[s.true_sr(t, iss) for iss in ISSUERS] for t in range(n_ticks)] for s in specs]
    )  # (n_gateways, n_ticks, n_issuers)
    oracle = sr.max(axis=0)  # (n_ticks, n_issuers)
    base_latency = np.array([s.base_latency_ms for s in specs])
    jitter = np.array([s.latency_jitter_ms for s in specs])
    latency_mult = np.ones((env.n_gateways, n_ticks))
    for i, spec in enumerate(specs):
        for event in spec.events:
            lo, hi = max(0, event.start), min(n_ticks, event.end)
            latency_mult[i, lo:hi] *= event.latency_multiplier
    costs = np.array([s.cost_bps for s in specs])

    ticks = (np.arange(n) * n_ticks // n).astype(int)
    draws = env.rng.random(n)
    noise = env.rng.standard_normal(n)
    # Issuers come from a separate stream so that adding this dimension does
    # not perturb the outcome draws -- the routing benchmark stays comparable
    # to runs made before issuers existed.
    issuers = np.random.default_rng(seed + 10_000).choice(
        n_issuers, size=n, p=np.asarray(ISSUER_MIX)
    )

    chosen = np.empty(n, dtype=int)
    success = np.empty(n, dtype=bool)
    latency = np.empty(n, dtype=float)

    for i in range(n):
        t = int(ticks[i])
        g = router.select(t)
        ok = bool(draws[i] < sr[g, t, issuers[i]])
        lat = max(1.0, base_latency[g] * latency_mult[g, t] + jitter[g] * noise[i])

        chosen[i] = g
        success[i] = ok
        latency[i] = lat
        router.update(Outcome(gateway=g, success=ok, latency_ms=lat, tick=t))

    return RunResult(
        router=getattr(router, "name", "unknown"),
        chosen=chosen,
        success=success,
        latency_ms=latency,
        instant_regret=oracle[ticks, issuers] - sr[chosen, ticks, issuers],
        cost_bps=costs[chosen],
        n_gateways=env.n_gateways,
        tick=ticks,
        issuer=issuers,
    )


def run_many(
    router_factory: RouterFactory,
    specs: list[GatewaySpec] | None = None,
    config: SimConfig | None = None,
) -> list[RunResult]:
    """One run per seed. Seed drives both the environment and the router."""
    specs = specs if specs is not None else default_fleet()
    config = config or SimConfig()
    return [run_once(router_factory, specs, config, seed) for seed in config.seeds]


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())
