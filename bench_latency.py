#!/usr/bin/env python3
"""Routing decision latency, measured rather than assumed.

    python bench_latency.py
    python bench_latency.py --n 200000 --constraints 3

A routing decision sits on the hot path: it runs once per payment, and its
budget is measured in milliseconds. Every other number in this project is about
whether the router chooses *well*; this one is about whether it chooses *fast
enough to be allowed to choose at all*.

What is timed is the decision only -- resolve active constraints, then select a
gateway. Not the HTTP layer, not the posterior update, not event publishing,
because those are either off the critical path or someone else's budget.

Percentiles, not the mean. A mean hides the tail, and the tail is what times
out.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from src.agent.constraints import ConstraintStore, RoutingConstraint
from src.gateways import ISSUERS, Outcome, RoutingContext, default_fleet
from src.routers import (
    ContextualThompsonRouter,
    EpsilonGreedyRouter,
    StaticWeightedRouter,
    ThompsonRouter,
    UCB1Router,
)

BUDGET_US = 100_000.0  # 100 ms, the figure the JusTrust brief quotes


def build_store(names: list[str], n_constraints: int, seed: int = 0) -> ConstraintStore:
    """A store holding `n_constraints` live constraints, to price the lookup."""
    store = ConstraintStore(names, seed=seed)
    pairs = [("PG-Delta", "HDFC"), ("PG-Bravo", "ICICI"), ("PG-Alpha", None),
             ("PG-Charlie", "SBI"), ("PG-Echo", "AXIS")]
    for gateway, issuer in pairs[:n_constraints]:
        store.add(RoutingConstraint(
            gateway=gateway, issuer=issuer, reason="benchmark",
            created_tick=0, expires_tick=10**9, confidence=0.8,
            source_scope="issuer_specific" if issuer else "single_gateway",
            canary_rate=0.02,
        ))
    return store


def measure(router, store, n: int, warmup: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for t in range(warmup):
        context = RoutingContext(tick=t, issuer=ISSUERS[t % len(ISSUERS)])
        gateway = router.select(t, context, store.blocked(context))
        router.update(Outcome(gateway=gateway, success=True, latency_ms=1.0,
                              tick=t, issuer=context.issuer))

    samples = np.empty(n, dtype=float)
    for i in range(n):
        issuer = ISSUERS[i % len(ISSUERS)]
        context = RoutingContext(tick=i, issuer=issuer)
        started = time.perf_counter()
        blocked = store.blocked(context)
        gateway = router.select(i, context, blocked)
        samples[i] = (time.perf_counter() - started) * 1e6
        # Outside the timed region: updating is not part of the decision.
        router.update(Outcome(gateway=gateway, success=bool(rng.random() < 0.93),
                              latency_ms=1.0, tick=i, issuer=issuer))
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50_000, help="timed decisions per router")
    ap.add_argument("--warmup", type=int, default=2_000)
    ap.add_argument("--constraints", type=int, default=1,
                    help="active constraints during the run (0-5)")
    args = ap.parse_args()

    specs = default_fleet()
    names = [s.name for s in specs]
    n = len(specs)
    routers = [
        ("static-weighted", StaticWeightedRouter(n, seed=0)),
        ("epsilon-greedy", EpsilonGreedyRouter(n, epsilon=0.1, gamma=0.999, seed=0)),
        ("ucb1", UCB1Router(n, gamma=0.999, seed=0)),
        ("thompson", ThompsonRouter(n, gamma=0.999, seed=0)),
        ("contextual-thompson", ContextualThompsonRouter(n, gamma=0.999, seed=0)),
    ]

    print(f"{args.n:,} timed decisions per router, {args.warmup:,} warm-up, "
          f"{args.constraints} active constraint(s)")
    print(f"budget: {BUDGET_US:,.0f} us (100 ms)\n")
    print(f"{'router':22s} {'p50':>9s} {'p95':>9s} {'p99':>9s} {'max':>10s}  {'vs budget':>12s}")
    print("-" * 78)

    worst = 0.0
    for label, router in routers:
        store = build_store(names, args.constraints)
        samples = measure(router, store, args.n, args.warmup)
        p50, p95, p99 = np.percentile(samples, [50, 95, 99])
        headroom = BUDGET_US / p99
        worst = max(worst, p99)
        print(f"{label:22s} {p50:>8.1f}u {p95:>8.1f}u {p99:>8.1f}u "
              f"{samples.max():>9.1f}u  {headroom:>10,.0f}x")

    print("-" * 78)
    print(f"\nSlowest p99 across all routers: {worst:.1f} us "
          f"({BUDGET_US / worst:,.0f}x inside the 100 ms budget).")
    print("\nTwo caveats worth stating rather than burying:")
    print("  * This is the decision only. Network, TLS, serialisation and the")
    print("    downstream gateway call dominate any real end-to-end budget.")
    print("  * CPython on one core. The point is not that Python is fast enough")
    print("    at 350M/day -- it is that the *algorithm* costs microseconds, so")
    print("    a Go or Rust port is a transport decision, not an algorithmic one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
