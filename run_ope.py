#!/usr/bin/env python3
"""Off-policy evaluation: estimate a policy's value from another policy's logs.

    python run_ope.py
    python run_ope.py --transactions 60000 --propensity-samples 256

The deployment question this answers: *we have a candidate router. Is it better
than the one in production?* An A/B test answers it by routing real payments
through the candidate and counting the failures. Off-policy evaluation answers
it from logs already collected, before anything reaches a customer.

The run does four things:

1. Deploys a stochastic logging policy (Thompson sampling) and records, for
   every transaction, the propensity it assigned to the action it took.
2. Freezes several candidate policies.
3. Estimates each candidate's value from the logs alone -- IPS, clipped IPS,
   SNIPS, DM, DR.
4. Computes each candidate's **true** value analytically from the simulator's
   ground truth, and reports how far each estimator missed.

Step 4 is the part that makes this worth building here rather than taking on
faith. In production you never learn whether an estimate was good; in a
simulator you can check every one.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from src.gateways import ISSUER_MIX, ISSUERS, GatewayEnvironment, GatewaySpec, Outcome, RoutingContext
from src.ope import LoggedData, evaluate_all, true_value
from src.routers import ContextualThompsonRouter, ThompsonRouter
from src.simulator import SimConfig

DAYS = 2
SEED = 7


def fleet() -> list[GatewaySpec]:
    """A fleet where the best gateway differs by issuer.

    PG-Delta is best overall but declines HDFC; PG-Bravo is the best choice for
    HDFC specifically. A policy that knows the context can beat one that does
    not, which is what makes the candidates genuinely different rather than
    noisy variations on the same thing.
    """
    from src.gateways import DegradationEvent

    return [
        GatewaySpec("PG-Alpha", base_sr=0.915, base_latency_ms=240, cost_bps=18),
        GatewaySpec("PG-Bravo", base_sr=0.930, base_latency_ms=210, cost_bps=22),
        GatewaySpec("PG-Charlie", base_sr=0.905, base_latency_ms=180, cost_bps=15),
        GatewaySpec(
            "PG-Delta", base_sr=0.950, base_latency_ms=150, cost_bps=9,
            events=[DegradationEvent(start=0, end=10**9, sr_multiplier=0.55,
                                     issuer="HDFC", label="Delta is bad for HDFC")],
        ),
        GatewaySpec("PG-Echo", base_sr=0.925, base_latency_ms=200, cost_bps=20),
    ]


def collect_logs(
    specs: list[GatewaySpec],
    config: SimConfig,
    seed: int,
    propensity_samples: int,
) -> tuple[LoggedData, np.ndarray]:
    """Run the logging policy, recording propensities. Returns logs and truth.

    The propensity Monte Carlo uses its own generator, never the router's, so
    logging propensities does not perturb the routing decisions being logged.
    """
    env = GatewayEnvironment(specs, seed=seed)
    n, n_ticks = config.transactions, config.n_ticks
    sr = np.array(
        [[[s.true_sr(t, iss) for iss in ISSUERS] for t in range(n_ticks)] for s in specs]
    )

    ticks = (np.arange(n) * n_ticks // n).astype(int)
    draws = env.rng.random(n)
    issuers = np.random.default_rng(seed + 10_000).choice(
        len(ISSUERS), size=n, p=np.asarray(ISSUER_MIX)
    )
    propensity_rng = np.random.default_rng(seed + 999)

    router = ThompsonRouter(len(specs), gamma=0.999, seed=seed)
    actions = np.empty(n, dtype=int)
    rewards = np.empty(n, dtype=float)
    propensities = np.empty(n, dtype=float)

    for i in range(n):
        t, issuer_idx = int(ticks[i]), int(issuers[i])
        context = RoutingContext(tick=t, issuer=ISSUERS[issuer_idx])
        probs = router.action_probabilities(
            t, context, n_samples=propensity_samples, rng=propensity_rng
        )
        action = router.select(t, context)
        success = bool(draws[i] < sr[action, t, issuer_idx])

        actions[i] = action
        rewards[i] = float(success)
        # Floor the recorded propensity: a Monte Carlo estimate can land on
        # exactly zero for an action that was nonetheless taken, and a zero
        # denominator would make the importance weight infinite.
        propensities[i] = max(float(probs[action]), 1.0 / (propensity_samples * 10))
        router.update(Outcome(gateway=action, success=success, latency_ms=1.0,
                              tick=t, issuer=ISSUERS[issuer_idx]))

    logged = LoggedData(
        context=issuers, action=actions, reward=rewards, propensity=propensities,
        tick=ticks, n_actions=len(specs), n_contexts=len(ISSUERS),
    )
    return logged, sr


def build_targets(
    logged: LoggedData, specs: list[GatewaySpec], config: SimConfig, seed: int
) -> dict[str, np.ndarray]:
    """Candidate policies, each a fixed (n_contexts, n_actions) probability matrix.

    All are frozen. A learning policy cannot be evaluated this way -- importance
    sampling has no way to reconstruct the different state it would have reached
    from the different data it would have seen.
    """
    n_actions, n_contexts = logged.n_actions, logged.n_contexts

    uniform = np.full((n_contexts, n_actions), 1.0 / n_actions)

    # Greedy on the pooled empirical mean: what a flat policy would settle on.
    pooled = np.zeros(n_actions)
    counts = np.zeros(n_actions)
    np.add.at(pooled, logged.action, logged.reward)
    np.add.at(counts, logged.action, 1.0)
    best = int(np.argmax(pooled / np.maximum(counts, 1)))
    flat_greedy = np.zeros((n_contexts, n_actions))
    flat_greedy[:, best] = 1.0

    # A contextual bandit trained on an independent run, then frozen. This is
    # the realistic candidate: something trained offline that you would like to
    # evaluate before shipping.
    trained = ContextualThompsonRouter(n_actions, gamma=1.0, pooling=20.0, seed=seed + 1)
    env = GatewayEnvironment(specs, seed=seed + 1)
    n_ticks = config.n_ticks
    sr = np.array(
        [[[s.true_sr(t, iss) for iss in ISSUERS] for t in range(n_ticks)] for s in specs]
    )
    train_n = min(config.transactions, 40_000)
    train_ticks = (np.arange(train_n) * n_ticks // train_n).astype(int)
    train_issuers = np.random.default_rng(seed + 20_000).choice(
        len(ISSUERS), size=train_n, p=np.asarray(ISSUER_MIX)
    )
    train_draws = env.rng.random(train_n)
    for i in range(train_n):
        t, iss = int(train_ticks[i]), int(train_issuers[i])
        context = RoutingContext(tick=t, issuer=ISSUERS[iss])
        a = trained.select(t, context)
        ok = bool(train_draws[i] < sr[a, t, iss])
        trained.update(Outcome(gateway=a, success=ok, latency_ms=1.0, tick=t,
                               issuer=ISSUERS[iss]))

    frozen_rng = np.random.default_rng(seed + 555)
    contextual = np.array([
        trained.action_probabilities(
            0, RoutingContext(tick=0, issuer=name), n_samples=4096, rng=frozen_rng
        )
        for name in ISSUERS
    ])

    return {
        "uniform": uniform,
        "flat-greedy": flat_greedy,
        "frozen-contextual": contextual,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--transactions", type=int, default=40_000)
    ap.add_argument("--propensity-samples", type=int, default=128)
    ap.add_argument("--clip", type=float, default=20.0)
    args = ap.parse_args()

    specs = fleet()
    config = SimConfig(days=DAYS, transactions=args.transactions, seeds=(SEED,))

    print(f"logging policy : thompson (gamma=0.999), {args.transactions:,} transactions")
    print(f"propensities   : {args.propensity_samples} Monte Carlo samples per decision")
    logged, sr = collect_logs(specs, config, SEED, args.propensity_samples)
    print(f"logged value   : {logged.observed_value:.4f} "
          f"(what the logging policy actually earned)")
    print(f"propensity min : {logged.propensity.min():.4f}  "
          f"median {np.median(logged.propensity):.4f}\n")

    targets = build_targets(logged, specs, config, SEED)
    expanded = {name: matrix[logged.context] for name, matrix in targets.items()}

    for name, target in expanded.items():
        truth = true_value(logged, target, sr)
        estimates, _ = evaluate_all(logged, target, sr=None, clip=args.clip)
        lift = (truth - logged.observed_value) * 10_000

        print("=" * 78)
        print(f"candidate: {name}")
        print(f"  true value {truth:.4f}  ({lift:+.0f} bps vs the logging policy)")
        print(f"  {'estimator':18s} {'estimate':>10s} {'error':>9s} {'std err':>9s} "
              f"{'ESS':>9s}  95% CI")
        print("  " + "-" * 72)
        for estimate in estimates:
            error_bps = estimate.error_vs(truth) * 10_000
            ess = "-" if math.isnan(estimate.ess) else f"{estimate.ess:,.0f}"
            covers = "covers" if estimate.covers(truth) else "MISSES"
            print(f"  {estimate.estimator:18s} {estimate.value:>10.4f} "
                  f"{error_bps:>+8.0f}b {estimate.std_err:>9.4f} {ess:>9s}  {covers}")
        worst = max(estimates, key=lambda e: 0 if math.isnan(e.max_weight) else e.max_weight)
        print(f"  max importance weight {worst.max_weight:,.0f}"
              f"   off-support rows {worst.off_support_fraction:.1%}")
        print()

    print("=" * 78)
    print("Reading this:")
    print("  * ESS is the number that matters. 40,000 logged rows with an ESS of")
    print("    a few hundred means the estimate rests on a few hundred effective")
    print("    observations, and its confidence interval is decorative.")
    print("  * IPS is unbiased and usually the worst estimate here -- one lucky")
    print("    low-propensity row dominates the average.")
    print("  * SNIPS and DR are what you would actually ship. DR is consistent if")
    print("    EITHER the reward model or the propensities are right.")
    print("  * The further a candidate is from the logging policy, the worse every")
    print("    estimator does. Off-policy evaluation is interpolation, not")
    print("    extrapolation: it cannot tell you about actions nobody ever took.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
