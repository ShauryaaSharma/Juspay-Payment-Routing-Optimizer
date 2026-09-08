#!/usr/bin/env python3
"""Contextual bandit vs. investigation agent: which fixes an issuer outage faster?

    python contextual_vs_agent.py              # deterministic agent, no API key
    python contextual_vs_agent.py --live       # investigate with claude-opus-5

This is the experiment that answers the obvious objection to the agent layer:
*why build an LLM investigation loop to discover that PG-Delta is declining HDFC
cards, when a bandit conditioned on the issuer learns it on its own?*

Four arms, all on the same fleet, seed, and incident:

    flat                Thompson sampling. Cannot express the fix at all.
    contextual          Per-(gateway, issuer) posteriors, pooled per gateway.
    contextual-nopool   Same, with pooling=0 -- shows what pooling buys.
    closed-loop         Flat router + agent-supplied constraint.

## The prediction was wrong

Going in, the expected result was a crossover: the agent winning at low volume
(one investigation, fixed cost) and the contextual bandit winning at high volume
(enough failed transactions to learn quickly, no model required).

There is no crossover. **The contextual bandit wins at every volume tested**,
by roughly 50 bps of overall success rate over the closed loop:

    overall SR      25k      50k     100k     200k
    flat          91.82%   92.20%   91.74%   92.18%
    contextual    92.92%   93.22%   93.00%   93.33%   <- best throughout
    closed-loop   92.22%   92.51%   92.35%   92.80%

The mechanism is visible in how much healthy traffic keeps using PG-Delta, the
fleet's best gateway:

    share of other issuers still on PG-Delta
    flat              1-4%     abandons it entirely
    contextual       50-66%    never stops trusting it for them
    closed-loop      32-38%    recovers some, but not all

Two reasons the agent trails, and the second is the interesting one:

1. **Detection lag.** The constraint lands 60 minutes in. The flat bandit has
   already spent that hour souring on PG-Delta for everyone.
2. **The constraint stops the bleeding but does not repair the damage.** It
   prevents further HDFC failures from being attributed to PG-Delta, but the
   flat posterior underneath is already contaminated by the failures that
   happened before detection -- and the canary keeps adding a trickle more. A
   contextual bandit never contaminates it, because HDFC's failures only ever
   touched HDFC's cell.

## What this means for the agent layer

For a failure mode you can enumerate in advance -- issuer x gateway -- a
contextual bandit is simply the better tool, and this experiment says so. The
agent's value has to be argued somewhere else:

* it handles failures nobody enumerated as a dimension (a fleet-wide event, a
  diurnal trough mistaken for an incident), which a bandit has no cell for;
* it produces an explanation a human can act on, which a posterior does not;
* it does not pay the dimensionality cost, which matters when the context is
  card network x issuer x amount band x merchant rather than five issuers.

The strongest version of this system is both: a contextual bandit for the axes
you know about, an agent for the ones you do not.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from close_loop import (
    AFFECTED_ISSUER,
    DAYS,
    DELTA,
    DETECT_AT,
    INCIDENT_END,
    INCIDENT_START,
    SEED,
    scenario_fleet,
)
from src.agent import Investigator, InvestigatorConfig, MemoryStore
from src.agent.config import settings
from src.agent.llm import BaselinePolicyClient, default_client
from src.agent.loop import LoopBudget
from src.closed_loop import run_closed_loop
from src.gateways import ISSUERS
from src.plotting import line_chart
from src.routers import ContextualThompsonRouter, ThompsonRouter
from src.simulator import SimConfig

VOLUMES = (25_000, 50_000, 100_000, 200_000)
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _sr(result, lo, hi, issuer=None, exclude=False) -> float:
    """Realised success rate in a window, for one issuer or all the others."""
    tick = np.asarray(result.run.tick)
    mask = (tick >= lo) & (tick < hi)
    if issuer is not None:
        match = np.asarray(result.run.issuer) == ISSUERS.index(issuer)
        mask &= ~match if exclude else match
    return float(np.asarray(result.run.success)[mask].mean())


def adaptation_minutes(
    result, gateway: int, issuer: str, window_minutes: int = 30, threshold: float = 0.05
) -> float:
    """Minutes from incident onset until the issuer stops reaching the gateway.

    The window is measured in **simulated minutes, not transactions**. An
    earlier version used a fixed 600-transaction trailing window, which at low
    volume takes ~220 simulated minutes just to fill -- so every arm reported
    almost the same number and those numbers scaled as 1/volume. That metric
    was measuring how long the window took to fill, not how long the router
    took to adapt, and it made four very different arms look identical.
    """
    tick = np.asarray(result.run.tick)
    chosen = np.asarray(result.run.chosen)
    match = np.asarray(result.run.issuer) == ISSUERS.index(issuer)

    for minute in range(INCIDENT_START, INCIDENT_END):
        lo = max(INCIDENT_START, minute - window_minutes)
        in_window = match & (tick >= lo) & (tick < minute)
        n = int(in_window.sum())
        if n < 30:  # too little traffic in the window to call it
            continue
        if float((chosen[in_window] == gateway).mean()) < threshold:
            return float(minute - INCIDENT_START)
    return float("inf")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--pooling", type=float, default=40.0)
    ap.add_argument("--no-chart", action="store_true")
    args = ap.parse_args()

    env = settings(refresh=True)
    client = default_client(prefer_live=True) if args.live else BaselinePolicyClient()
    if args.live and hasattr(client, "preflight"):
        ok, message = client.preflight()
        if not ok:
            print(f"cannot reach the model: {message}")
            return 2

    def investigate(telemetry, window):
        return Investigator(
            store=telemetry, client=client, memory=MemoryStore(),
            config=InvestigatorConfig(budget=LoopBudget(max_steps=env.loop.max_steps)),
        ).investigate(
            "Conversion is down. The routing controller flagged a calibration break "
            f"at minute {window[1]}. Work out what is wrong.",
            window, case_id="contextual_vs_agent",
        )

    n = len(scenario_fleet())
    arms = {
        "flat": (lambda g, s: ThompsonRouter(g, gamma=0.999, seed=s), None),
        "contextual": (
            lambda g, s: ContextualThompsonRouter(g, gamma=0.999, pooling=args.pooling, seed=s),
            None,
        ),
        "contextual-nopool": (
            lambda g, s: ContextualThompsonRouter(g, gamma=0.999, pooling=0.0, seed=s), None
        ),
        "closed-loop": (lambda g, s: ThompsonRouter(g, gamma=0.999, seed=s), investigate),
    }

    print(f"scenario : PG-Delta (best gateway, 94.5%) declines {AFFECTED_ISSUER} "
          f"from minute {INCIDENT_START} to {INCIDENT_END}")
    print(f"agent    : {getattr(client, 'name', type(client).__name__)}, "
          f"detection at minute {DETECT_AT} (+{DETECT_AT - INCIDENT_START} min lag)")
    print(f"pooling  : k={args.pooling}\n")

    specs = scenario_fleet()
    table: dict[str, dict[int, dict[str, float]]] = {a: {} for a in arms}

    for volume in VOLUMES:
        config = SimConfig(days=DAYS, transactions=volume, seeds=(SEED,))
        per_minute = volume / config.n_ticks
        print(f"--- {volume:,} transactions over {DAYS} days "
              f"({per_minute:.0f}/min, ~{per_minute * 60:.0f}/hour) ---")
        for label, (factory, hook) in arms.items():
            result = run_closed_loop(
                specs, config, factory, SEED,
                detect_ticks=(DETECT_AT,) if hook else (),
                investigate=hook,
                ttl_minutes=env.constraints.ttl_minutes,
                canary_rate=env.constraints.canary_rate,
            )
            lo, hi = INCIDENT_START, INCIDENT_END
            tick = np.asarray(result.run.tick)
            others = np.asarray(result.run.issuer) != ISSUERS.index(AFFECTED_ISSUER)
            in_incident = (tick >= lo) & (tick < hi)
            stats = {
                "hdfc_sr": _sr(result, lo, hi, AFFECTED_ISSUER),
                "other_sr": _sr(result, lo, hi, AFFECTED_ISSUER, exclude=True),
                "overall_sr": _sr(result, lo, hi),
                "adapt_min": adaptation_minutes(result, DELTA, AFFECTED_ISSUER),
                # The metric that actually separates the arms: how much healthy
                # traffic kept using the fleet's best gateway.
                "others_on_delta": float(
                    (np.asarray(result.run.chosen)[others & in_incident] == DELTA).mean()
                ),
            }
            table[label][volume] = stats

            note = ""
            if hook and result.constraints:
                c = result.constraints[0]
                note = f"  [{'scoped' if c.issuer else 'BLUNT'}]"
            adapt = ("never" if np.isinf(stats["adapt_min"])
                     else f"{stats['adapt_min']:.0f} min")
            print(f"  {label:19s} overall={stats['overall_sr']:.2%}  "
                  f"others={stats['other_sr']:.2%}  {AFFECTED_ISSUER}={stats['hdfc_sr']:.2%}  "
                  f"others-on-Delta={stats['others_on_delta']:.0%}  "
                  f"adapt={adapt}{note}")
        print()

    _summary(table)
    if not args.no_chart:
        path = _chart(table)
        print(f"\nwrote {path}")
    return 0


def _summary(table) -> None:
    print("=" * 78)
    print("Overall success rate during the incident, by traffic volume")
    print("=" * 78)
    header = f"{'arm':20s}" + "".join(f"{v // 1000:>13d}k" for v in VOLUMES)
    print(header)
    print("-" * 78)
    for label, by_volume in table.items():
        row = f"{label:20s}" + "".join(
            f"{by_volume[v]['overall_sr']:>13.2%}" for v in VOLUMES
        )
        print(row)
    print("-" * 78)
    print(f"\n{AFFECTED_ISSUER} is rescued by every arm (~90-91% throughout): a flat")
    print("bandit fixes it too, by abandoning PG-Delta for everyone. The separation")
    print("is in what that costs the other issuers:\n")
    for label, by_volume in table.items():
        row = f"{label:20s}" + "".join(
            f"{by_volume[v]['other_sr']:>13.2%}" for v in VOLUMES
        )
        print(row)
    print("\nShare of other issuers still routed to PG-Delta (the best gateway):")
    for label, by_volume in table.items():
        row = f"{label:20s}" + "".join(
            f"{by_volume[v]['others_on_delta']:>13.0%}" for v in VOLUMES
        )
        print(row)

    print("\nMinutes to stop routing HDFC to PG-Delta:")
    for label, by_volume in table.items():
        cells = []
        for v in VOLUMES:
            m = by_volume[v]["adapt_min"]
            cells.append("never".rjust(13) if np.isinf(m) else f"{m:>12.0f}m")
        print(f"{label:20s}" + "".join(cells))

    best_low = max(table, key=lambda a: table[a][VOLUMES[0]]["overall_sr"])
    best_high = max(table, key=lambda a: table[a][VOLUMES[-1]]["overall_sr"])
    print(f"\nBest at {VOLUMES[0]:,} tx : {best_low}")
    print(f"Best at {VOLUMES[-1]:,} tx: {best_high}")
    if best_low != best_high:
        print("A crossover: the cheaper approach wins once volume makes learning fast.")
    else:
        print("No crossover in this range -- one approach dominates throughout.")


def _chart(table) -> str:
    series = {
        label: np.array([table[label][v]["overall_sr"] for v in VOLUMES])
        for label in table
    }
    lo = min(float(v.min()) for v in series.values())
    hi = max(float(v.max()) for v in series.values())
    pad = max((hi - lo) * 0.25, 0.002)
    return line_chart(
        os.path.join(RESULTS, "08_contextual_vs_agent.svg"),
        series,
        "Overall success rate during the incident, by traffic volume",
        "transactions over 2 days: 25k / 50k / 100k / 200k",
        "success rate",
        x_max=len(VOLUMES) - 1,
        # Tight bounds on purpose. The arms differ by ~1.5 percentage points;
        # anchoring the axis at 1.0 flattened all four lines onto one row.
        y_range=(lo - pad, hi + pad),
        xticks=len(VOLUMES) - 1,
    )


if __name__ == "__main__":
    sys.exit(main())
