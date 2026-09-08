#!/usr/bin/env python3
"""Benchmark every routing strategy against the same non-stationary fleet.

    python benchmark.py                      # default: 7 days, 100k tx, 3 seeds
    python benchmark.py --transactions 20000 --seeds 1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

from src.gateways import TICKS_PER_DAY, default_fleet
from src.metrics import outage_response_ticks
from src.plotting import line_chart, stacked_area
from src.routers import (
    EpsilonGreedyRouter,
    PIDThompsonRouter,
    StaticWeightedRouter,
    ThompsonRouter,
    UCB1Router,
)
from src.simulator import SimConfig, mean_std, run_many

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# thompson-stationary and pid-thompson share gamma=1.0 so the only difference
# between them is the control loop -- that pair isolates what the controller
# is actually worth. thompson-discounted is the strong baseline both must beat.
STRATEGIES = {
    "static-weighted": lambda n, s: StaticWeightedRouter(n, seed=s),
    "epsilon-greedy": lambda n, s: EpsilonGreedyRouter(n, epsilon=0.10, gamma=0.999, seed=s),
    "ucb1": lambda n, s: UCB1Router(n, c=0.6, gamma=0.999, seed=s),
    "thompson-stationary": lambda n, s: ThompsonRouter(n, gamma=1.0, seed=s),
    "thompson-discounted": lambda n, s: ThompsonRouter(n, gamma=0.999, seed=s),
    "pid-thompson": lambda n, s: PIDThompsonRouter(n, gamma=1.0, seed=s),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--transactions", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    specs = default_fleet()
    config = SimConfig(days=args.days, transactions=args.transactions,
                       seeds=tuple(range(args.seeds)))
    n_ticks = config.n_ticks
    tx_per_tick = config.transactions / n_ticks

    # The Bravo outage is the headline stress event; locate it in tx-space.
    bravo = 1
    outage = specs[bravo].events[0]
    outage_span = (outage.start * tx_per_tick, outage.end * tx_per_tick)

    print(f"fleet={len(specs)} gateways  days={config.days}  "
          f"transactions={config.transactions:,}  seeds={args.seeds}\n")

    runs: dict[str, list] = {}
    rows = []
    for name, factory in STRATEGIES.items():
        results = run_many(factory, specs, config)
        runs[name] = results

        sr_m, sr_s = mean_std([r.realised_sr for r in results])
        rg_m, rg_s = mean_std([r.total_regret for r in results])
        cost_m, _ = mean_std([r.mean_cost_bps for r in results])
        p99_m, _ = mean_std([r.latency_percentile(99) for r in results])
        resp = np.median([outage_response_ticks(r, bravo, outage, tx_per_tick) for r in results])

        rows.append({
            "router": name,
            "realised_sr": round(sr_m, 5),
            "sr_stddev": round(sr_s, 5),
            "total_regret": round(rg_m, 1),
            "regret_stddev": round(rg_s, 1),
            "mean_cost_bps": round(cost_m, 2),
            "p99_latency_ms": round(p99_m, 1),
            "outage_response_min": resp,
        })
        resp_str = "never" if np.isinf(resp) else f"{resp:.0f} min"
        print(f"  {name:22s} SR={sr_m:.4f} (+/-{sr_s:.4f})  "
              f"regret={rg_m:8.1f}  cost={cost_m:5.2f}bps  "
              f"outage-response={resp_str}")

    baseline = next(r for r in rows if r["router"] == "static-weighted")
    best = max(rows, key=lambda r: r["realised_sr"])
    lift_bps = (best["realised_sr"] - baseline["realised_sr"]) * 10_000
    print(f"\n  best = {best['router']}: {lift_bps:+.0f} bps SR vs static baseline "
          f"({best['realised_sr']:.2%} vs {baseline['realised_sr']:.2%})")

    csv_path = os.path.join(RESULTS, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  wrote {csv_path}")

    if args.no_charts:
        return 0

    shade = [(outage_span[0], outage_span[1], "Bravo outage")]
    names = [s.name for s in specs]

    ticks = (np.arange(config.transactions) * n_ticks // config.transactions).astype(int)
    sr_matrix = np.array([[s.true_sr(t) for t in range(n_ticks)] for s in specs])
    line_chart(
        os.path.join(RESULTS, "01_environment.svg"),
        {names[i]: sr_matrix[i][ticks] for i in range(len(specs))},
        "Ground-truth gateway success rate (never visible to the router)",
        "transaction", "true success rate", x_max=config.transactions,
        y_range=(0.0, 1.0), shaded=shade,
    )
    line_chart(
        os.path.join(RESULTS, "02_cumulative_regret.svg"),
        {k: v[0].cumulative_regret for k, v in runs.items()},
        "Cumulative regret vs. a clairvoyant router (lower is better)",
        "transaction", "cumulative regret", x_max=config.transactions, shaded=shade,
    )
    line_chart(
        os.path.join(RESULTS, "03_rolling_sr.svg"),
        {k: v[0].rolling_sr(2000) for k, v in runs.items()},
        "Realised success rate, 2k-transaction trailing window",
        "transaction", "success rate", x_max=config.transactions,
        y_range=(0.0, 1.0), shaded=shade,
    )
    for key, fname in [("static-weighted", "04_alloc_static.svg"),
                       ("pid-thompson", "05_alloc_pid.svg")]:
        stacked_area(os.path.join(RESULTS, fname), runs[key][0].allocation(), names,
                     f"Traffic allocation - {key}")

    print(f"  wrote 5 charts to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
