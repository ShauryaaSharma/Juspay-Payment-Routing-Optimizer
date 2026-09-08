#!/usr/bin/env python3
"""Discount-factor sensitivity: the experiment that justifies the control loop.

A discounted bandit has one knob, ``gamma``, and its correct value depends on
how fast the world drifts *relative to transaction volume*. This sweep holds
the drift fixed, varies volume, and asks two questions:

1. How much success rate does a mis-set ``gamma`` cost?
2. Does the PID loop recover that loss without being told the right value?

Run:  python sweep_gamma.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

from src.gateways import default_fleet
from src.plotting import line_chart
from src.routers import PIDThompsonRouter, ThompsonRouter
from src.simulator import SimConfig, run_once

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GAMMAS = [1.0, 0.99999, 0.9999, 0.999, 0.99]
VOLUMES = [20_000, 100_000]
SEEDS = 3


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    specs = default_fleet()
    rows = []

    for tx in VOLUMES:
        config = SimConfig(transactions=tx, seeds=tuple(range(SEEDS)))
        print(f"\n{tx:,} transactions/week")
        pid = np.mean([
            run_once(lambda n, s: PIDThompsonRouter(n, gamma=1.0, seed=s), specs, config, sd).realised_sr
            for sd in range(SEEDS)
        ])
        for gamma in GAMMAS:
            sr = np.mean([
                run_once(lambda n, s: ThompsonRouter(n, gamma=gamma, seed=s), specs, config, sd).realised_sr
                for sd in range(SEEDS)
            ])
            rows.append({"transactions": tx, "gamma": gamma,
                         "thompson_sr": round(float(sr), 5),
                         "pid_sr": round(float(pid), 5)})
            print(f"  gamma={gamma:<9} thompson SR={sr:.4f}")
        print(f"  {'pid (gamma=1.0)':<15}      SR={pid:.4f}  <- no gamma tuning")

    path = os.path.join(RESULTS, "gamma_sweep.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    series = {}
    for tx in VOLUMES:
        sub = [r for r in rows if r["transactions"] == tx]
        series[f"thompson @ {tx // 1000}k tx"] = np.array([r["thompson_sr"] for r in sub])
        series[f"pid @ {tx // 1000}k tx"] = np.array([sub[0]["pid_sr"]] * len(sub))
    line_chart(os.path.join(RESULTS, "06_gamma_sensitivity.svg"), series,
               "Success rate vs. discount factor (x-axis: gamma index, 1.0 -> 0.99)",
               "gamma index (0=1.0, 4=0.99)", "realised success rate",
               x_max=len(GAMMAS) - 1, y_range=(0.92, 0.96))
    print(f"\nwrote {path} and 06_gamma_sensitivity.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
