#!/usr/bin/env python3
"""Run the investigation-agent eval suite.

    python run_evals.py                      # deterministic baseline, no API key needed
    python run_evals.py --live               # against claude-opus-5
    python run_evals.py --live --judge       # add the write-up quality rubric
    python run_evals.py --gate               # fail (exit 1) on regression vs. the baseline
    python run_evals.py --save-baseline      # record current scores as the gate baseline

Three arms run by default, each isolating one variable:

    v1-baseline    naive prompt, no memory      -- the control
    v2-procedural  improved prompt, no memory   -- what prompt engineering bought
    v2+memory      improved prompt, with memory -- what memory bought on recurrence
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from src.agent.evals.cases import build_cases
from src.agent.evals.graders import LLMJudge
from src.agent.evals.harness import (
    ArmConfig,
    check_gate,
    run_arm,
    write_report,
)
from src.agent.config import settings
from src.agent.llm import BaselinePolicyClient, default_client
from src.agent.loop import LoopBudget
from src.agent.traces import TraceStore, new_session_id

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
BASELINE_PATH = os.path.join(RESULTS, "eval_baseline.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true",
                    help="Use claude-opus-5 instead of the deterministic baseline policy.")
    ap.add_argument("--judge", action="store_true",
                    help="Also grade write-up quality with an LLM judge (requires --live).")
    ap.add_argument("--passes", type=int, default=2,
                    help="Passes over the case set per arm; memory is measured on pass 2.")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--gate", action="store_true",
                    help="Exit non-zero if exact_match regressed against the saved baseline.")
    ap.add_argument("--save-baseline", action="store_true",
                    help="Write current scores as the new gate baseline.")
    args = ap.parse_args()

    client = default_client(prefer_live=True) if args.live else BaselinePolicyClient()
    provider = getattr(client, "name", type(client).__name__)

    # Fail fast rather than writing a report full of provider errors that
    # looks like a quality collapse instead of a missing credential.
    if args.live and hasattr(client, "preflight"):
        ok, message = client.preflight()
        if not ok:
            print(f"cannot reach {provider}: {message}\n")
            print("Set ANTHROPIC_API_KEY, or run `ant auth login`.")
            print("Omit --live to run the deterministic baseline policy instead.")
            return 2
        print(f"preflight : {message}")
    judge = LLMJudge(client) if (args.judge and args.live) else None
    if args.judge and not args.live:
        print("note: --judge needs --live; the baseline policy cannot grade prose. Skipping.\n")

    cases = build_cases()
    budget = LoopBudget(max_steps=args.max_steps)
    arms = [
        ArmConfig("v1-baseline", "v1-baseline", use_memory=False, passes=args.passes, budget=budget),
        ArmConfig("v2-procedural", "v2-procedural", use_memory=False, passes=args.passes, budget=budget),
        ArmConfig("v2+memory", "v2-procedural", use_memory=True, passes=args.passes, budget=budget),
    ]

    cfg = settings(refresh=True)
    traces = TraceStore(cfg.traces)
    session = new_session_id("eval")

    print(f"provider : {provider}")
    print(f"cases    : {len(cases)}  ({', '.join(c.case_id for c in cases)})")
    print(f"arms     : {len(arms)} x {args.passes} passes")
    print(f"session  : {session}")
    print(f"traces   : {traces.target}\n")

    all_metrics, all_scores = [], []
    for arm in arms:
        print(f"[{arm.label}]  prompt={arm.prompt_version}  memory={arm.use_memory}")
        metrics, scores = run_arm(
            client, arm, cases, judge=judge, traces=traces, session_id=session
        )
        all_metrics.extend(metrics)
        all_scores.extend(scores)
        print()

    _print_table(all_metrics)
    _print_memory_effect(all_metrics)

    paths = write_report(all_metrics, all_scores, RESULTS, provider)
    print(f"\nwrote {paths['csv']}")
    print(f"wrote {paths['json']}")
    if traces.config.enabled:
        suffix = f", {traces.failures} failed" if traces.failures else ""
        print(f"traces: {traces.written} written to {traces.target}{suffix}")
    traces.close()

    if args.save_baseline:
        with open(BASELINE_PATH, "w", encoding="utf-8") as fh:
            json.dump({"provider": provider,
                       "metrics": [m.to_row() for m in all_metrics]}, fh, indent=2)
        print(f"wrote {BASELINE_PATH} (gate baseline)")

    if args.gate:
        ok, message = check_gate(all_metrics, BASELINE_PATH)
        print(f"\ngate: {message}")
        if not ok:
            return 1
    return 0


def _print_table(metrics) -> None:
    print("=" * 100)
    header = (f"{'arm':16s} {'pass':>4s} {'exact':>7s} {'partial':>8s} {'scope':>7s} "
              f"{'brier':>7s} {'seg%':>6s} {'tools':>6s} {'tokens':>8s}")
    print(header)
    print("-" * 100)
    for m in metrics:
        print(f"{m.arm:16s} {m.pass_index:>4d} {m.exact_match:>6.0%} {m.partial_credit:>8.2f} "
              f"{m.scope_accuracy:>6.0%} {m.brier:>7.3f} {m.segmentation_rate:>5.0%} "
              f"{m.mean_tool_calls:>6.1f} {m.mean_tokens:>8.0f}")
        if m.judge_scores:
            dims = "  ".join(f"{k}={v:.1f}" for k, v in sorted(m.judge_scores.items()))
            print(f"{'':16s} {'':>4s} judge: {dims}")
    print("=" * 100)
    print("exact = scope AND gateway AND issuer all correct (the headline).")
    print("brier = calibration error, lower is better; confidently wrong is punished hardest.")
    print("seg%  = share of cases where the agent segmented by issuer.")


def _print_memory_effect(metrics) -> None:
    """Isolate the recurrence effect: pass 0 vs pass 1, memory on vs off."""
    by_arm: dict[str, dict[int, object]] = {}
    for m in metrics:
        by_arm.setdefault(m.arm, {})[m.pass_index] = m

    rows = []
    for arm, passes in by_arm.items():
        if 0 in passes and 1 in passes:
            first, second = passes[0], passes[1]
            rows.append((
                arm,
                second.exact_match - first.exact_match,
                second.mean_tool_calls - first.mean_tool_calls,
            ))
    if not rows:
        return
    print("\nRecurrence effect (pass 1 minus pass 0) -- what memory is worth:")
    for arm, d_exact, d_tools in rows:
        print(f"  {arm:16s} exact {d_exact:+.0%}   tool calls/case {d_tools:+.2f}")


if __name__ == "__main__":
    sys.exit(main())
