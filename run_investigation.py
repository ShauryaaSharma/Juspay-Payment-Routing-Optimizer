#!/usr/bin/env python3
"""Run a single investigation and print the full trace.

    python run_investigation.py                          # baseline policy, first case
    python run_investigation.py --case issuer_outage
    python run_investigation.py --case issuer_outage --live
    python run_investigation.py --list

With --live it calls claude-opus-5 and needs credentials (ANTHROPIC_API_KEY, or
an `ant auth login` profile). Without it, the deterministic baseline policy runs
and no network call is made.
"""

from __future__ import annotations

import argparse
import sys

from src.agent import Investigator, InvestigatorConfig, MemoryStore, summarise
from src.agent.evals.cases import build_cases, telemetry_for
from src.agent.config import settings
from src.agent.llm import BaselinePolicyClient, default_client
from src.agent.loop import LoopBudget
from src.agent.prompts import versions
from src.agent.traces import TraceStore, new_session_id


def main() -> int:
    cases = {c.case_id: c for c in build_cases()}
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="hard_outage", choices=list(cases))
    ap.add_argument("--live", action="store_true", help="Use claude-opus-5.")
    ap.add_argument("--prompt", default=None, choices=versions())
    ap.add_argument("--memory", action="store_true", help="Enable the memory layer.")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="List cases and exit.")
    args = ap.parse_args()

    if args.list:
        print(f"{'case':26s} {'difficulty':11s} expected")
        print("-" * 78)
        for case in cases.values():
            expected = case.expected_scope
            if case.expected_gateway:
                expected += f" / {case.expected_gateway}"
            if case.expected_issuer:
                expected += f" / {case.expected_issuer}"
            print(f"{case.case_id:26s} {case.difficulty:11s} {expected}")
        return 0

    case = cases[args.case]
    store = telemetry_for(case)
    client = default_client(prefer_live=True) if args.live else BaselinePolicyClient()

    cfg = settings(refresh=True)
    traces = TraceStore(cfg.traces)
    investigator = Investigator(
        store=store,
        client=client,
        memory=MemoryStore() if args.memory else None,
        config=InvestigatorConfig(
            prompt_version=args.prompt,
            use_memory=args.memory,
            budget=LoopBudget(max_steps=args.max_steps),
        ),
        traces=traces,
        session_id=new_session_id("investigate"),
    )

    print(f"case      : {case.case_id}  ({case.difficulty})")
    print(f"window    : {case.window_readable}")
    print(f"config    : {investigator.describe()}")
    print(f"\nalert     : {case.alert}\n")
    print("-" * 78)

    result = investigator.investigate(case.alert, case.window, case_id=case.case_id)
    print(summarise(result))
    if traces.config.enabled:
        print(f"  trace -> {traces.target}" if traces.written
              else f"  trace NOT written: {traces.last_error}")
    traces.close()

    print("-" * 78)
    expected = f"{case.expected_scope}"
    if case.expected_gateway:
        expected += f" / {case.expected_gateway}"
    if case.expected_issuer:
        expected += f" / {case.expected_issuer}"
    print(f"ground truth : {expected}")
    print(f"why          : {case.notes}")

    if result.diagnosis is not None:
        d = result.diagnosis
        correct = (
            d.scope == case.expected_scope
            and (d.primary_gateway or None) == case.expected_gateway
            and (d.affected_issuer or None) == case.expected_issuer
        )
        print(f"verdict      : {'CORRECT' if correct else 'INCORRECT'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
