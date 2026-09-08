"""Eval runner: arms, aggregate metrics, and a regression gate.

An *arm* is one configuration under test -- a provider, a prompt version, and
memory on or off. Running arms side by side against identical cases is what
turns "the new prompt feels better" into a number with a sign.

Memory is measured on **recurrence**, not on a single pass. Each arm runs the
case set twice with memory persisting between passes. The first pass is cold;
by the second, the store holds episodic records and any consolidated pattern.
If memory is worth anything, pass 2 should be cheaper (fewer tool calls to
reach the same answer) or more accurate. Measuring memory on a single cold
pass would measure nothing at all, since there would be nothing to remember.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from ..investigator import Investigator, InvestigatorConfig
from ..llm import INPUT_USD_PER_MTOK, OUTPUT_USD_PER_MTOK, LLMClient
from ..loop import LoopBudget
from ..memory import MemoryStore
from ..traces import TraceStore, new_session_id
from .cases import EvalCase, build_cases, telemetry_for
from .graders import CaseScore, LLMJudge, grade


@dataclass
class ArmConfig:
    """One configuration under test."""

    label: str
    prompt_version: str
    use_memory: bool
    passes: int = 2
    budget: LoopBudget = field(default_factory=LoopBudget)


@dataclass
class PassMetrics:
    """Aggregate scores for one pass of one arm."""

    arm: str
    pass_index: int
    n_cases: int
    exact_match: float
    partial_credit: float
    scope_accuracy: float
    completion_rate: float
    hallucination_rate: float
    segmentation_rate: float
    brier: float
    mean_tool_calls: float
    mean_steps: float
    mean_tokens: float
    mean_wall_seconds: float
    total_cost_usd: float
    judge_scores: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = {k: v for k, v in asdict(self).items() if k != "judge_scores"}
        row.update({f"judge_{k}": v for k, v in self.judge_scores.items()})
        return row


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def aggregate(arm: str, pass_index: int, scores: list[CaseScore]) -> PassMetrics:
    judge_dims = ("actionability", "groundedness", "overclaiming")
    judge_scores: dict[str, float] = {}
    for dim in judge_dims:
        values = [
            float(s.judge[dim]) for s in scores
            if s.judge and isinstance(s.judge.get(dim), (int, float))
        ]
        if values:
            judge_scores[dim] = _mean(values)

    return PassMetrics(
        arm=arm,
        pass_index=pass_index,
        n_cases=len(scores),
        exact_match=_mean([float(s.exact_match) for s in scores]),
        partial_credit=_mean([s.partial_credit for s in scores]),
        scope_accuracy=_mean([float(s.scope_correct) for s in scores]),
        completion_rate=_mean([float(s.completed) for s in scores]),
        hallucination_rate=_mean([float(not s.no_hallucination) for s in scores]),
        segmentation_rate=_mean([float(s.used_segmentation) for s in scores]),
        brier=_mean([s.brier for s in scores]),
        mean_tool_calls=_mean([float(s.tool_calls) for s in scores]),
        mean_steps=_mean([float(s.steps) for s in scores]),
        mean_tokens=_mean([float(s.tokens) for s in scores]),
        mean_wall_seconds=_mean([s.wall_seconds for s in scores]),
        total_cost_usd=round(
            sum(s.tokens for s in scores) / 1_000_000 * (INPUT_USD_PER_MTOK + OUTPUT_USD_PER_MTOK) / 2,
            4,
        ),
        judge_scores=judge_scores,
    )


def run_arm(
    client: LLMClient,
    arm: ArmConfig,
    cases: list[EvalCase] | None = None,
    judge: LLMJudge | None = None,
    verbose: bool = True,
    traces: TraceStore | None = None,
    session_id: str | None = None,
) -> tuple[list[PassMetrics], list[CaseScore]]:
    """Run one arm over the case set, `arm.passes` times, sharing one memory.

    Traces carry the arm label and pass index, so a stored run can be sliced by
    configuration afterwards -- which is the point of storing them at all.
    """
    cases = cases or build_cases()
    session_id = session_id or new_session_id("eval")
    memory = MemoryStore() if arm.use_memory else None
    all_metrics: list[PassMetrics] = []
    all_scores: list[CaseScore] = []

    for pass_index in range(arm.passes):
        pass_scores: list[CaseScore] = []
        for case in cases:
            store = telemetry_for(case)
            investigator = Investigator(
                store=store,
                client=client,
                memory=memory,
                config=InvestigatorConfig(
                    prompt_version=arm.prompt_version,
                    use_memory=arm.use_memory,
                    write_memory=arm.use_memory,
                    budget=arm.budget,
                ),
                traces=traces,
                session_id=f"{session_id}/{arm.label}/pass{pass_index}",
            )
            result = investigator.investigate(
                case.alert, case.window, case_id=case.case_id
            )
            score = grade(case, result, store.names)
            if judge is not None:
                score.judge = judge.score(case, result)
            pass_scores.append(score)

            if verbose:
                mark = "PASS" if score.exact_match else "FAIL"
                print(
                    f"    [{mark}] {case.case_id:24s} "
                    f"tools={score.tool_calls:2d} conf={score.confidence:.2f} "
                    f"{score.stop_reason}"
                )

        # Consolidation runs between passes: patterns only exist once there
        # are several episodes to compare.
        if memory is not None:
            memory.consolidate()

        metrics = aggregate(arm.label, pass_index, pass_scores)
        all_metrics.append(metrics)
        all_scores.extend(pass_scores)
        if verbose:
            print(
                f"  pass {pass_index}: exact={metrics.exact_match:.0%} "
                f"partial={metrics.partial_credit:.2f} brier={metrics.brier:.3f} "
                f"tools/case={metrics.mean_tool_calls:.1f}"
            )
    return all_metrics, all_scores


def write_report(
    metrics: list[PassMetrics],
    scores: list[CaseScore],
    results_dir: str,
    provider: str,
) -> dict[str, str]:
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "eval_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        rows = [m.to_row() for m in metrics]
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(results_dir, "eval_report.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "provider": provider,
                "metrics": [m.to_row() for m in metrics],
                "cases": [asdict(s) for s in scores],
            },
            fh, indent=2,
        )
    return {"csv": csv_path, "json": json_path}


def check_gate(
    metrics: list[PassMetrics], baseline_path: str, tolerance: float = 0.001
) -> tuple[bool, str]:
    """Compare the headline metric against a stored baseline.

    This is the piece that makes the eval a *gate* rather than a dashboard:
    wire it into CI and a prompt edit that quietly breaks the hard cases fails
    the build instead of shipping. Tolerance is near-zero because the offline
    baseline arm is fully deterministic; a live-model arm needs a wider band
    and several seeds before its variance is known.
    """
    if not os.path.exists(baseline_path):
        return True, f"no baseline at {baseline_path}; nothing to compare against"

    with open(baseline_path, encoding="utf-8") as fh:
        baseline = json.load(fh)
    previous = {(m["arm"], m["pass_index"]): m["exact_match"] for m in baseline.get("metrics", [])}

    regressions = []
    for m in metrics:
        was = previous.get((m.arm, m.pass_index))
        if was is None:
            continue
        if m.exact_match < was - tolerance:
            regressions.append(
                f"{m.arm} pass {m.pass_index}: exact_match {was:.3f} -> {m.exact_match:.3f}"
            )
    if regressions:
        return False, "REGRESSION: " + "; ".join(regressions)
    return True, "no regression against baseline"
