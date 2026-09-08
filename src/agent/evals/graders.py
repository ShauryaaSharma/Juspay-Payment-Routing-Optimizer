"""Grading. Deterministic first, LLM judge only where determinism cannot reach.

The split matters. Scope, gateway, and issuer are known exactly, so grading
them with a model would be strictly worse: slower, more expensive, and less
reliable than `==`. An LLM judge is reserved for the one thing that has no
ground truth -- whether the written summary would actually help the engineer
who gets paged -- and its score is reported in a separate column so a fluent
wrong answer can never inflate the headline number.

The headline metric is `exact_match`: scope AND gateway AND issuer all correct.
Partial credit is reported alongside it for diagnosis, but a diagnosis that
names the wrong gateway is wrong, and the top-line number should say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ...gateways import ISSUERS
from ..schemas import Diagnosis, InvestigationResult
from .cases import EvalCase

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "actionability": {
            "type": "integer", "minimum": 1, "maximum": 5,
            "description": "Could an on-call engineer act on this immediately? 5 = names the "
                           "specific failing path and a concrete next step.",
        },
        "groundedness": {
            "type": "integer", "minimum": 1, "maximum": 5,
            "description": "Are the cited numbers specific and consistent with the stated "
                           "finding? 5 = every claim carries a concrete figure.",
        },
        "overclaiming": {
            "type": "integer", "minimum": 1, "maximum": 5,
            "description": "Does confidence match the evidence? 5 = appropriately hedged; "
                           "1 = states certainty the investigation does not support.",
        },
        "comment": {"type": "string", "description": "One sentence justifying the scores."},
    },
    "required": ["actionability", "groundedness", "overclaiming", "comment"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You grade incident diagnoses written by an automated \
investigation agent for a payment gateway fleet.

You are shown the agent's written summary and evidence, and separately the \
ground truth. Grade only the QUALITY OF THE WRITE-UP on the three rubric \
dimensions. Whether the diagnosis was factually correct is already scored \
elsewhere and is not your job.

Be strict. A confident, well-written, wrong diagnosis should score LOW on \
overclaiming. A hedged, correct one should score high. Vague summaries that \
cite no numbers score low on groundedness even when the conclusion is right."""


@dataclass
class CaseScore:
    """Per-case grades. One row of the eval report."""

    case_id: str
    difficulty: str
    completed: bool
    scope_correct: bool = False
    gateway_correct: bool = False
    issuer_correct: bool = False
    no_hallucination: bool = True
    evidence_present: bool = False
    used_segmentation: bool = False
    confidence: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    steps: int = 0
    tokens: int = 0
    wall_seconds: float = 0.0
    stop_reason: str = ""
    judge: dict[str, Any] = field(default_factory=dict)

    @property
    def exact_match(self) -> bool:
        return (
            self.completed
            and self.scope_correct
            and self.gateway_correct
            and self.issuer_correct
        )

    @property
    def partial_credit(self) -> float:
        """0.5 for the right scope, 0.3 the right gateway, 0.2 the right issuer."""
        if not self.completed:
            return 0.0
        return (
            0.5 * self.scope_correct + 0.3 * self.gateway_correct + 0.2 * self.issuer_correct
        )

    @property
    def brier(self) -> float:
        """Calibration: squared error between stated confidence and correctness.

        An agent that is right and says 0.9 scores 0.01. One that is wrong and
        says 0.9 scores 0.81. Confidently wrong is the expensive failure in an
        on-call context, and this is the metric that names it.
        """
        return (self.confidence - (1.0 if self.exact_match else 0.0)) ** 2


def _matches(actual: str | None, expected: str | None) -> bool:
    if expected is None:
        return actual is None
    return actual is not None and actual.strip().lower() == expected.strip().lower()


def grade(
    case: EvalCase,
    result: InvestigationResult,
    fleet_names: list[str],
) -> CaseScore:
    """Deterministic grading against the planted ground truth."""
    score = CaseScore(
        case_id=case.case_id,
        difficulty=case.difficulty,
        completed=result.diagnosis is not None,
        tool_calls=result.usage.tool_calls,
        tool_errors=result.usage.tool_errors,
        steps=result.usage.steps,
        tokens=result.usage.input_tokens + result.usage.output_tokens,
        wall_seconds=result.usage.wall_seconds,
        stop_reason=result.stop_reason,
        used_segmentation="segment_failures" in result.tool_names_used(),
    )
    if result.diagnosis is None:
        # A run that produced nothing is maximally wrong AND maximally
        # unconfident; leaving confidence at 0.0 keeps Brier honest.
        return score

    d: Diagnosis = result.diagnosis
    score.confidence = d.confidence
    score.scope_correct = d.scope == case.expected_scope
    score.gateway_correct = _matches(d.primary_gateway, case.expected_gateway)
    score.issuer_correct = _matches(d.affected_issuer, case.expected_issuer)
    score.evidence_present = bool(d.evidence)

    valid_gateways = {n.lower() for n in fleet_names}
    valid_issuers = {i.lower() for i in ISSUERS}
    score.no_hallucination = (
        (d.primary_gateway is None or d.primary_gateway.lower() in valid_gateways)
        and (d.affected_issuer is None or d.affected_issuer.lower() in valid_issuers)
    )
    return score


class LLMJudge:
    """Rubric grader for write-up quality. Optional; skipped without a client.

    Known limits, stated because a judge whose weaknesses are undocumented is
    worse than no judge: it sees ground truth, so it is not blind; it is not
    calibrated against human raters; and it scores prose, not correctness.
    Treat its numbers as a directional signal on writing quality, never as the
    measure of whether the agent works.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def score(self, case: EvalCase, result: InvestigationResult) -> dict[str, Any]:
        if result.diagnosis is None:
            return {}
        d = result.diagnosis
        payload = {
            "agent_summary": d.summary,
            "agent_evidence": d.evidence,
            "agent_recommended_action": d.recommended_action,
            "agent_confidence": d.confidence,
            "ground_truth": {
                "scope": case.expected_scope,
                "gateway": case.expected_gateway,
                "issuer": case.expected_issuer,
            },
        }
        try:
            response = self.client.create(
                JUDGE_SYSTEM,
                [{"role": "user", "content": json.dumps(payload, indent=2)}],
                [],
            )
            return json.loads(response.text)
        except Exception as exc:  # a judge failure must never fail the eval run
            return {"error": str(exc)}
