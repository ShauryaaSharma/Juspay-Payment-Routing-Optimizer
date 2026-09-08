"""Structured types for the investigation agent.

Plain dataclasses rather than Pydantic models, deliberately: the offline path
must run with numpy as the only dependency so the eval harness is testable on
a machine with no API key and no SDK installed. The JSON Schema the model is
constrained to is written out explicitly in ``DIAGNOSIS_SCHEMA`` and validated
by ``Diagnosis.from_payload``, which is the same contract ``messages.parse()``
would enforce -- just without the import.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Scope = Literal["single_gateway", "issuer_specific", "fleet_wide", "no_incident"]
VALID_SCOPES: tuple[str, ...] = (
    "single_gateway",
    "issuer_specific",
    "fleet_wide",
    "no_incident",
)

# The schema handed to the model via output_config.format. Kept adjacent to
# the dataclass it validates into so the two cannot drift apart.
DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scope": {
            "type": "string",
            "enum": list(VALID_SCOPES),
            "description": (
                "single_gateway: one gateway degraded across all traffic. "
                "issuer_specific: degradation confined to one issuing bank. "
                "fleet_wide: every gateway degraded together. "
                "no_incident: the data does not support an incident."
            ),
        },
        "primary_gateway": {
            "type": ["string", "null"],
            "description": "Gateway name, or null when scope is fleet_wide/no_incident.",
        },
        "affected_issuer": {
            "type": ["string", "null"],
            "description": "Issuer name, only when scope is issuer_specific.",
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "summary": {
            "type": "string",
            "description": "Two or three sentences an on-call engineer can act on.",
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific numbers observed via tools that support the finding.",
        },
        "recommended_action": {"type": "string"},
    },
    "required": [
        "scope",
        "primary_gateway",
        "affected_issuer",
        "confidence",
        "summary",
        "evidence",
        "recommended_action",
    ],
    "additionalProperties": False,
}


class DiagnosisFormatError(ValueError):
    """The model returned something that is not a usable diagnosis."""


@dataclass
class Diagnosis:
    """What the agent concluded. The unit the eval harness grades."""

    scope: str
    primary_gateway: str | None
    affected_issuer: str | None
    confidence: float
    summary: str
    evidence: list[str] = field(default_factory=list)
    recommended_action: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Diagnosis":
        """Validate a decoded JSON object into a Diagnosis.

        Structured outputs make malformed JSON unlikely but not impossible --
        a refusal, a truncated stream, or the offline baseline can all produce
        something off-contract. Failing loudly here keeps a bad parse from
        being scored as a wrong answer, which would quietly corrupt the eval.
        """
        if not isinstance(payload, dict):
            raise DiagnosisFormatError(f"expected an object, got {type(payload).__name__}")

        missing = [k for k in DIAGNOSIS_SCHEMA["required"] if k not in payload]
        if missing:
            raise DiagnosisFormatError(f"missing required field(s): {', '.join(missing)}")

        scope = payload["scope"]
        if scope not in VALID_SCOPES:
            raise DiagnosisFormatError(f"unknown scope {scope!r}")

        try:
            confidence = float(payload["confidence"])
        except (TypeError, ValueError) as exc:
            raise DiagnosisFormatError(f"confidence is not a number: {exc}") from exc
        confidence = min(1.0, max(0.0, confidence))

        evidence = payload.get("evidence") or []
        if not isinstance(evidence, list):
            raise DiagnosisFormatError("evidence must be a list")

        return cls(
            scope=scope,
            primary_gateway=payload.get("primary_gateway") or None,
            affected_issuer=payload.get("affected_issuer") or None,
            confidence=confidence,
            summary=str(payload.get("summary", "")),
            evidence=[str(e) for e in evidence],
            recommended_action=str(payload.get("recommended_action", "")),
        )

    @classmethod
    def from_json(cls, text: str) -> "Diagnosis":
        try:
            return cls.from_payload(json.loads(text))
        except json.JSONDecodeError as exc:
            raise DiagnosisFormatError(f"response was not valid JSON: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    """One tool invocation and what came back. The unit of the audit trail."""

    name: str
    arguments: dict[str, Any]
    result: Any
    is_error: bool = False
    duration_ms: float = 0.0


@dataclass
class Step:
    """One turn of the agent loop."""

    index: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    wall_seconds: float = 0.0

    def cost_usd(self, input_per_mtok: float, output_per_mtok: float) -> float:
        """Dollar cost at the caller-supplied rates.

        Rates are passed in rather than hardcoded so the number cannot go
        stale in this file when pricing changes.
        """
        return (
            self.input_tokens / 1_000_000 * input_per_mtok
            + self.output_tokens / 1_000_000 * output_per_mtok
        )


@dataclass
class InvestigationResult:
    """Everything one investigation produced, including how it got there."""

    diagnosis: Diagnosis | None
    steps: list[Step]
    usage: Usage
    stop_reason: str
    error: str | None = None
    memory_hits: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.diagnosis is not None

    def tool_names_used(self) -> list[str]:
        return [c.name for s in self.steps for c in s.tool_calls]
