"""LLM providers behind one interface.

Two implementations, and the second is not a mock:

* ``AnthropicClient`` calls Claude.
* ``BaselinePolicyClient`` runs a fixed heuristic investigation with no model
  at all -- fleet sweep, pick the worst gateway, compare against baseline,
  segment if the damage looks partial.

The baseline exists because an eval without a floor cannot tell you anything.
If Claude scores 0.82 and the question is "is that good", the only useful
answer is "compared to what" -- and a hand-written heuristic is the honest
comparison. It also means the loop, the graders, and the harness are all
exercisable and testable with no API key, which keeps the eval suite runnable
in CI.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import settings
from .schemas import DIAGNOSIS_SCHEMA

# Defaults, overridable via AGENT_MODEL / AGENT_MAX_TOKENS / AGENT_EFFORT and
# the AGENT_*_USD_PER_MTOK pricing variables. Kept as module constants so
# existing call sites and tests keep working unchanged.
DEFAULT_MODEL = settings().model.model
MAX_TOKENS = settings().model.max_tokens

# Published rates for claude-opus-5, USD per million tokens. Passed explicitly
# into Usage.cost_usd rather than baked into it, so a pricing change is one
# environment variable rather than a hunt through the codebase.
INPUT_USD_PER_MTOK = settings().model.input_usd_per_mtok
OUTPUT_USD_PER_MTOK = settings().model.output_usd_per_mtok


@dataclass
class PendingToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[PendingToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    raw_content: Any = None  # echoed back into messages verbatim
    refusal_category: str | None = None


class LLMClient(Protocol):
    name: str

    def create(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse: ...


class AnthropicClient:
    """Claude, configured for a tool-using agent that must return strict JSON.

    Three choices worth naming:

    * ``output_config.format`` pins the final answer to the diagnosis schema,
      so the loop never has to salvage JSON out of prose.
    * The system prompt carries ``cache_control``. Tools render before system,
      and both are byte-stable across an investigation, so every step after the
      first reads the prefix from cache instead of paying for it again.
    * Adaptive thinking is on. This is a diagnostic task with a real chance of
      reaching a confident wrong answer, and the reasoning is what stops the
      model from blaming the first gateway it looks at.
    """

    def __init__(
        self,
        model: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
    ) -> None:
        cfg = settings().model
        model = model or cfg.model
        effort = effort or cfg.effort
        max_tokens = max_tokens or cfg.max_tokens
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "the anthropic SDK is required for AnthropicClient: pip install anthropic"
            ) from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"{model}/{effort}"

    def preflight(self) -> tuple[bool, str]:
        """Verify credentials before spending a whole eval run discovering they are missing.

        Uses `count_tokens`, which exercises the same auth path as a real
        request but generates nothing and costs nothing. Without this, an eval
        run with no key completes "successfully" with every case recorded as a
        provider error -- 48 rows of zeros that look like a catastrophic
        quality regression rather than a missing environment variable.
        """
        try:
            self.client.messages.count_tokens(
                model=self.model, messages=[{"role": "user", "content": "ping"}]
            )
            return True, f"credentials OK for {self.model}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {str(exc)[:160]}"

    def create(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        anthropic = self._anthropic
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=messages,
                tools=tools,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": DIAGNOSIS_SCHEMA},
                },
            )
        except anthropic.APIStatusError as exc:
            # Surface the status so the loop can distinguish a retryable
            # outage from a request it must not send again.
            raise RuntimeError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(f"could not reach the Anthropic API: {exc}") from exc

        # A refusal is HTTP 200 with no usable content -- check before reading.
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            return LLMResponse(
                stop_reason="refusal",
                refusal_category=category,
                raw_content=response.content,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            PendingToolCall(id=b.id, name=b.name, input=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        return LLMResponse(
            text=text,
            tool_calls=calls,
            stop_reason=response.stop_reason or "end_turn",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            raw_content=response.content,
        )


class BaselinePolicyClient:
    """A fixed investigation heuristic. No model, no API key, fully deterministic.

    The policy an experienced engineer would run by hand:

        fleet sweep -> worst gateway -> compare against a baseline window
        -> segment by issuer if the damage looks partial -> conclude

    It scores well on incidents that match that shape and badly on the ones
    that do not, which is exactly what makes it a useful floor: the eval's job
    is to show where judgement beats a fixed procedure.
    """

    name = "baseline-policy"

    # Below this, a gateway is plainly dead rather than selectively broken.
    COLLAPSE_SR = 0.55
    # Above this, nothing is wrong at the fleet level.
    HEALTHY_SR = 0.90

    def __init__(self) -> None:
        self._counter = 0

    def _call(self, name: str, **kwargs: Any) -> LLMResponse:
        self._counter += 1
        return LLMResponse(
            tool_calls=[PendingToolCall(id=f"baseline_{self._counter}", name=name, input=kwargs)],
            stop_reason="tool_use",
        )

    @staticmethod
    def _observed(messages: list[dict[str, Any]]) -> list[tuple[str, Any]]:
        """Replay this conversation's tool results in order.

        The policy is a state machine over what it has already seen, so it has
        to reconstruct that from the transcript -- the same information the
        model would be reading.
        """
        seen: list[tuple[str, Any]] = []
        pending: list[str] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, PendingToolCall):
                    pending.append(block.name)
                elif isinstance(block, dict) and block.get("type") == "tool_result":
                    name = pending.pop(0) if pending else "?"
                    try:
                        seen.append((name, json.loads(block.get("content", "{}"))))
                    except (json.JSONDecodeError, TypeError):
                        seen.append((name, {}))
        return seen

    @staticmethod
    def _window(messages: list[dict[str, Any]]) -> tuple[int, int]:
        """Recover the incident window from the brief the loop rendered."""
        for message in messages:
            content = message.get("content")
            if isinstance(content, str) and "Minutes " in content:
                for line in content.splitlines():
                    if line.startswith("Minutes ") and " to " in line:
                        try:
                            head = line.removeprefix("Minutes ").split(" since")[0]
                            lo, hi = head.split(" to ")
                            return int(lo), int(hi)
                        except (ValueError, IndexError):
                            break
        return 0, 1440

    def create(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        seen = self._observed(messages)
        start, end = self._window(messages)
        baseline_start, baseline_end = max(0, start - (end - start)), start
        done = {name for name, _ in seen}

        if "get_fleet_health" not in done:
            return self._call("get_fleet_health", start_minute=start, end_minute=end)

        fleet = next(r for name, r in seen if name == "get_fleet_health")
        rated = [g for g in fleet.get("per_gateway", []) if g.get("success_rate") is not None]
        if not rated:
            return self._finish(self._diagnosis("no_incident", None, None, 0.2,
                                                "No gateway carried enough traffic to assess.", []))

        healthy = [g for g in rated if g["success_rate"] >= self.HEALTHY_SR]
        worst = min(rated, key=lambda g: g["success_rate"])
        fleet_sr = fleet.get("fleet_success_rate")

        # Everything is down together: no single gateway is at fault.
        if not healthy and fleet_sr is not None and fleet_sr < self.HEALTHY_SR:
            return self._finish(self._diagnosis(
                "fleet_wide", None, None, 0.6,
                f"All {len(rated)} gateways are below {self.HEALTHY_SR:.0%}; "
                f"fleet success rate is {fleet_sr:.2%}.",
                [f"fleet success rate {fleet_sr:.2%} across {len(rated)} gateways"],
            ))

        if worst["success_rate"] >= self.HEALTHY_SR:
            return self._finish(self._diagnosis(
                "no_incident", None, None, 0.5,
                f"Every gateway is at or above {self.HEALTHY_SR:.0%}.",
                [f"worst gateway {worst['gateway']} at {worst['success_rate']:.2%}"],
            ))

        name = worst["gateway"]
        if "compare_windows" not in done:
            return self._call(
                "compare_windows", gateway=name,
                baseline_start=baseline_start, baseline_end=baseline_end,
                current_start=start, current_end=end,
            )

        # Partial damage is the signature of an issuer-scoped failure.
        if worst["success_rate"] > self.COLLAPSE_SR and "segment_failures" not in done:
            return self._call(
                "segment_failures", gateway=name,
                start_minute=start, end_minute=end, by="issuer",
            )

        evidence = [f"{name} success rate {worst['success_rate']:.2%} in the incident window"]
        segments = next((r for n, r in seen if n == "segment_failures"), None)
        if segments:
            rated_segments = [
                s for s in segments.get("segments", []) if s.get("success_rate") is not None
            ]
            if rated_segments:
                low = min(rated_segments, key=lambda s: s["success_rate"])
                others = [s["success_rate"] for s in rated_segments if s is not low]
                typical = sum(others) / len(others) if others else 1.0
                if low["success_rate"] < 0.6 * typical:
                    return self._finish(self._diagnosis(
                        "issuer_specific", name, low["issuer"], 0.7,
                        f"{name} is declining {low['issuer']} traffic at "
                        f"{low['success_rate']:.2%} while other issuers convert near "
                        f"{typical:.2%}.",
                        evidence + [
                            f"{low['issuer']} on {name}: {low['success_rate']:.2%}",
                            f"other issuers on {name} average {typical:.2%}",
                        ],
                    ))

        return self._finish(self._diagnosis(
            "single_gateway", name, None, 0.65,
            f"{name} degraded to {worst['success_rate']:.2%} while other gateways held.",
            evidence,
        ))

    @staticmethod
    def _diagnosis(scope, gateway, issuer, confidence, summary, evidence) -> dict[str, Any]:
        return {
            "scope": scope,
            "primary_gateway": gateway,
            "affected_issuer": issuer,
            "confidence": confidence,
            "summary": summary,
            "evidence": evidence,
            "recommended_action": (
                "Shift traffic away from the affected path and raise it with the provider."
                if scope != "no_incident" else "No action; continue monitoring."
            ),
        }

    @staticmethod
    def _finish(payload: dict[str, Any]) -> LLMResponse:
        return LLMResponse(
            text=json.dumps(payload), stop_reason="end_turn", raw_content=json.dumps(payload)
        )


def default_client(prefer_live: bool | None = None) -> LLMClient:
    """Pick a provider.

    Defaults to the live model when credentials exist and the SDK is importable,
    and to the deterministic baseline otherwise -- so `python run_evals.py`
    works on a fresh clone with no key, and the report says which one ran.
    """
    if prefer_live is False:
        return BaselinePolicyClient()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if prefer_live or has_key:
        try:
            return AnthropicClient()
        except ImportError:
            if prefer_live:
                raise
    return BaselinePolicyClient()
