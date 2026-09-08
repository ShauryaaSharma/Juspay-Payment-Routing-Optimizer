"""The agent loop.

Written by hand rather than delegated to the SDK's tool runner. The runner is
the right default for most agents, but everything interesting about running an
agent in production lives in the parts it abstracts away: what happens when the
budget runs out mid-investigation, when a tool errors three times in a row, when
the model asks the same question twice, when the final answer does not parse.
Those are the failure modes that page someone, so they are the ones worth
owning explicitly.

Five properties this loop guarantees:

1. **Bounded.** Steps, tool calls, output tokens, and wall-clock all have caps.
   An agent with no ceiling is an unbounded bill and an unbounded latency tail.
2. **Terminating.** Every exit path sets a `stop_reason`. There is no branch
   that can spin.
3. **Recoverable.** A tool error is data, not an exception -- it goes back to
   the model as an error result. Only *repeated* errors break the loop.
4. **Non-repeating.** An identical repeated tool call gets a nudge instead of
   the same bytes again, which is how these loops usually stall.
5. **Answer-forcing.** The loop reserves its final step to demand a
   conclusion, so budget exhaustion degrades to a low-confidence answer rather
   than to nothing at all.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from .llm import LLMClient, LLMResponse, PendingToolCall
from .schemas import (
    Diagnosis,
    DiagnosisFormatError,
    InvestigationResult,
    Step,
    ToolCall,
    Usage,
)
from .tools import ToolDispatcher, tool_definitions


@dataclass
class LoopBudget:
    """Hard ceilings. Defaults sized for this task, not universal truths."""

    max_steps: int = 8
    max_tool_calls: int = 16
    max_output_tokens: int = 40_000
    wall_seconds: float = 300.0
    max_consecutive_tool_errors: int = 3
    repair_attempts: int = 1


FORCE_ANSWER = (
    "You have no investigation budget left. Do not call any more tools. "
    "Report your diagnosis now in the required JSON format, based only on what "
    "you have already observed, and set `confidence` to reflect how little you "
    "were able to check."
)

REPAIR_NUDGE = (
    "That response could not be parsed as a valid diagnosis ({error}). "
    "Reply with the diagnosis JSON only, matching the required schema exactly."
)

DUPLICATE_NUDGE = (
    "You already called {name} with exactly these arguments and received a result "
    "above. Re-read it rather than repeating the call. If you have what you need, "
    "report your diagnosis now."
)


class AgentLoop:
    """Drives one investigation to a diagnosis or to a recorded failure."""

    def __init__(
        self,
        client: LLMClient,
        dispatcher: ToolDispatcher,
        budget: LoopBudget | None = None,
    ) -> None:
        self.client = client
        self.dispatcher = dispatcher
        self.budget = budget or LoopBudget()
        self.tools = tool_definitions()

    def run(self, system: str, brief: str) -> InvestigationResult:
        started = time.perf_counter()
        messages: list[dict[str, Any]] = [{"role": "user", "content": brief}]
        steps: list[Step] = []
        usage = Usage()
        seen_calls: set[str] = set()
        consecutive_errors = 0
        repairs_used = 0
        forced = False

        while True:
            elapsed = time.perf_counter() - started
            stop = self._budget_exceeded(usage, elapsed, len(steps))
            # Rather than bailing the moment a budget trips, spend one final
            # turn demanding an answer. A low-confidence diagnosis beats none.
            if stop and not forced:
                forced = True
                messages.append({"role": "user", "content": FORCE_ANSWER})
                stop = None
            elif stop:
                return self._finish(None, steps, usage, stop, started, "budget exhausted")

            try:
                response = self.client.create(system, messages, self.tools)
            except Exception as exc:  # provider failure -- report, never crash
                return self._finish(None, steps, usage, "provider_error", started, str(exc))

            usage.input_tokens += response.input_tokens
            usage.output_tokens += response.output_tokens
            usage.cache_read_tokens += response.cache_read_tokens
            step = Step(
                index=len(steps),
                text=response.text,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

            if response.stop_reason == "refusal":
                steps.append(step)
                return self._finish(
                    None, steps, usage, "refusal", started,
                    f"model declined ({response.refusal_category or 'unspecified'})",
                )

            if response.tool_calls and not forced:
                consecutive_errors = self._execute_tools(
                    response, messages, step, usage, seen_calls, consecutive_errors
                )
                steps.append(step)
                if consecutive_errors >= self.budget.max_consecutive_tool_errors:
                    return self._finish(
                        None, steps, usage, "tool_error_limit", started,
                        f"{consecutive_errors} consecutive tool errors",
                    )
                continue

            # No tool calls: this is meant to be the final answer.
            steps.append(step)
            try:
                diagnosis = Diagnosis.from_json(response.text)
            except DiagnosisFormatError as exc:
                if repairs_used < self.budget.repair_attempts and not forced:
                    repairs_used += 1
                    messages.append(self._assistant_turn(response))
                    messages.append(
                        {"role": "user", "content": REPAIR_NUDGE.format(error=exc)}
                    )
                    continue
                return self._finish(None, steps, usage, "invalid_output", started, str(exc))
            return self._finish(diagnosis, steps, usage, "completed", started, None)

    # -- internals ---------------------------------------------------------

    def _execute_tools(
        self,
        response: LLMResponse,
        messages: list[dict[str, Any]],
        step: Step,
        usage: Usage,
        seen_calls: set[str],
        consecutive_errors: int,
    ) -> int:
        messages.append(self._assistant_turn(response))
        results: list[dict[str, Any]] = []

        for pending in response.tool_calls:
            signature = f"{pending.name}:{json.dumps(pending.input, sort_keys=True)}"
            if signature in seen_calls:
                call = ToolCall(
                    name=pending.name, arguments=pending.input,
                    result={"note": DUPLICATE_NUDGE.format(name=pending.name)},
                    is_error=False,
                )
            else:
                seen_calls.add(signature)
                call = self.dispatcher.dispatch(pending.name, pending.input)

            step.tool_calls.append(call)
            usage.tool_calls += 1
            if call.is_error:
                usage.tool_errors += 1
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            results.append({
                "type": "tool_result",
                "tool_use_id": pending.id,
                "content": self.dispatcher.render(call),
                "is_error": call.is_error,
            })

        # All results go back in a SINGLE user message. Splitting them across
        # several messages teaches the model to stop issuing parallel calls.
        messages.append({"role": "user", "content": results})
        return consecutive_errors

    @staticmethod
    def _assistant_turn(response: LLMResponse) -> dict[str, Any]:
        """Echo the assistant turn back verbatim where the provider gives us one."""
        content: Any = response.raw_content
        if content is None:
            content = response.tool_calls or response.text
        return {"role": "assistant", "content": content}

    def _budget_exceeded(self, usage: Usage, elapsed: float, n_steps: int) -> str | None:
        """Which ceiling, if any, has been hit.

        ``n_steps`` is passed in rather than read off ``usage``: usage.steps is
        only finalised in `_finish`, so reading it here would compare against a
        permanent zero and the step budget would never bind.
        """
        b = self.budget
        if n_steps >= b.max_steps:
            return "budget_steps"
        if usage.tool_calls >= b.max_tool_calls:
            return "budget_tool_calls"
        if usage.output_tokens >= b.max_output_tokens:
            return "budget_tokens"
        if elapsed >= b.wall_seconds:
            return "budget_time"
        return None

    def _finish(
        self,
        diagnosis: Diagnosis | None,
        steps: list[Step],
        usage: Usage,
        stop_reason: str,
        started: float,
        error: str | None,
    ) -> InvestigationResult:
        usage.steps = len(steps)
        usage.wall_seconds = round(time.perf_counter() - started, 3)
        return InvestigationResult(
            diagnosis=diagnosis, steps=steps, usage=usage,
            stop_reason=stop_reason, error=error,
        )
