"""Wires the pieces into one callable investigation.

Memory is injected into the brief rather than exposed as a tool. Both designs
work; injection was chosen because it makes memory a clean on/off variable for
the eval harness. A `search_memory` tool would let the agent decide when to
look things up, but then "memory helped" and "the agent chose to use memory"
are confounded, and the A/B stops answering the question it was built to ask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import prompts
from .config import Settings, settings
from .llm import LLMClient
from .loop import AgentLoop, LoopBudget
from .memory import MemoryStore
from .schemas import InvestigationResult
from .prompt_source import PromptSource, ResolvedPrompt
from .telemetry import TelemetryStore, format_clock
from .tools import ToolDispatcher
from .traces import TraceStore, build_record, new_session_id


@dataclass
class InvestigatorConfig:
    prompt_version: str | None = None
    use_memory: bool = True
    memory_k: int = 3
    write_memory: bool = True
    budget: LoopBudget | None = None

    @classmethod
    def from_env(cls, config: Settings | None = None) -> "InvestigatorConfig":
        """Build a config from environment settings.

        Every default here reproduces the hardcoded values this class shipped
        with, so an unset variable can never change a measured result.
        """
        cfg = config or settings()
        return cls(
            prompt_version=cfg.prompt_version,
            use_memory=cfg.memory.enabled,
            memory_k=cfg.memory.recall_k,
            write_memory=cfg.memory.enabled,
            budget=LoopBudget(
                max_steps=cfg.loop.max_steps,
                max_tool_calls=cfg.loop.max_tool_calls,
                max_output_tokens=cfg.loop.max_output_tokens,
                wall_seconds=cfg.loop.wall_seconds,
                max_consecutive_tool_errors=cfg.loop.max_consecutive_tool_errors,
                repair_attempts=cfg.loop.repair_attempts,
            ),
        )


class Investigator:
    """One fleet, one telemetry store, many investigations."""

    def __init__(
        self,
        store: TelemetryStore,
        client: LLMClient,
        memory: MemoryStore | None = None,
        config: InvestigatorConfig | None = None,
        traces: TraceStore | None = None,
        session_id: str | None = None,
        prompt_source: PromptSource | None = None,
    ) -> None:
        self.store = store
        self.client = client
        self.memory = memory
        self.config = config or InvestigatorConfig()
        # Resolved rather than looked up: the prompt may come from LangSmith,
        # and which source won has to be recorded on the trace.
        self.prompt_source = prompt_source or PromptSource()
        self.prompt: ResolvedPrompt = self.prompt_source.resolve(self.config.prompt_version)
        self.loop = AgentLoop(client, ToolDispatcher(store), self.config.budget)
        self.traces = traces
        self.session_id = session_id or new_session_id()

    def investigate(
        self,
        alert: str,
        window: tuple[int, int],
        verified: bool | None = None,
        case_id: str | None = None,
    ) -> InvestigationResult:
        recalled: list[str] = []
        if self.memory is not None and self.config.use_memory:
            # Query on the alert plus the fleet's vocabulary, so a memory about
            # PG-Bravo surfaces for an alert that never names it.
            query = f"{alert} {' '.join(self.store.names)}"
            recalled = self.memory.recall_text(query, k=self.config.memory_k)

        brief = prompts.render_incident_brief(alert, window, recalled)
        result = self.loop.run(self.prompt.system, brief)
        result.memory_hits = recalled

        if (
            self.memory is not None
            and self.config.write_memory
            and result.diagnosis is not None
        ):
            self.memory.remember_investigation(result.diagnosis, window, verified)

        # Traced last, so the record reflects the completed investigation --
        # including a failed one, which is the case most worth having later.
        if self.traces is not None and self.traces.config.enabled:
            self.traces.write(build_record(
                result,
                session_id=self.session_id,
                alert=alert,
                window=window,
                case_id=case_id,
                provider=getattr(self.client, "name", type(self.client).__name__),
                prompt_version=self.prompt.version,
                memory_enabled=self.memory is not None and self.config.use_memory,
                include_tool_results=self.traces.config.include_tool_results,
                # Provenance travels with the trace, so a stored run always
                # says which prompt text produced it -- the property a hosted
                # prompt store otherwise takes away.
                extra=self.prompt.describe(),
            ))
        return result

    def describe(self) -> dict[str, Any]:
        """The configuration that produced a result -- goes into every eval row."""
        return {
            "provider": getattr(self.client, "name", type(self.client).__name__),
            "prompt_version": self.prompt.version,
            "prompt_source": self.prompt.source,
            "memory": self.config.use_memory and self.memory is not None,
            "max_steps": self.loop.budget.max_steps,
            "max_tool_calls": self.loop.budget.max_tool_calls,
        }


def summarise(result: InvestigationResult) -> str:
    """Human-readable rendering for the CLI."""
    lines = []
    if result.diagnosis is None:
        lines.append(f"FAILED ({result.stop_reason}): {result.error}")
    else:
        d = result.diagnosis
        target = d.primary_gateway or "fleet"
        issuer = f" / {d.affected_issuer}" if d.affected_issuer else ""
        lines.append(f"{d.scope.upper()} on {target}{issuer}  (confidence {d.confidence:.2f})")
        lines.append(f"  {d.summary}")
        if d.evidence:
            lines.append("  evidence:")
            lines += [f"    - {e}" for e in d.evidence]
        if d.recommended_action:
            lines.append(f"  action: {d.recommended_action}")

    u = result.usage
    lines.append(
        f"  {u.steps} steps, {u.tool_calls} tool calls ({u.tool_errors} errors), "
        f"{u.input_tokens + u.output_tokens} tokens, {u.wall_seconds:.2f}s"
    )
    for step in result.steps:
        for call in step.tool_calls:
            flag = " [error]" if call.is_error else ""
            args = ", ".join(f"{k}={v}" for k, v in sorted(call.arguments.items()))
            lines.append(f"    step {step.index}: {call.name}({args}){flag}")
    return "\n".join(lines)
