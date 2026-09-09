"""Langfuse as a trace backend.

The existing JSONL and Postgres backends store a trace as one flat row. Langfuse
stores it as a tree, which is the shape an investigation actually has:

    agent      one investigation
    ├─ span    step 0
    │  └─ tool   get_fleet_health   (input: window, output: the JSON it returned)
    ├─ span    step 1
    │  └─ tool   segment_failures
    └─ span    step 2 -- the final answer

That tree is worth more than the flat row for the thing traces exist for:
opening a wrong diagnosis and seeing *which tool result the reasoning turned
on*. A JSONL row has the same data, but you read it with `jq`.

Langfuse over LangSmith for this: it self-hosts. Payment telemetry under RBI
localisation rules cannot go to a SaaS-only endpoint, and `LANGFUSE_HOST`
pointing at your own deployment is a one-line change.

Cost and token counts are attached to a `generation` observation rather than
invented as metadata, so Langfuse's own cost tracking works instead of being
duplicated.

**It never raises.** Same contract as every other sink in this project: a
broken observability backend is an observability outage, not an
incident-response outage. Failures are counted, warned once, and dropped.
"""

from __future__ import annotations

from typing import Any

from .config import TraceSettings
from .traces import TraceRecord


class LangfuseSink:
    """Writes one trace tree per investigation. Degrades to a no-op."""

    name = "langfuse"

    def __init__(self, config: TraceSettings) -> None:
        self.config = config
        self.client: Any = None
        self.written = 0
        self.failures = 0
        self.last_error: str | None = None
        self._warned = False

        if not (config.langfuse_public_key and config.langfuse_secret_key):
            self.last_error = "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set"
            return
        try:
            from langfuse import Langfuse
        except ImportError:
            self.last_error = "langfuse not installed: pip install langfuse"
            return
        try:
            self.client = Langfuse(
                public_key=config.langfuse_public_key,
                secret_key=config.langfuse_secret_key,
                host=config.langfuse_host,
                environment=config.langfuse_environment,
                timeout=int(config.connect_timeout),
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.client = None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @property
    def target(self) -> str:
        host = self.config.langfuse_host or "https://cloud.langfuse.com"
        return f"{host} ({self.config.langfuse_environment or 'default'})"

    def auth_check(self) -> tuple[bool, str]:
        """Verify credentials without writing a trace."""
        if not self.enabled:
            return False, self.last_error or "not configured"
        try:
            self.client.auth_check()
            return True, f"credentials OK for {self.target}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {str(exc)[:160]}"

    # -- writing ----------------------------------------------------------

    def write(self, record: TraceRecord) -> bool:
        if not self.enabled:
            return False
        try:
            self._emit(record)
            self.written += 1
            return True
        except Exception as exc:
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            if not self._warned:
                self._warned = True
                print(
                    f"[traces] langfuse write to {self.target} failed: "
                    f"{self.last_error}. Continuing without Langfuse tracing."
                )
            return False

    def _emit(self, record: TraceRecord) -> None:
        root = self.client.start_observation(
            name=f"investigation:{record.case_id or 'adhoc'}",
            as_type="agent",
            input={"alert": record.alert, "window": [record.window_start, record.window_end]},
            output=record.diagnosis,
            metadata={
                "session_id": record.session_id,
                "run_label": record.run_label,
                "provider": record.provider,
                "prompt_version": record.prompt_version,
                "memory_enabled": record.memory_enabled,
                "stop_reason": record.stop_reason,
                "case_id": record.case_id,
                "memory_hits": record.memory_hits,
                # Kept as metadata rather than a score: correctness is graded
                # by the eval harness against planted ground truth, not here.
                "tool_errors": record.tool_errors,
            },
            level="ERROR" if record.stop_reason != "completed" else "DEFAULT",
            status_message=record.error,
            version=record.prompt_version,
        )
        try:
            for step in record.steps:
                self._emit_step(root, record, step)
            # The model call, carrying usage so Langfuse computes cost itself
            # rather than trusting a number we pass in.
            if record.input_tokens or record.output_tokens:
                generation = root.start_observation(
                    name="model",
                    as_type="generation",
                    model=record.model,
                    input={"prompt_version": record.prompt_version},
                    output=record.diagnosis,
                    usage_details={
                        "input": record.input_tokens,
                        "output": record.output_tokens,
                        "cache_read_input_tokens": record.cache_read_tokens,
                    },
                    metadata={"wall_seconds": record.wall_seconds},
                )
                generation.end()
        finally:
            root.end()

    def _emit_step(self, parent: Any, record: TraceRecord, step: dict[str, Any]) -> None:
        span = parent.start_observation(
            name=f"step {step.get('index', '?')}",
            as_type="span",
            input=None,
            output=step.get("text") or None,
            metadata={
                "input_tokens": step.get("input_tokens"),
                "output_tokens": step.get("output_tokens"),
            },
        )
        try:
            for call in step.get("tool_calls", []):
                tool = span.start_observation(
                    name=call.get("name", "tool"),
                    as_type="tool",
                    input=call.get("arguments"),
                    output=call.get("result"),
                    level="ERROR" if call.get("is_error") else "DEFAULT",
                    metadata={"duration_ms": call.get("duration_ms")},
                )
                tool.end()
        finally:
            span.end()

    # -- lifecycle --------------------------------------------------------

    def flush(self) -> None:
        """Langfuse batches in the background; a short-lived script must flush."""
        if self.enabled:
            try:
                self.client.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self.enabled:
            try:
                self.client.shutdown()
            except Exception:
                pass
            finally:
                self.client = None

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "langfuse", "target": self.target,
            "written": self.written, "failures": self.failures,
            "last_error": self.last_error,
        }
