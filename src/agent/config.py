"""Environment-driven configuration.

Every tunable in the agent layer resolves here, and every default reproduces
the behaviour the README's numbers were measured with. That property matters:
if an unset variable silently changed a default, the published results would
depend on the shell you happened to run them in.

Credentials are read from the environment and never from code, never written
to a trace, and never printed. `Settings.describe()` exists so a run can log
its own configuration without leaking a DSN or an API key.

A minimal `.env` loader is included rather than taking a dependency on
python-dotenv -- the project's offline path runs on numpy alone, and a config
file parser is thirty lines. Real environment variables always win over `.env`,
which is what makes the file safe to commit an example of and safe to override
in CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

_DOTENV_LOADED = False


def load_dotenv(path: str = ".env", override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Existing environment variables are preserved unless ``override`` is set,
    so a deployed secret is never shadowed by a stale checked-out file.
    """
    loaded: dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
            loaded[key] = value
    return loaded


def _ensure_dotenv() -> None:
    global _DOTENV_LOADED
    if not _DOTENV_LOADED:
        load_dotenv()
        _DOTENV_LOADED = True


def _str(name: str, default: str) -> str:
    _ensure_dotenv()
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _opt(name: str) -> str | None:
    _ensure_dotenv()
    value = os.environ.get(name)
    return value if value not in (None, "") else None


def _int(name: str, default: int) -> int:
    raw = _opt(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = _opt(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = _opt(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelSettings:
    model: str = "claude-opus-5"
    effort: str = "high"
    max_tokens: int = 16_000
    input_usd_per_mtok: float = 5.00
    output_usd_per_mtok: float = 25.00

    @classmethod
    def from_env(cls) -> "ModelSettings":
        return cls(
            model=_str("AGENT_MODEL", "claude-opus-5"),
            effort=_str("AGENT_EFFORT", "high"),
            max_tokens=_int("AGENT_MAX_TOKENS", 16_000),
            input_usd_per_mtok=_float("AGENT_INPUT_USD_PER_MTOK", 5.00),
            output_usd_per_mtok=_float("AGENT_OUTPUT_USD_PER_MTOK", 25.00),
        )


@dataclass(frozen=True)
class LoopSettings:
    max_steps: int = 8
    max_tool_calls: int = 16
    max_output_tokens: int = 40_000
    wall_seconds: float = 300.0
    max_consecutive_tool_errors: int = 3
    repair_attempts: int = 1

    @classmethod
    def from_env(cls) -> "LoopSettings":
        return cls(
            max_steps=_int("AGENT_MAX_STEPS", 8),
            max_tool_calls=_int("AGENT_MAX_TOOL_CALLS", 16),
            max_output_tokens=_int("AGENT_MAX_OUTPUT_TOKENS", 40_000),
            wall_seconds=_float("AGENT_WALL_SECONDS", 300.0),
            max_consecutive_tool_errors=_int("AGENT_MAX_TOOL_ERRORS", 3),
            repair_attempts=_int("AGENT_REPAIR_ATTEMPTS", 1),
        )


@dataclass(frozen=True)
class MemorySettings:
    enabled: bool = True
    path: str | None = None  # None keeps the store in-process
    recall_k: int = 3
    min_support: int = 2

    @classmethod
    def from_env(cls) -> "MemorySettings":
        return cls(
            enabled=_bool("AGENT_MEMORY_ENABLED", True),
            path=_opt("AGENT_MEMORY_PATH"),
            recall_k=_int("AGENT_MEMORY_RECALL_K", 3),
            min_support=_int("AGENT_MEMORY_MIN_SUPPORT", 2),
        )


@dataclass(frozen=True)
class TraceSettings:
    """Where investigation traces are written.

    ``backend`` is one of:
      none      -- discard (default when nothing is configured)
      jsonl     -- append one JSON object per investigation to ``path``
      postgres  -- insert a row per investigation into ``table`` at ``dsn``

    Setting TRACE_DSN alone is enough: the backend defaults to postgres when a
    DSN is present, because configuring a database and then silently not using
    it is the more surprising outcome.
    """

    backend: str = "none"
    path: str = "results/traces.jsonl"
    dsn: str | None = None
    table: str = "agent_traces"
    include_tool_results: bool = True
    # Seconds to wait for a database connection. Deliberately short: a trace
    # store is observability, and an unreachable database must degrade to "no
    # traces" in seconds rather than stalling an investigation behind libpq's
    # default, which can block for minutes.
    connect_timeout: int = 5

    @classmethod
    def from_env(cls) -> "TraceSettings":
        dsn = _opt("TRACE_DSN") or _opt("DATABASE_URL")
        default_backend = "postgres" if dsn else "jsonl"
        backend = _str("TRACE_BACKEND", default_backend).lower()
        if backend not in {"none", "jsonl", "postgres"}:
            raise ValueError(
                f"TRACE_BACKEND must be one of none/jsonl/postgres, got {backend!r}"
            )
        if backend == "postgres" and not dsn:
            raise ValueError("TRACE_BACKEND=postgres requires TRACE_DSN (or DATABASE_URL)")
        return cls(
            backend=backend,
            path=_str("TRACE_PATH", "results/traces.jsonl"),
            dsn=dsn,
            table=_str("TRACE_TABLE", "agent_traces"),
            include_tool_results=_bool("TRACE_INCLUDE_TOOL_RESULTS", True),
            connect_timeout=_int("TRACE_CONNECT_TIMEOUT", 5),
        )

    @property
    def enabled(self) -> bool:
        return self.backend != "none"


@dataclass(frozen=True)
class ConstraintSettings:
    ttl_minutes: int = 480
    canary_rate: float = 0.02
    min_confidence: float = 0.6

    @classmethod
    def from_env(cls) -> "ConstraintSettings":
        return cls(
            ttl_minutes=_int("CONSTRAINT_TTL_MINUTES", 480),
            canary_rate=_float("CONSTRAINT_CANARY_RATE", 0.02),
            min_confidence=_float("CONSTRAINT_MIN_CONFIDENCE", 0.6),
        )


@dataclass(frozen=True)
class Settings:
    model: ModelSettings = field(default_factory=ModelSettings)
    loop: LoopSettings = field(default_factory=LoopSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    traces: TraceSettings = field(default_factory=TraceSettings)
    constraints: ConstraintSettings = field(default_factory=ConstraintSettings)
    prompt_version: str | None = None
    run_label: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=ModelSettings.from_env(),
            loop=LoopSettings.from_env(),
            memory=MemorySettings.from_env(),
            traces=TraceSettings.from_env(),
            constraints=ConstraintSettings.from_env(),
            prompt_version=_opt("AGENT_PROMPT_VERSION"),
            run_label=_opt("AGENT_RUN_LABEL"),
        )

    def describe(self) -> dict[str, Any]:
        """Loggable configuration. Never includes a secret.

        The DSN is reduced to its host and database name so a run can record
        *which* database it wrote to without recording how to connect to it.
        """
        return {
            "model": self.model.model,
            "effort": self.model.effort,
            "prompt_version": self.prompt_version or "default",
            "memory_enabled": self.memory.enabled,
            "max_steps": self.loop.max_steps,
            "trace_backend": self.traces.backend,
            "trace_target": redact_dsn(self.traces.dsn) if self.traces.dsn else self.traces.path,
            "run_label": self.run_label,
        }


def redact_dsn(dsn: str | None) -> str:
    """Reduce a connection string to host/database, dropping any credentials."""
    if not dsn:
        return ""
    remainder = dsn.split("://", 1)[-1]
    if "@" in remainder:  # strip user:password
        remainder = remainder.split("@", 1)[1]
    host_and_db = remainder.split("?", 1)[0]
    return f"<redacted>@{host_and_db}"


_CACHED: Settings | None = None


def reset_settings() -> None:
    """Drop the cached settings without re-reading the environment.

    Distinct from ``settings(refresh=True)``, which re-parses immediately and
    therefore raises if the environment is currently invalid. Teardown code
    needs to clear the cache unconditionally, so it uses this.
    """
    global _CACHED, _DOTENV_LOADED
    _CACHED = None
    _DOTENV_LOADED = False


def settings(refresh: bool = False) -> Settings:
    """Process-wide settings, resolved once.

    Pass ``refresh=True`` after mutating the environment -- tests do this, and
    it is cheaper than making every call site re-read os.environ.
    """
    global _CACHED, _DOTENV_LOADED
    if refresh:
        _DOTENV_LOADED = False
        _CACHED = None
    if _CACHED is None:
        _CACHED = Settings.from_env()
    return _CACHED
