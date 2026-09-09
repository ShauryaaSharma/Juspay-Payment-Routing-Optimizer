"""Trace persistence: what the agent actually did, kept.

An investigation already produces a rich record -- every step, every tool call
with its arguments and result, token counts, timings, the stop reason. Until
now that was discarded when the process exited. This module writes it down.

Three reasons it earns its place:

* **Debugging.** A wrong diagnosis is nearly impossible to explain from the
  answer alone. The trace shows which tool returned what, and where the
  reasoning turned.
* **Eval material.** The best eval cases start as real traces someone flagged.
  A trace store is the pipeline from "that looked wrong" to "that is now a
  regression test".
* **Cost and latency over time.** Tokens and wall-clock per diagnosis, tracked
  per prompt version, is how you notice that a prompt edit tripled spend.

Two backends behind one interface: JSONL for local runs, Postgres for a shared
database. Credentials come from the environment only -- see `config.py`.

**Tracing never fails a run.** Every write is wrapped: a broken database is an
observability outage, not an incident-response outage. Failures are counted and
surfaced once, not raised.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import Settings, TraceSettings, redact_dsn, settings
from .schemas import InvestigationResult

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    trace_id        TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL,
    run_label       TEXT,
    provider        TEXT,
    model           TEXT,
    prompt_version  TEXT,
    memory_enabled  BOOLEAN,
    case_id         TEXT,
    alert           TEXT,
    window_start    INTEGER,
    window_end      INTEGER,
    stop_reason     TEXT,
    error           TEXT,
    scope           TEXT,
    primary_gateway TEXT,
    affected_issuer TEXT,
    confidence      DOUBLE PRECISION,
    steps_count     INTEGER,
    tool_calls      INTEGER,
    tool_errors     INTEGER,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cache_read_tokens INTEGER,
    wall_seconds    DOUBLE PRECISION,
    cost_usd        DOUBLE PRECISION,
    diagnosis       JSONB,
    steps           JSONB,
    memory_hits     JSONB,
    extra           JSONB
);
CREATE INDEX IF NOT EXISTS {table}_session_idx ON {table} (session_id);
CREATE INDEX IF NOT EXISTS {table}_created_idx ON {table} (created_at DESC);
CREATE INDEX IF NOT EXISTS {table}_case_idx ON {table} (case_id);
"""

# Flattened columns are duplicated out of the JSONB blobs on purpose: these are
# the fields you aggregate over (cost by prompt version, error rate by day),
# and doing that through JSONB extraction on every query is needlessly slow.
_COLUMNS = (
    "trace_id", "session_id", "created_at", "run_label", "provider", "model",
    "prompt_version", "memory_enabled", "case_id", "alert", "window_start",
    "window_end", "stop_reason", "error", "scope", "primary_gateway",
    "affected_issuer", "confidence", "steps_count", "tool_calls", "tool_errors",
    "input_tokens", "output_tokens", "cache_read_tokens", "wall_seconds",
    "cost_usd", "diagnosis", "steps", "memory_hits", "extra",
)
_JSON_COLUMNS = frozenset({"diagnosis", "steps", "memory_hits", "extra"})


@dataclass
class TraceRecord:
    """One investigation, flattened for storage."""

    trace_id: str
    session_id: str
    created_at: str
    run_label: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    memory_enabled: bool = False
    case_id: str | None = None
    alert: str | None = None
    window_start: int | None = None
    window_end: int | None = None
    stop_reason: str | None = None
    error: str | None = None
    scope: str | None = None
    primary_gateway: str | None = None
    affected_issuer: str | None = None
    confidence: float | None = None
    steps_count: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    wall_seconds: float = 0.0
    cost_usd: float = 0.0
    diagnosis: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    memory_hits: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_record(
    result: InvestigationResult,
    *,
    session_id: str,
    alert: str | None = None,
    window: tuple[int, int] | None = None,
    case_id: str | None = None,
    provider: str | None = None,
    prompt_version: str | None = None,
    memory_enabled: bool = False,
    config: Settings | None = None,
    include_tool_results: bool = True,
    extra: dict[str, Any] | None = None,
) -> TraceRecord:
    """Turn an InvestigationResult into a storable row."""
    cfg = config or settings()
    usage = result.usage
    diagnosis = result.diagnosis

    steps: list[dict[str, Any]] = []
    for step in result.steps:
        calls = []
        for call in step.tool_calls:
            entry: dict[str, Any] = {
                "name": call.name,
                "arguments": call.arguments,
                "is_error": call.is_error,
                "duration_ms": call.duration_ms,
            }
            if include_tool_results:
                entry["result"] = call.result
            calls.append(entry)
        steps.append({
            "index": step.index,
            "text": step.text,
            "input_tokens": step.input_tokens,
            "output_tokens": step.output_tokens,
            "tool_calls": calls,
        })

    return TraceRecord(
        trace_id=uuid.uuid4().hex,
        session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        run_label=cfg.run_label,
        provider=provider,
        model=cfg.model.model,
        prompt_version=prompt_version,
        memory_enabled=memory_enabled,
        case_id=case_id,
        alert=alert,
        window_start=window[0] if window else None,
        window_end=window[1] if window else None,
        stop_reason=result.stop_reason,
        error=result.error,
        scope=diagnosis.scope if diagnosis else None,
        primary_gateway=diagnosis.primary_gateway if diagnosis else None,
        affected_issuer=diagnosis.affected_issuer if diagnosis else None,
        confidence=diagnosis.confidence if diagnosis else None,
        steps_count=usage.steps,
        tool_calls=usage.tool_calls,
        tool_errors=usage.tool_errors,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        wall_seconds=usage.wall_seconds,
        cost_usd=round(
            usage.cost_usd(cfg.model.input_usd_per_mtok, cfg.model.output_usd_per_mtok), 6
        ),
        diagnosis=diagnosis.to_dict() if diagnosis else None,
        steps=steps,
        memory_hits=list(result.memory_hits),
        extra=extra or {},
    )


class TraceStore:
    """Writes trace records. Never raises on a write failure."""

    def __init__(self, config: TraceSettings | None = None) -> None:
        self.config = config or settings().traces
        self._langfuse = None
        if self.config.backend == "langfuse":
            from .langfuse_sink import LangfuseSink

            self._langfuse = LangfuseSink(self.config)
        self.written = 0
        self.failures = 0
        self.last_error: str | None = None
        self._warned = False
        self._conn: Any = None

    # -- public -----------------------------------------------------------

    @property
    def target(self) -> str:
        if self._langfuse is not None:
            return self._langfuse.target
        if self.config.backend == "postgres":
            return redact_dsn(self.config.dsn) + f"/{self.config.table}"
        if self.config.backend == "jsonl":
            return self.config.path
        return "disabled"

    def write(self, record: TraceRecord) -> bool:
        if not self.config.enabled:
            return False
        try:
            if self.config.backend == "langfuse":
                if not self._langfuse.write(record):
                    # The sink already counted and warned; do not double-count.
                    self.failures += 1
                    self.last_error = self._langfuse.last_error
                    return False
            elif self.config.backend == "jsonl":
                self._write_jsonl(record)
            else:
                self._write_postgres(record)
            self.written += 1
            return True
        except Exception as exc:  # observability must never break the run
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            if not self._warned:
                self._warned = True
                print(
                    f"[traces] write to {self.target} failed: {self.last_error}. "
                    f"Continuing without tracing; further failures are counted, not printed."
                )
            return False

    def auth_check(self) -> tuple[bool, str]:
        """Verify the backend accepts writes, without writing a real trace."""
        if self._langfuse is not None:
            return self._langfuse.auth_check()
        return True, f"backend {self.config.backend!r} needs no auth check"

    def ensure_schema(self) -> str:
        """Create the Postgres table and indexes. No-op for other backends."""
        if self.config.backend != "postgres":
            return f"nothing to create for backend {self.config.backend!r}"
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL.format(table=self.config.table))
        conn.commit()
        return f"schema ready at {self.target}"

    def close(self) -> None:
        if self._langfuse is not None:
            # Langfuse batches in a background thread; a short-lived script
            # exits before the queue drains unless it is flushed.
            self._langfuse.flush()
            self._langfuse.close()
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.config.backend,
            "target": self.target,
            "written": self.written,
            "failures": self.failures,
            "last_error": self.last_error,
        }

    # -- backends ---------------------------------------------------------

    def _write_jsonl(self, record: TraceRecord) -> None:
        path = self.config.path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")

    def _connect(self) -> Any:
        if self._conn is not None and not getattr(self._conn, "closed", 0):
            return self._conn
        try:
            import psycopg
        except ImportError as exc:
            raise ImportError(
                "TRACE_BACKEND=postgres needs the psycopg driver: pip install 'psycopg[binary]'"
            ) from exc
        # connect_timeout is a libpq parameter; without it an unreachable host
        # blocks for the OS TCP timeout and the agent waits with it.
        self._conn = psycopg.connect(
            self.config.dsn, connect_timeout=self.config.connect_timeout
        )
        return self._conn

    def _write_postgres(self, record: TraceRecord) -> None:
        conn = self._connect()
        payload = record.to_dict()
        values = [
            json.dumps(payload[c]) if c in _JSON_COLUMNS else payload[c] for c in _COLUMNS
        ]
        placeholders = ", ".join(["%s"] * len(_COLUMNS))
        statement = (
            f"INSERT INTO {self.config.table} ({', '.join(_COLUMNS)}) "
            f"VALUES ({placeholders}) ON CONFLICT (trace_id) DO NOTHING"
        )
        try:
            with conn.cursor() as cur:
                cur.execute(statement, values)
            conn.commit()
        except Exception:
            # A failed statement poisons the transaction; roll back so the
            # connection stays usable for the next write.
            try:
                conn.rollback()
            except Exception:
                self.close()
            raise


def new_session_id(prefix: str = "run") -> str:
    """A human-sortable session id: run-20260908T014233-1a2b3c."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def _main() -> int:
    """`python -m src.agent.traces <init|check|tail>`"""
    import argparse

    ap = argparse.ArgumentParser(description="Trace store maintenance.")
    ap.add_argument("command", choices=["init", "check", "tail"])
    ap.add_argument("-n", type=int, default=5, help="rows for `tail`")
    args = ap.parse_args()

    cfg = settings(refresh=True)
    store = TraceStore(cfg.traces)
    print(f"backend : {cfg.traces.backend}")
    print(f"target  : {store.target}")

    if args.command == "init":
        try:
            print(store.ensure_schema())
        except Exception as exc:
            print(f"failed: {type(exc).__name__}: {exc}")
            return 1
        return 0

    if args.command == "check":
        probe = TraceRecord(
            trace_id=uuid.uuid4().hex, session_id="connectivity-check",
            created_at=datetime.now(timezone.utc).isoformat(), stop_reason="probe",
            extra={"note": "written by `python -m src.agent.traces check`"},
        )
        ok = store.write(probe)
        print("write ok" if ok else f"write failed: {store.last_error}")
        return 0 if ok else 1

    if cfg.traces.backend != "jsonl":
        print("`tail` reads the JSONL backend only; query your database directly.")
        return 1
    if not os.path.exists(cfg.traces.path):
        print("no traces yet")
        return 0
    with open(cfg.traces.path, encoding="utf-8") as fh:
        rows = fh.readlines()[-args.n:]
    for row in rows:
        r = json.loads(row)
        print(
            f"{r['created_at']}  {str(r.get('case_id') or '-'):24s} "
            f"{str(r.get('scope') or r['stop_reason']):18s} "
            f"tools={r['tool_calls']:2d} tokens={r['input_tokens'] + r['output_tokens']:6d} "
            f"${r['cost_usd']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
