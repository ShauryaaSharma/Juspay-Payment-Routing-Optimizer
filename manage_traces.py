#!/usr/bin/env python3
"""Trace store maintenance.

    python manage_traces.py check     # verify the configured backend accepts a write
    python manage_traces.py init      # create the Postgres table and indexes
    python manage_traces.py tail      # show the most recent local traces
    python manage_traces.py config    # print resolved settings (secrets redacted)

Configuration comes from the environment or a `.env` file; see `.env.example`.
To point at a hosted database:

    TRACE_DSN=postgresql://user:pass@host:5432/db?sslmode=require
    python manage_traces.py init && python manage_traces.py check

A root-level script rather than `python -m src.agent.traces`: the package
`__init__` already imports the traces module, so running it as `-m` triggers a
double-import RuntimeWarning. Same code, no warning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone

from src.agent.config import settings
from src.agent.traces import TraceRecord, TraceStore


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["check", "init", "tail", "config"])
    ap.add_argument("-n", type=int, default=10, help="rows for `tail`")
    args = ap.parse_args()

    cfg = settings(refresh=True)
    store = TraceStore(cfg.traces)

    if args.command == "config":
        print(json.dumps(cfg.describe(), indent=2))
        return 0

    print(f"backend : {cfg.traces.backend}")
    print(f"target  : {store.target}")

    if args.command == "init":
        try:
            print(store.ensure_schema())
        except Exception as exc:
            print(f"failed: {type(exc).__name__}: {exc}")
            print("\nCheck TRACE_DSN, that the database is reachable, and that the")
            print("user may CREATE TABLE. For most hosted providers the connection")
            print("string needs `?sslmode=require`.")
            return 1
        finally:
            store.close()
        return 0

    if args.command == "check":
        ok, message = store.auth_check()
        print(f"auth    : {message}")
        if not ok:
            store.close()
            return 1
        record = TraceRecord(
            trace_id=uuid.uuid4().hex,
            session_id="connectivity-check",
            created_at=datetime.now(timezone.utc).isoformat(),
            stop_reason="probe",
            extra={"note": "written by manage_traces.py check"},
        )
        ok = store.write(record)
        store.close()
        if ok:
            print("write ok -- a probe row was stored. Safe to delete it.")
            return 0
        print(f"write failed: {store.last_error}")
        print("\nIf the table is missing, run: python manage_traces.py init")
        return 1

    # tail
    if cfg.traces.backend == "langfuse":
        print("`tail` reads the JSONL backend only. Langfuse traces are browsable")
        print(f"at {store.target} -- filter by session_id or case_id.")
        return 1
    if cfg.traces.backend != "jsonl":
        print("`tail` reads the JSONL backend only. For Postgres, query directly:")
        print(f"  SELECT created_at, case_id, scope, tool_calls, cost_usd")
        print(f"  FROM {cfg.traces.table} ORDER BY created_at DESC LIMIT {args.n};")
        return 1
    if not os.path.exists(cfg.traces.path):
        print("no traces yet")
        return 0
    with open(cfg.traces.path, encoding="utf-8") as fh:
        rows = fh.readlines()[-args.n:]
    print(f"\n{'when':22s} {'case':22s} {'outcome':18s} {'tools':>5s} {'tokens':>7s} {'cost':>9s}")
    print("-" * 92)
    for line in rows:
        r = json.loads(line)
        tokens = r["input_tokens"] + r["output_tokens"]
        print(
            f"{r['created_at'][:19]:22s} {str(r.get('case_id') or '-')[:21]:22s} "
            f"{str(r.get('scope') or r['stop_reason'])[:17]:18s} "
            f"{r['tool_calls']:>5d} {tokens:>7d} ${r['cost_usd']:>8.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
