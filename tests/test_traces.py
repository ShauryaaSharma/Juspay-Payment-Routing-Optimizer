"""Config and trace-persistence tests.

Two things get disproportionate attention here:

* **Secrets never leak.** A DSN contains a password, and traces are the thing
  most likely to be shipped somewhere else. There are tests asserting it does
  not appear in any logged or stored surface.
* **Tracing never breaks a run.** Observability failing is an annoyance;
  observability taking down incident response is an outage. Every write path
  is tested for failing soft.
"""

import json
import os

import pytest

from src.agent.config import (
    Settings,
    TraceSettings,
    load_dotenv,
    redact_dsn,
    reset_settings,
)
from src.agent.schemas import Diagnosis, InvestigationResult, Step, ToolCall, Usage
from src.agent.traces import TraceRecord, TraceStore, build_record, new_session_id

SECRET_DSN = "postgresql://admin:sup3rs3cret@db.example.com:5432/agent?sslmode=require"


@pytest.fixture
def clean_env(monkeypatch):
    """Isolate each test from the developer's real environment and .env file."""
    for key in list(os.environ):
        if key.startswith(("AGENT_", "TRACE_", "CONSTRAINT_")) or key == "DATABASE_URL":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    reset_settings()
    yield monkeypatch
    # Clear rather than re-parse: monkeypatch has not yet undone the env, and
    # tests that deliberately set an invalid value would fail in teardown.
    reset_settings()


def _result(diagnosis=True, stop_reason="completed"):
    d = Diagnosis(
        scope="issuer_specific", primary_gateway="PG-Delta", affected_issuer="HDFC",
        confidence=0.7, summary="s", evidence=["e"], recommended_action="a",
    ) if diagnosis else None
    step = Step(
        index=0,
        tool_calls=[ToolCall("segment_failures", {"gateway": "PG-Delta"}, {"ok": 1}, False, 1.5)],
        text="", input_tokens=100, output_tokens=50,
    )
    usage = Usage(input_tokens=100, output_tokens=50, steps=1, tool_calls=1, wall_seconds=0.5)
    return InvestigationResult(d, [step], usage, stop_reason)


# -- config ---------------------------------------------------------------


def test_defaults_reproduce_the_measured_behaviour(clean_env):
    """An unset variable must never change a published result."""
    cfg = Settings.from_env()
    assert cfg.model.model == "claude-opus-5"
    assert cfg.loop.max_steps == 8
    assert cfg.loop.max_tool_calls == 16
    assert cfg.memory.enabled is True
    assert cfg.constraints.ttl_minutes == 480
    assert cfg.constraints.canary_rate == 0.02
    assert cfg.constraints.min_confidence == 0.6


def test_env_overrides_are_typed(clean_env):
    clean_env.setenv("AGENT_MAX_STEPS", "3")
    clean_env.setenv("AGENT_WALL_SECONDS", "12.5")
    clean_env.setenv("AGENT_MEMORY_ENABLED", "false")
    cfg = Settings.from_env()
    assert cfg.loop.max_steps == 3
    assert cfg.loop.wall_seconds == 12.5
    assert cfg.memory.enabled is False


def test_malformed_numeric_env_fails_loudly(clean_env):
    """Better a clear error at startup than a silent fallback to a default."""
    clean_env.setenv("AGENT_MAX_STEPS", "eight")
    with pytest.raises(ValueError, match="AGENT_MAX_STEPS"):
        Settings.from_env()


@pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "yes", "on"])
def test_boolean_parsing_accepts_common_spellings(clean_env, truthy):
    clean_env.setenv("AGENT_MEMORY_ENABLED", truthy)
    assert Settings.from_env().memory.enabled is True


def test_dsn_alone_selects_the_postgres_backend(clean_env):
    """Configuring a database and silently not using it is the surprising outcome."""
    clean_env.setenv("TRACE_DSN", SECRET_DSN)
    assert Settings.from_env().traces.backend == "postgres"


def test_database_url_is_accepted_as_an_alias(clean_env):
    clean_env.setenv("DATABASE_URL", SECRET_DSN)
    assert Settings.from_env().traces.dsn == SECRET_DSN


def test_postgres_backend_without_a_dsn_is_rejected(clean_env):
    clean_env.setenv("TRACE_BACKEND", "postgres")
    with pytest.raises(ValueError, match="TRACE_DSN"):
        Settings.from_env()


def test_unknown_backend_is_rejected(clean_env):
    clean_env.setenv("TRACE_BACKEND", "mongodb")
    with pytest.raises(ValueError, match="none/jsonl/postgres"):
        Settings.from_env()


def test_dotenv_does_not_override_a_real_environment_variable(clean_env, tmp_path):
    """A deployed secret must never be shadowed by a checked-out file."""
    env_file = tmp_path / ".env"
    env_file.write_text("AGENT_MODEL=from-file\n")
    clean_env.setenv("AGENT_MODEL", "from-environment")
    load_dotenv(str(env_file))
    assert os.environ["AGENT_MODEL"] == "from-environment"


def test_dotenv_parses_comments_quotes_and_blanks(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\n\nAGENT_RUN_LABEL="quoted value"\nNOT_A_PAIR\n')
    monkeypatch.delenv("AGENT_RUN_LABEL", raising=False)
    loaded = load_dotenv(str(env_file))
    assert loaded == {"AGENT_RUN_LABEL": "quoted value"}


# -- secrets --------------------------------------------------------------


def test_redact_dsn_strips_credentials_but_keeps_the_target():
    redacted = redact_dsn(SECRET_DSN)
    assert "sup3rs3cret" not in redacted
    assert "admin" not in redacted
    assert "db.example.com:5432/agent" in redacted


def test_describe_never_contains_the_password(clean_env):
    clean_env.setenv("TRACE_DSN", SECRET_DSN)
    described = json.dumps(Settings.from_env().describe())
    assert "sup3rs3cret" not in described


def test_store_target_never_contains_the_password():
    store = TraceStore(TraceSettings(backend="postgres", dsn=SECRET_DSN))
    assert "sup3rs3cret" not in store.target
    assert "sup3rs3cret" not in json.dumps(store.stats())


# -- trace records --------------------------------------------------------


def test_record_captures_the_diagnosis_and_the_cost(clean_env):
    record = build_record(
        _result(), session_id="s1", alert="a", window=(10, 20), case_id="c1",
        provider="test", prompt_version="v2-procedural", memory_enabled=True,
    )
    assert record.scope == "issuer_specific"
    assert record.primary_gateway == "PG-Delta"
    assert record.affected_issuer == "HDFC"
    assert record.case_id == "c1"
    assert record.window_start == 10 and record.window_end == 20
    assert record.tool_calls == 1
    # 100 in @ $5/Mtok + 50 out @ $25/Mtok
    assert record.cost_usd == pytest.approx(100 / 1e6 * 5 + 50 / 1e6 * 25)


def test_failed_investigations_are_still_traced(clean_env):
    """The failures are the traces most worth having later."""
    record = build_record(
        _result(diagnosis=False, stop_reason="budget_steps"), session_id="s1"
    )
    assert record.stop_reason == "budget_steps"
    assert record.diagnosis is None
    assert record.scope is None


def test_tool_results_can_be_excluded(clean_env):
    with_results = build_record(_result(), session_id="s", include_tool_results=True)
    without = build_record(_result(), session_id="s", include_tool_results=False)
    assert "result" in with_results.steps[0]["tool_calls"][0]
    assert "result" not in without.steps[0]["tool_calls"][0]
    # Arguments are kept either way -- they are what makes a trace replayable.
    assert without.steps[0]["tool_calls"][0]["arguments"] == {"gateway": "PG-Delta"}


def test_session_ids_are_unique_and_sortable():
    a, b = new_session_id("eval"), new_session_id("eval")
    assert a != b
    assert a.startswith("eval-")


# -- backends -------------------------------------------------------------


def test_jsonl_backend_appends_one_line_per_trace(tmp_path, clean_env):
    path = str(tmp_path / "sub" / "traces.jsonl")
    store = TraceStore(TraceSettings(backend="jsonl", path=path))
    for _ in range(3):
        assert store.write(build_record(_result(), session_id="s"))
    lines = open(path, encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["scope"] == "issuer_specific"
    assert store.written == 3 and store.failures == 0


def test_disabled_backend_writes_nothing(tmp_path):
    store = TraceStore(TraceSettings(backend="none"))
    assert store.write(build_record(_result(), session_id="s")) is False
    assert store.written == 0


def test_a_broken_backend_never_raises(clean_env, capsys):
    """Observability failing must not take down incident response."""
    store = TraceStore(TraceSettings(backend="jsonl", path="/nonexistent\x00/bad.jsonl"))
    assert store.write(build_record(_result(), session_id="s")) is False
    assert store.failures == 1
    assert store.last_error
    # Warns once, then stays quiet rather than flooding the log.
    store.write(build_record(_result(), session_id="s"))
    assert store.failures == 2
    assert capsys.readouterr().out.count("[traces]") == 1


def test_postgres_write_failure_is_captured_not_raised(clean_env, monkeypatch):
    """Covers both a missing driver and an unreachable server.

    Originally this used a real localhost DSN and assumed psycopg was absent.
    With the driver installed it attempted an actual TCP connection and hung
    the suite -- which is exactly the production failure the connect timeout
    below now bounds. The connection is stubbed so the test asserts the
    error-handling contract rather than the local machine's database setup.
    """
    store = TraceStore(TraceSettings(backend="postgres", dsn=SECRET_DSN))
    monkeypatch.setattr(
        store, "_connect", lambda: (_ for _ in ()).throw(OSError("connection refused"))
    )
    assert store.write(build_record(_result(), session_id="s")) is False
    assert store.failures == 1
    assert "connection refused" in store.last_error


def test_postgres_connect_timeout_defaults_to_seconds_not_minutes(clean_env):
    """An unreachable trace DB must not stall an investigation."""
    assert TraceSettings().connect_timeout == 5
    clean_env.setenv("TRACE_DSN", SECRET_DSN)
    clean_env.setenv("TRACE_CONNECT_TIMEOUT", "2")
    assert Settings.from_env().traces.connect_timeout == 2


def test_ensure_schema_is_a_noop_for_non_postgres_backends():
    store = TraceStore(TraceSettings(backend="jsonl", path="results/x.jsonl"))
    assert "nothing to create" in store.ensure_schema()


# -- integration ----------------------------------------------------------


def test_investigator_writes_a_trace_per_investigation(tmp_path, clean_env):
    from src.agent import Investigator, InvestigatorConfig
    from src.agent.evals.cases import build_cases, telemetry_for
    from src.agent.llm import BaselinePolicyClient

    path = str(tmp_path / "traces.jsonl")
    store = TraceStore(TraceSettings(backend="jsonl", path=path))
    case = build_cases()[0]

    investigator = Investigator(
        store=telemetry_for(case), client=BaselinePolicyClient(),
        config=InvestigatorConfig(), traces=store, session_id="test-session",
    )
    investigator.investigate(case.alert, case.window, case_id=case.case_id)

    row = json.loads(open(path, encoding="utf-8").readline())
    assert row["session_id"] == "test-session"
    assert row["case_id"] == case.case_id
    assert row["provider"] == "baseline-policy"
    assert row["prompt_version"] == "v2-procedural"
    assert row["stop_reason"] == "completed"
    assert len(row["steps"]) >= 1
    assert row["steps"][0]["tool_calls"][0]["name"] == "get_fleet_health"


def test_investigation_still_succeeds_when_tracing_is_broken(tmp_path, clean_env):
    from src.agent import Investigator, InvestigatorConfig
    from src.agent.evals.cases import build_cases, telemetry_for
    from src.agent.llm import BaselinePolicyClient

    store = TraceStore(TraceSettings(backend="jsonl", path="/nonexistent\x00/bad.jsonl"))
    case = build_cases()[0]
    result = Investigator(
        store=telemetry_for(case), client=BaselinePolicyClient(),
        config=InvestigatorConfig(), traces=store,
    ).investigate(case.alert, case.window)

    assert result.diagnosis is not None  # the run is unaffected
    assert store.failures == 1


# -- documentation consistency -------------------------------------------


def test_trace_query_cookbook_ground_truth_matches_the_eval_cases():
    """docs/TRACE_QUERIES.md inlines the answer key; keep it honest.

    Two accuracy queries in the cookbook hardcode the expected scope, gateway,
    and issuer for every eval case, so they can be run standalone against a
    database. That is duplicated state and it will drift the moment a case is
    added or changed -- at which point the queries would silently report the
    wrong accuracy. This test fails instead.
    """
    import re

    from src.agent.evals.cases import build_cases

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc = open(os.path.join(root, "docs", "TRACE_QUERIES.md"), encoding="utf-8").read()

    expected = {
        c.case_id: (c.expected_scope, c.expected_gateway, c.expected_issuer)
        for c in build_cases()
    }
    ctes = re.findall(
        r"WITH truth\(case_id, scope, gateway, issuer\) AS \(VALUES(.*?)\n\)", doc, re.S
    )
    assert ctes, "no ground-truth CTE found in the cookbook"

    row = re.compile(
        r"\('([^']+)'\s*,\s*'([^']+)'\s*,\s*('[^']*'|NULL)\s*,\s*('[^']*'|NULL)\s*\)"
    )
    for index, cte in enumerate(ctes):
        parsed = {
            m[0]: (
                m[1],
                None if m[2] == "NULL" else m[2].strip("'"),
                None if m[3] == "NULL" else m[3].strip("'"),
            )
            for m in row.findall(cte)
        }
        assert parsed == expected, f"cookbook CTE {index} has drifted from build_cases()"


def test_trace_query_cookbook_only_references_real_columns():
    """A typo'd column name in the docs is a query that fails on first use."""
    import re

    from src.agent.traces import _COLUMNS

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc = open(os.path.join(root, "docs", "TRACE_QUERIES.md"), encoding="utf-8").read()

    # The schema table's first column lists column names, one row per line and
    # sometimes several per row ("`trace_id`, `session_id`"). Collect them all.
    documented: set[str] = set()
    for line in doc.splitlines():
        if not line.startswith("| `"):
            continue
        first_cell = line.split("|")[1]
        documented.update(re.findall(r"`([a-z_]+)`", first_cell))

    assert documented, "no schema table found in the cookbook"
    unknown = documented - set(_COLUMNS)
    assert not unknown, f"cookbook documents columns that do not exist: {sorted(unknown)}"

    # And the reverse: a column nobody documented is a column nobody will query.
    undocumented = set(_COLUMNS) - documented
    assert not undocumented, f"columns missing from the cookbook: {sorted(undocumented)}"
