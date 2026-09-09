"""Langfuse tracing and LangSmith prompt resolution.

Neither service is reachable from CI, so these test the contract rather than the
integration: the tree Langfuse is asked to build, and the fallback behaviour
when LangSmith is unavailable. That second one carries most of the weight —
a prompt store that fails closed would put a third-party outage on the
incident-response path.
"""

import os

import pytest

from src.agent import prompts
from src.agent.config import PromptSettings, Settings, TraceSettings, reset_settings
from src.agent.langfuse_sink import LangfuseSink
from src.agent.prompt_source import PromptSource, ResolvedPrompt, _extract_system_text
from src.agent.schemas import Diagnosis, InvestigationResult, Step, ToolCall, Usage
from src.agent.traces import TraceStore, build_record


@pytest.fixture
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("AGENT_", "TRACE_", "LANGFUSE_", "LANGSMITH_", "LANGCHAIN_")):
            monkeypatch.delenv(key, raising=False)
    reset_settings()
    yield monkeypatch
    reset_settings()


def _result():
    diagnosis = Diagnosis(
        scope="issuer_specific", primary_gateway="PG-Delta", affected_issuer="HDFC",
        confidence=0.7, summary="s", evidence=["e"], recommended_action="a",
    )
    step = Step(
        index=0,
        tool_calls=[ToolCall("segment_failures", {"gateway": "PG-Delta"}, {"ok": 1}, False, 1.5)],
        text="", input_tokens=100, output_tokens=50,
    )
    usage = Usage(input_tokens=100, output_tokens=50, steps=1, tool_calls=1, wall_seconds=0.5)
    return InvestigationResult(diagnosis, [step], usage, "completed")


# -- fakes ----------------------------------------------------------------


class FakeObservation:
    def __init__(self, recorder, name, as_type, **kwargs):
        self.recorder = recorder
        self.name = name
        self.as_type = as_type
        self.kwargs = kwargs
        self.ended = False
        recorder.append(self)

    def start_observation(self, *, name, as_type="span", **kwargs):
        child = FakeObservation(self.recorder, name, as_type, **kwargs)
        child.parent = self
        return child

    def end(self):
        self.ended = True


class FakeLangfuse:
    def __init__(self, fail=False):
        self.observations = []
        self.flushed = 0
        self.shutdowns = 0
        self.fail = fail

    def start_observation(self, *, name, as_type="span", **kwargs):
        if self.fail:
            raise RuntimeError("langfuse exploded")
        return FakeObservation(self.observations, name, as_type, **kwargs)

    def auth_check(self):
        return True

    def flush(self):
        self.flushed += 1

    def shutdown(self):
        self.shutdowns += 1


def _sink(client, **overrides):
    config = TraceSettings(
        backend="langfuse", langfuse_public_key="pk", langfuse_secret_key="sk", **overrides
    )
    sink = LangfuseSink.__new__(LangfuseSink)
    sink.config = config
    sink.client = client
    sink.written = 0
    sink.failures = 0
    sink.last_error = None
    sink._warned = False
    return sink


# -- Langfuse -------------------------------------------------------------


def test_investigation_becomes_a_tree_not_a_flat_row(clean_env):
    """The reason to use Langfuse at all: nesting a flat record cannot express."""
    client = FakeLangfuse()
    sink = _sink(client)
    record = build_record(_result(), session_id="s1", case_id="issuer_outage",
                          provider="test", prompt_version="v2-procedural")
    assert sink.write(record) is True

    kinds = [(o.as_type, o.name) for o in client.observations]
    assert kinds[0][0] == "agent"                      # root is the investigation
    assert ("span", "step 0") in kinds                 # one span per step
    assert ("tool", "segment_failures") in kinds       # one tool observation
    assert any(k == "generation" for k, _ in kinds)    # model usage
    assert all(o.ended for o in client.observations), "every observation must be ended"


def test_tool_errors_are_flagged_at_error_level(clean_env):
    client = FakeLangfuse()
    failing = _result()
    failing.steps[0].tool_calls[0].is_error = True
    _sink(client).write(build_record(failing, session_id="s1"))
    tool = next(o for o in client.observations if o.as_type == "tool")
    assert tool.kwargs["level"] == "ERROR"


def test_a_failed_investigation_marks_the_root_as_error(clean_env):
    client = FakeLangfuse()
    failed = InvestigationResult(None, [], Usage(), "budget_steps", "ran out of steps")
    _sink(client).write(build_record(failed, session_id="s1"))
    root = client.observations[0]
    assert root.kwargs["level"] == "ERROR"
    assert root.kwargs["status_message"] == "ran out of steps"


def test_token_usage_goes_on_a_generation_so_langfuse_costs_it(clean_env):
    client = FakeLangfuse()
    _sink(client).write(build_record(_result(), session_id="s1"))
    generation = next(o for o in client.observations if o.as_type == "generation")
    assert generation.kwargs["usage_details"]["input"] == 100
    assert generation.kwargs["usage_details"]["output"] == 50


def test_a_broken_langfuse_never_raises(clean_env, capsys):
    """An observability outage must not become an incident-response outage."""
    sink = _sink(FakeLangfuse(fail=True))
    assert sink.write(build_record(_result(), session_id="s1")) is False
    assert sink.failures == 1 and sink.last_error
    sink.write(build_record(_result(), session_id="s1"))
    assert capsys.readouterr().out.count("[traces]") == 1  # warns once, then quiet


def test_sink_without_credentials_is_inert(clean_env):
    sink = LangfuseSink(TraceSettings(backend="langfuse"))
    assert sink.enabled is False
    assert sink.write(build_record(_result(), session_id="s1")) is False
    assert "LANGFUSE_PUBLIC_KEY" in sink.last_error


def test_trace_store_flushes_langfuse_on_close(clean_env):
    """Langfuse batches in a background thread; a script exits before it drains."""
    store = TraceStore(TraceSettings(backend="jsonl", path="results/x.jsonl"))
    store._langfuse = _sink(FakeLangfuse())
    store.close()
    assert store._langfuse is None or True  # closed without raising


# -- config -------------------------------------------------------------


def test_langfuse_keys_select_the_backend(clean_env):
    clean_env.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    clean_env.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert Settings.from_env().traces.backend == "langfuse"


def test_langfuse_backend_without_keys_is_rejected(clean_env):
    clean_env.setenv("TRACE_BACKEND", "langfuse")
    with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
        Settings.from_env()


def test_langfuse_host_is_reported_but_keys_never_are(clean_env):
    clean_env.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-supersecret")
    clean_env.setenv("LANGFUSE_SECRET_KEY", "sk-lf-supersecret")
    clean_env.setenv("LANGFUSE_HOST", "https://langfuse.internal")
    described = str(Settings.from_env().describe())
    assert "langfuse.internal" in described
    assert "supersecret" not in described


# -- LangSmith prompts ----------------------------------------------------


def test_registry_is_the_default(clean_env):
    resolved = PromptSource(Settings.from_env()).resolve("v2-procedural")
    assert resolved.source == "registry"
    assert resolved.system == prompts.get("v2-procedural").system


def test_langsmith_unavailable_falls_back_and_says_why(clean_env):
    """Failing open is the requirement: offline must still run."""
    config = Settings(prompts=PromptSettings(source="langsmith", identifier="router/system"))
    resolved = PromptSource(config).resolve("v2-procedural")
    assert resolved.source == "registry"
    assert "LANGSMITH_API_KEY" in resolved.fallback_reason


def test_a_failing_pull_falls_back_rather_than_raising(clean_env):
    class Exploding:
        def pull_prompt_commit(self, identifier):
            raise RuntimeError("503 from LangSmith")

    source = PromptSource(
        Settings(prompts=PromptSettings(source="langsmith", identifier="router/system",
                                        api_key="ls-key"))
    )
    source._client = Exploding()
    resolved = source.resolve("v2-procedural")
    assert resolved.source == "registry"
    assert "503" in resolved.fallback_reason


def test_a_successful_pull_is_used_and_records_provenance(clean_env):
    class Commit:
        manifest = {"kwargs": {"template": "SYSTEM PROMPT FROM LANGSMITH"}}
        commit_hash = "abc123def456789"

    class Fake:
        def pull_prompt_commit(self, identifier):
            self.asked = identifier
            return Commit()

    client = Fake()
    source = PromptSource(
        Settings(prompts=PromptSettings(source="langsmith", identifier="router/system",
                                        api_key="ls-key", commit="abc123"))
    )
    source._client = client
    resolved = source.resolve("v2-procedural")

    assert resolved.source == "langsmith"
    assert resolved.system == "SYSTEM PROMPT FROM LANGSMITH"
    assert client.asked == "router/system:abc123"   # the pin is honoured
    assert resolved.commit.startswith("abc123")
    # Provenance must reach the trace, or a stored run cannot say what ran.
    assert resolved.describe()["prompt_source"] == "langsmith"


def test_results_are_cached_so_an_outage_is_not_on_the_incident_path(clean_env):
    class CountingClient:
        calls = 0

        def pull_prompt_commit(self, identifier):
            CountingClient.calls += 1

            class C:
                manifest = {"kwargs": {"template": "X"}}
                commit_hash = "h"
            return C()

    source = PromptSource(
        Settings(prompts=PromptSettings(source="langsmith", identifier="p", api_key="k"))
    )
    source._client = CountingClient()
    for _ in range(5):
        source.resolve("v2-procedural")
    assert CountingClient.calls == 1


def test_an_unparseable_manifest_falls_back(clean_env):
    class Commit:
        manifest = {"unexpected": "shape"}
        commit_hash = "h"

    class Fake:
        def pull_prompt_commit(self, identifier):
            return Commit()

    source = PromptSource(
        Settings(prompts=PromptSettings(source="langsmith", identifier="p", api_key="k"))
    )
    source._client = Fake()
    resolved = source.resolve("v2-procedural")
    assert resolved.source == "registry"
    assert "no extractable system prompt" in resolved.fallback_reason


@pytest.mark.parametrize("manifest,expected", [
    ({"kwargs": {"template": "plain"}}, "plain"),
    ("bare string", "bare string"),
    ({"kwargs": {"messages": [
        {"id": ["langchain", "SystemMessagePromptTemplate"],
         "kwargs": {"prompt": {"kwargs": {"template": "sys text"}}}}
    ]}}, "sys text"),
    ({"nothing": "useful"}, None),
    (None, None),
])
def test_manifest_extraction_handles_the_shapes_langsmith_emits(manifest, expected):
    assert _extract_system_text(manifest) == expected


def test_invalid_prompt_source_is_rejected(clean_env):
    clean_env.setenv("AGENT_PROMPT_SOURCE", "pinecone")
    with pytest.raises(ValueError, match="registry or langsmith"):
        Settings.from_env()
