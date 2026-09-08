"""Loop tests: every termination path, every recovery path.

An agent loop's bugs do not show up on the happy path -- they show up when a
budget runs out, a tool errors repeatedly, or the model returns something
unparseable. Those are the branches under test here, driven by scripted fake
clients so each failure is reproducible rather than waited for.
"""

import json

import pytest

from src.agent.llm import LLMResponse, PendingToolCall
from src.agent.loop import AgentLoop, LoopBudget
from src.agent.schemas import Diagnosis, DiagnosisFormatError
from src.agent.tools import ToolDispatcher

VALID_DIAGNOSIS = {
    "scope": "single_gateway",
    "primary_gateway": "PG-Bravo",
    "affected_issuer": None,
    "confidence": 0.8,
    "summary": "PG-Bravo collapsed.",
    "evidence": ["PG-Bravo at 5%"],
    "recommended_action": "Drain it.",
}


class ScriptedClient:
    """Replays a fixed list of responses, then repeats the last one forever."""

    name = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.last_messages = None

    def create(self, system, messages, tools):
        self.calls += 1
        self.last_messages = list(messages)
        index = min(self.calls - 1, len(self.responses) - 1)
        return self.responses[index]


class FakeStore:
    """Minimal telemetry stand-in.

    Mirrors the real store's *error* behaviour as well as its happy path -- an
    unknown gateway raises KeyError here exactly as it does in TelemetryStore.
    A fake that only models success would let the loop's error-recovery tests
    pass against a dispatcher that cannot actually catch what production throws.
    """

    names = ["PG-Alpha", "PG-Bravo"]

    def _index(self, gateway):
        lookup = {n.lower(): i for i, n in enumerate(self.names)}
        key = str(gateway).strip().lower()
        if key not in lookup:
            raise KeyError(f"unknown gateway {gateway!r}; valid: {', '.join(self.names)}")
        return lookup[key]

    def fleet_health(self, start_minute, end_minute):
        return {"fleet_success_rate": 0.9, "per_gateway": []}

    def list_gateways(self):
        return {"gateways": [{"name": n} for n in self.names]}

    def gateway_health(self, gateway, start_minute, end_minute):
        return {"gateway": self.names[self._index(gateway)], "success_rate": 0.9}

    def segment_failures(self, gateway, start_minute, end_minute, by="issuer"):
        return {"gateway": self.names[self._index(gateway)], "segments": []}

    def compare_windows(self, gateway, *windows):
        return {"gateway": self.names[self._index(gateway)], "success_rate_delta": 0.0}

    def traffic_shift(self, *windows):
        return {"shifts": []}


def _tool_response(name="get_fleet_health", **args):
    if name == "get_fleet_health":
        args = args or {"start_minute": 0, "end_minute": 100}
    return LLMResponse(
        tool_calls=[PendingToolCall(id=f"t{id(args)}", name=name, input=args)],
        stop_reason="tool_use",
    )


def _final(payload=None):
    return LLMResponse(text=json.dumps(payload or VALID_DIAGNOSIS), stop_reason="end_turn")


def _loop(responses, budget=None):
    return AgentLoop(ScriptedClient(responses), ToolDispatcher(FakeStore()), budget)


def test_happy_path_returns_a_diagnosis():
    result = _loop([_tool_response(), _final()]).run("sys", "brief")
    assert result.stop_reason == "completed"
    assert result.diagnosis.primary_gateway == "PG-Bravo"
    assert result.usage.tool_calls == 1


def test_step_budget_forces_an_answer_rather_than_returning_nothing():
    """Budget exhaustion must degrade to a low-confidence answer, not a crash."""
    loop = _loop([_tool_response(), _tool_response(), _final()], LoopBudget(max_steps=2))
    result = loop.run("sys", "brief")
    assert result.diagnosis is not None
    assert result.stop_reason == "completed"


def test_tool_call_budget_binds():
    responses = [_tool_response(start_minute=i, end_minute=i + 10) for i in range(10)]
    loop = _loop(responses + [_final()], LoopBudget(max_steps=20, max_tool_calls=3))
    result = loop.run("sys", "brief")
    assert result.usage.tool_calls <= 4  # the forcing turn may not add one


def test_loop_always_terminates_even_when_the_model_never_stops():
    """A model that only ever calls tools must not spin forever."""
    loop = _loop([_tool_response(start_minute=0, end_minute=1)], LoopBudget(max_steps=4))
    result = loop.run("sys", "brief")
    assert result.stop_reason in {"budget_steps", "invalid_output", "completed"}
    assert result.usage.steps <= 6


def test_duplicate_tool_call_gets_a_nudge_not_a_repeat():
    """Identical repeated calls are how these loops stall."""
    dup = _tool_response(start_minute=0, end_minute=100)
    loop = _loop([dup, dup, _final()], LoopBudget(max_steps=6))
    result = loop.run("sys", "brief")
    second = result.steps[1].tool_calls[0]
    assert "already called" in json.dumps(second.result)


def test_repeated_tool_errors_break_the_loop():
    """Distinct failing calls must trip the circuit breaker."""
    bads = [
        _tool_response("get_gateway_health", gateway=f"NOPE{i}", start_minute=0, end_minute=10)
        for i in range(5)
    ]
    loop = _loop(bads, LoopBudget(max_steps=10, max_consecutive_tool_errors=3))
    result = loop.run("sys", "brief")
    assert result.stop_reason == "tool_error_limit"
    assert result.diagnosis is None


def test_identical_failing_call_is_deduplicated_rather_than_counted():
    """Documents a real interaction between two safety mechanisms.

    Repeating the *same* bad call hits duplicate suppression first, which
    returns a nudge instead of re-running the tool -- so the consecutive-error
    counter resets and the error circuit breaker never trips. The loop still
    terminates, via the step budget rather than the error limit. Both paths are
    bounded, but only one of them is the one you would guess.
    """
    bad = _tool_response("get_gateway_health", gateway="NOPE", start_minute=0, end_minute=10)
    loop = _loop([bad] * 6, LoopBudget(max_steps=4, max_consecutive_tool_errors=3))
    result = loop.run("sys", "brief")
    assert result.stop_reason != "tool_error_limit"
    assert result.usage.tool_errors == 1  # only the first actually ran
    assert result.usage.steps <= 6


def test_a_single_tool_error_is_recoverable():
    """One bad call must not end the investigation."""
    bad = _tool_response("get_gateway_health", gateway="NOPE", start_minute=0, end_minute=10)
    loop = _loop([bad, _final()], LoopBudget(max_steps=6))
    result = loop.run("sys", "brief")
    assert result.stop_reason == "completed"
    assert result.usage.tool_errors == 1


def test_unparseable_output_triggers_one_repair_then_gives_up():
    junk = LLMResponse(text="I think PG-Bravo is down.", stop_reason="end_turn")
    loop = _loop([junk, junk, junk], LoopBudget(max_steps=8, repair_attempts=1))
    result = loop.run("sys", "brief")
    assert result.stop_reason == "invalid_output"
    assert result.diagnosis is None


def test_repair_attempt_can_succeed():
    junk = LLMResponse(text="not json", stop_reason="end_turn")
    loop = _loop([junk, _final()], LoopBudget(max_steps=8, repair_attempts=1))
    result = loop.run("sys", "brief")
    assert result.stop_reason == "completed"


def test_refusal_is_reported_not_raised():
    refusal = LLMResponse(stop_reason="refusal", refusal_category="cyber")
    result = _loop([refusal]).run("sys", "brief")
    assert result.stop_reason == "refusal"
    assert "cyber" in result.error


def test_provider_exception_is_captured_as_a_result():
    class Exploding:
        name = "boom"

        def create(self, system, messages, tools):
            raise RuntimeError("API down")

    loop = AgentLoop(Exploding(), ToolDispatcher(FakeStore()))
    result = loop.run("sys", "brief")
    assert result.stop_reason == "provider_error"
    assert "API down" in result.error


def test_tool_results_are_returned_in_a_single_user_message():
    """Splitting parallel results teaches the model to stop parallelising."""
    parallel = LLMResponse(
        tool_calls=[
            PendingToolCall(id="a", name="get_fleet_health",
                            input={"start_minute": 0, "end_minute": 10}),
            PendingToolCall(id="b", name="list_gateways", input={}),
        ],
        stop_reason="tool_use",
    )
    client = ScriptedClient([parallel, _final()])
    AgentLoop(client, ToolDispatcher(FakeStore())).run("sys", "brief")
    tool_messages = [
        m for m in client.last_messages
        if isinstance(m.get("content"), list)
        and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
    ]
    assert len(tool_messages) == 1
    assert len(tool_messages[0]["content"]) == 2


def test_diagnosis_rejects_malformed_payloads():
    with pytest.raises(DiagnosisFormatError):
        Diagnosis.from_payload({"scope": "single_gateway"})  # missing fields
    with pytest.raises(DiagnosisFormatError):
        Diagnosis.from_payload({**VALID_DIAGNOSIS, "scope": "made_up"})
    with pytest.raises(DiagnosisFormatError):
        Diagnosis.from_json("{not json")


def test_diagnosis_clamps_confidence_into_range():
    d = Diagnosis.from_payload({**VALID_DIAGNOSIS, "confidence": 4.2})
    assert d.confidence == 1.0
