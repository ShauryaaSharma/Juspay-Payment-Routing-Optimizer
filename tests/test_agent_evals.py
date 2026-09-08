"""Tests for memory, tools, graders, and the eval set itself.

The eval-set tests matter more than they look. A grader that silently passes
everything, or a case whose ground truth is unreachable from the telemetry,
turns the whole harness into a number generator. These pin both.
"""

import pytest

from src.agent.evals.cases import build_cases, telemetry_for
from src.agent.evals.graders import grade
from src.agent.evals.harness import ArmConfig, aggregate, check_gate, run_arm
from src.agent.llm import BaselinePolicyClient
from src.agent.memory import MemoryStore
from src.agent.schemas import Diagnosis, InvestigationResult, Usage
from src.agent.telemetry import MIN_SAMPLES_FOR_SR, TelemetryStore, format_clock
from src.agent.tools import ToolDispatcher, tool_definitions


def _diagnosis(scope="single_gateway", gateway="PG-Bravo", issuer=None, confidence=0.8):
    return Diagnosis(
        scope=scope, primary_gateway=gateway, affected_issuer=issuer,
        confidence=confidence, summary="s", evidence=["e"], recommended_action="a",
    )


def _result(diagnosis, tool_names=()):
    from src.agent.schemas import Step, ToolCall

    step = Step(index=0, tool_calls=[ToolCall(n, {}, {}) for n in tool_names])
    return InvestigationResult(
        diagnosis=diagnosis, steps=[step], usage=Usage(), stop_reason="completed"
    )


# -- tool surface ---------------------------------------------------------


def test_every_tool_schema_is_strict_and_closed():
    """Strict mode requires additionalProperties:false and a required list."""
    for tool in tool_definitions():
        schema = tool["input_schema"]
        assert tool["strict"] is True, tool["name"]
        assert schema["additionalProperties"] is False, tool["name"]
        assert set(schema["required"]) <= set(schema["properties"]), tool["name"]
        assert tool["description"].strip(), tool["name"]


def test_tool_definitions_are_stable_across_calls():
    """Tools render before the cached prefix; churn here kills the cache."""
    assert tool_definitions() == tool_definitions()


def test_dispatcher_reports_unknown_gateway_as_an_error_not_an_exception():
    case = build_cases()[0]
    dispatcher = ToolDispatcher(telemetry_for(case))
    call = dispatcher.dispatch(
        "get_gateway_health", {"gateway": "PG-Nonexistent", "start_minute": 0, "end_minute": 100}
    )
    assert call.is_error
    assert "unknown gateway" in call.result["error"]


def test_dispatcher_rejects_an_inverted_window():
    dispatcher = ToolDispatcher(telemetry_for(build_cases()[0]))
    call = dispatcher.dispatch("get_fleet_health", {"start_minute": 500, "end_minute": 100})
    assert call.is_error


def test_dispatcher_rejects_an_unknown_tool():
    dispatcher = ToolDispatcher(telemetry_for(build_cases()[0]))
    assert dispatcher.dispatch("rm_rf", {}).is_error


def test_telemetry_withholds_rates_below_the_sampling_floor():
    """A rate computed from a handful of transactions is noise, not signal."""
    case = build_cases()[0]
    store = telemetry_for(case)
    tiny = store.fleet_health(0, 2)
    for entry in tiny["per_gateway"]:
        if entry["transactions"] < MIN_SAMPLES_FOR_SR:
            assert entry["success_rate"] is None


def test_format_clock_is_one_indexed_by_day():
    assert format_clock(0) == "day 1, 00:00"
    assert format_clock(24 * 60 + 90) == "day 2, 01:30"


# -- eval set integrity ---------------------------------------------------


def test_case_ids_are_unique():
    ids = [c.case_id for c in build_cases()]
    assert len(ids) == len(set(ids))


def test_every_scope_is_represented():
    scopes = {c.expected_scope for c in build_cases()}
    assert scopes == {"single_gateway", "issuer_specific", "fleet_wide", "no_incident"}


def test_ground_truth_is_internally_consistent():
    for case in build_cases():
        if case.expected_scope == "issuer_specific":
            assert case.expected_gateway and case.expected_issuer, case.case_id
        if case.expected_scope in {"fleet_wide", "no_incident"}:
            assert case.expected_gateway is None and case.expected_issuer is None, case.case_id


def test_planted_incidents_are_actually_visible_in_telemetry():
    """Guards against an eval case that no agent could possibly solve.

    For each issuer-scoped case the affected issuer must be measurably worse
    than the others on the same gateway -- if segmentation cannot reveal it,
    the case is unanswerable and the score it produces is meaningless.
    """
    for case in build_cases():
        if case.expected_scope != "issuer_specific":
            continue
        store = telemetry_for(case)
        segments = store.segment_failures(case.expected_gateway, *case.window)["segments"]
        rates = {s["issuer"]: s["success_rate"] for s in segments}
        assert all(v is not None for v in rates.values()), f"{case.case_id}: unmeasurable segment"
        affected = rates[case.expected_issuer]
        others = [v for k, v in rates.items() if k != case.expected_issuer]
        assert affected < 0.5 * min(others), f"{case.case_id}: signal too weak to detect"


def test_no_incident_case_really_has_no_degradation():
    case = next(c for c in build_cases() if c.case_id == "no_incident")
    fleet = telemetry_for(case).fleet_health(*case.window)
    for entry in fleet["per_gateway"]:
        assert entry["success_rate"] > 0.85, entry


# -- graders --------------------------------------------------------------


def test_exact_match_requires_all_three_fields():
    case = next(c for c in build_cases() if c.case_id == "issuer_outage")
    names = telemetry_for(case).names

    right = grade(case, _result(_diagnosis("issuer_specific", "PG-Delta", "HDFC")), names)
    assert right.exact_match

    wrong_issuer = grade(case, _result(_diagnosis("issuer_specific", "PG-Delta", "SBI")), names)
    assert not wrong_issuer.exact_match
    assert wrong_issuer.scope_correct and wrong_issuer.gateway_correct
    assert wrong_issuer.partial_credit == pytest.approx(0.8)


def test_hallucinated_gateway_is_flagged():
    case = build_cases()[0]
    names = telemetry_for(case).names
    score = grade(case, _result(_diagnosis(gateway="PG-Imaginary")), names)
    assert not score.no_hallucination


def test_brier_punishes_confident_wrongness_hardest():
    case = next(c for c in build_cases() if c.case_id == "hard_outage")
    names = telemetry_for(case).names
    confident_wrong = grade(case, _result(_diagnosis(gateway="PG-Alpha", confidence=0.95)), names)
    hedged_wrong = grade(case, _result(_diagnosis(gateway="PG-Alpha", confidence=0.3)), names)
    confident_right = grade(case, _result(_diagnosis(gateway="PG-Bravo", confidence=0.95)), names)
    assert confident_wrong.brier > hedged_wrong.brier > confident_right.brier


def test_failed_run_scores_zero_without_crashing():
    case = build_cases()[0]
    result = InvestigationResult(None, [], Usage(), "budget_steps", "ran out")
    score = grade(case, result, telemetry_for(case).names)
    assert not score.completed and not score.exact_match
    assert score.partial_credit == 0.0


def test_segmentation_use_is_tracked():
    case = build_cases()[0]
    names = telemetry_for(case).names
    assert grade(case, _result(_diagnosis(), ["segment_failures"]), names).used_segmentation
    assert not grade(case, _result(_diagnosis(), ["get_fleet_health"]), names).used_segmentation


# -- memory ---------------------------------------------------------------


def test_recall_finds_a_relevant_memory():
    store = MemoryStore()
    store.remember_investigation(_diagnosis(gateway="PG-Bravo"), (0, 360), verified=True)
    hits = store.recall_text("what is wrong with PG-Bravo", k=3)
    assert hits and "PG-Bravo" in hits[0]


def test_recall_on_an_empty_store_returns_nothing():
    assert MemoryStore().recall_text("anything") == []


def test_recall_ignores_memories_with_no_term_overlap():
    store = MemoryStore()
    store.remember_investigation(_diagnosis(gateway="PG-Bravo"), (0, 360))
    assert store.recall_text("zebra giraffe unrelated") == []


def test_consolidation_promotes_a_repeated_pattern():
    store = MemoryStore()
    for day in range(3):
        store.remember_investigation(_diagnosis(gateway="PG-Bravo"), (day * 1440, day * 1440 + 360))
    created = store.consolidate(min_support=2)
    assert len(created) == 1
    assert created[0].kind == "semantic"
    assert created[0].support == 3
    assert "RECURRING" in created[0].text


def test_consolidation_ignores_one_off_incidents():
    store = MemoryStore()
    store.remember_investigation(_diagnosis(gateway="PG-Bravo"), (0, 360))
    store.remember_investigation(_diagnosis(gateway="PG-Delta"), (1440, 1800))
    assert store.consolidate(min_support=2) == []


def test_consolidation_is_idempotent():
    """Rebuilt from scratch each call, so repeated runs must not stack up."""
    store = MemoryStore()
    for day in range(3):
        store.remember_investigation(_diagnosis(gateway="PG-Bravo"), (day * 1440, day * 1440 + 60))
    store.consolidate()
    first = store.stats()
    store.consolidate()
    assert store.stats() == first


def test_semantic_memories_outrank_the_episodes_behind_them():
    store = MemoryStore()
    for day in range(3):
        store.remember_investigation(_diagnosis(gateway="PG-Bravo"), (day * 1440, day * 1440 + 60))
    store.consolidate()
    top = store.recall("PG-Bravo single_gateway", k=1)
    assert top[0].kind == "semantic"


def test_memory_survives_a_save_load_round_trip(tmp_path):
    path = str(tmp_path / "mem.json")
    store = MemoryStore(path)
    store.remember_investigation(_diagnosis(gateway="PG-Bravo"), (0, 360), verified=True)
    store.save()

    reloaded = MemoryStore(path)
    assert reloaded.stats()["total"] == 1
    assert "PG-Bravo" in reloaded.recall_text("PG-Bravo")[0]


def test_unverified_and_incorrect_diagnoses_are_both_retained():
    """An agent that only remembers its wins cannot learn from its mistakes."""
    store = MemoryStore()
    store.remember_investigation(_diagnosis(), (0, 360), verified=False)
    store.remember_investigation(_diagnosis(), (1440, 1800), verified=None)
    texts = " ".join(m.text for m in store.memories)
    assert "later found incorrect" in texts and "unverified" in texts


# -- harness --------------------------------------------------------------


def test_baseline_policy_scores_below_ceiling_and_above_floor():
    """Locks in eval headroom.

    The first six cases were all solvable by "find the lowest success rate",
    and the baseline scored a perfect 100% on them -- a saturated eval that
    could not distinguish any two arms. Two cases were added that invert a
    threshold rule. If this ever returns to 1.0, the eval has lost its
    discriminating power again and needs harder cases, not celebration.
    """
    metrics, scores = run_arm(
        BaselinePolicyClient(),
        ArmConfig("baseline", "v2-procedural", use_memory=False, passes=1),
        verbose=False,
    )
    exact = metrics[0].exact_match
    assert 0.5 <= exact < 1.0, f"baseline at {exact:.0%}: eval has lost headroom"
    assert metrics[0].completion_rate == 1.0
    assert metrics[0].hallucination_rate == 0.0


def test_baseline_fails_exactly_the_discriminating_cases():
    _, scores = run_arm(
        BaselinePolicyClient(),
        ArmConfig("baseline", "v2-procedural", use_memory=False, passes=1),
        verbose=False,
    )
    failed = {s.case_id for s in scores if not s.exact_match}
    assert failed == {"diurnal_trough", "fleet_wide_partial"}


def test_gate_passes_when_no_baseline_file_exists(tmp_path):
    metrics = [aggregate("arm", 0, [])]
    ok, message = check_gate(metrics, str(tmp_path / "missing.json"))
    assert ok and "no baseline" in message


def test_gate_catches_a_regression(tmp_path):
    import json

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"metrics": [{"arm": "a", "pass_index": 0, "exact_match": 0.9}]}))

    class Fake:
        arm, pass_index, exact_match = "a", 0, 0.6

    ok, message = check_gate([Fake()], str(path))
    assert not ok and "REGRESSION" in message
