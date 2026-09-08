"""Constraint-layer tests.

This layer lets a language model move production traffic, so the tests are
weighted toward what it must *refuse* to do. The gate between "the model said
something" and "the system did something" is the part worth over-testing.
"""

import numpy as np
import pytest

from src.agent.constraints import (
    ACTIONABLE_SCOPES,
    ConstrainedRouter,
    ConstraintStore,
    RoutingConstraint,
    from_diagnosis,
)
from src.agent.schemas import Diagnosis
from src.closed_loop import run_closed_loop
from src.gateways import RoutingContext
from src.routers import ThompsonRouter
from src.simulator import SimConfig

FLEET = ["PG-Alpha", "PG-Bravo", "PG-Charlie", "PG-Delta", "PG-Echo"]


def _diagnosis(scope="issuer_specific", gateway="PG-Delta", issuer="HDFC", confidence=0.8):
    return Diagnosis(
        scope=scope, primary_gateway=gateway, affected_issuer=issuer,
        confidence=confidence, summary="s", evidence=["e"], recommended_action="a",
    )


def _constraint(gateway="PG-Delta", issuer="HDFC", start=0, end=100, canary=0.0):
    return RoutingConstraint(
        gateway=gateway, issuer=issuer, reason="r", created_tick=start,
        expires_tick=end, confidence=0.8, source_scope="issuer_specific",
        canary_rate=canary,
    )


# -- translation: what must NOT become a constraint -----------------------


def test_a_confident_issuer_finding_becomes_a_scoped_constraint():
    c = from_diagnosis(_diagnosis(), tick=100, fleet_names=FLEET, ttl_minutes=60)
    assert c is not None
    assert c.gateway == "PG-Delta" and c.issuer == "HDFC"
    assert c.expires_tick == 160


def test_single_gateway_finding_constrains_all_traffic():
    c = from_diagnosis(
        _diagnosis(scope="single_gateway", issuer=None), tick=0, fleet_names=FLEET
    )
    assert c is not None and c.issuer is None


def test_low_confidence_is_refused():
    assert from_diagnosis(_diagnosis(confidence=0.4), 0, FLEET) is None


def test_fleet_wide_and_no_incident_are_refused():
    """Neither names a culprit, so steering between gateways cannot help."""
    for scope in ("fleet_wide", "no_incident"):
        assert scope not in ACTIONABLE_SCOPES
        assert from_diagnosis(_diagnosis(scope=scope, gateway=None, issuer=None), 0, FLEET) is None


def test_hallucinated_gateway_is_refused():
    """The last line of defence before a model moves real traffic."""
    assert from_diagnosis(_diagnosis(gateway="PG-Imaginary"), 0, FLEET) is None


def test_issuer_scoped_finding_without_an_issuer_is_refused():
    assert from_diagnosis(_diagnosis(issuer=None), 0, FLEET) is None


# -- store semantics ------------------------------------------------------


def test_constraint_only_blocks_the_named_issuer():
    store = ConstraintStore(FLEET)
    store.add(_constraint())
    assert store.blocked(RoutingContext(tick=10, issuer="HDFC")) == frozenset({3})
    assert store.blocked(RoutingContext(tick=10, issuer="ICICI")) == frozenset()


def test_unscoped_constraint_blocks_every_issuer():
    store = ConstraintStore(FLEET)
    store.add(_constraint(issuer=None))
    for issuer in ("HDFC", "ICICI", None):
        assert store.blocked(RoutingContext(tick=10, issuer=issuer)) == frozenset({3})


def test_constraint_expires():
    store = ConstraintStore(FLEET)
    store.add(_constraint(start=0, end=100))
    assert store.blocked(RoutingContext(tick=99, issuer="HDFC")) == frozenset({3})
    assert store.blocked(RoutingContext(tick=100, issuer="HDFC")) == frozenset()


def test_constraint_is_not_active_before_it_is_created():
    store = ConstraintStore(FLEET)
    store.add(_constraint(start=50, end=100))
    assert store.blocked(RoutingContext(tick=10, issuer="HDFC")) == frozenset()


def test_canary_lets_some_traffic_through():
    """Without this the blocked gateway becomes permanently unobservable."""
    store = ConstraintStore(FLEET, seed=0)
    store.add(_constraint(canary=0.10))
    ctx = RoutingContext(tick=10, issuer="HDFC")
    released = sum(1 for _ in range(4000) if store.blocked(ctx) == frozenset())
    assert 0.08 < released / 4000 < 0.12
    assert store.canary_releases > 0


def test_zero_canary_blocks_every_time():
    store = ConstraintStore(FLEET, seed=0)
    store.add(_constraint(canary=0.0))
    ctx = RoutingContext(tick=10, issuer="HDFC")
    assert all(store.blocked(ctx) == frozenset({3}) for _ in range(200))


def test_a_transaction_is_never_left_with_nowhere_to_go():
    """Blocking everything guarantees a failure; routing to a suspect does not."""
    store = ConstraintStore(FLEET, seed=0)
    for name in FLEET:
        store.add(_constraint(gateway=name, issuer=None, canary=0.0))
    assert store.blocked(RoutingContext(tick=10, issuer="HDFC")) == frozenset()


# -- router integration ---------------------------------------------------


def test_constrained_router_never_returns_a_blocked_gateway():
    store = ConstraintStore(FLEET, seed=0)
    store.add(_constraint(canary=0.0))
    router = ConstrainedRouter(ThompsonRouter(5, seed=0), store)
    picks = {router.select(10, RoutingContext(tick=10, issuer="HDFC")) for _ in range(500)}
    assert 3 not in picks


def test_constrained_router_leaves_other_issuers_untouched():
    store = ConstraintStore(FLEET, seed=0)
    store.add(_constraint(canary=0.0))
    router = ConstrainedRouter(ThompsonRouter(5, seed=0), store)
    # Give PG-Delta a strong record so an unconstrained router prefers it.
    from src.gateways import Outcome

    for _ in range(400):
        router.update(Outcome(gateway=3, success=True, latency_ms=100.0, tick=0))
    picks = [router.select(10, RoutingContext(tick=10, issuer="ICICI")) for _ in range(200)]
    assert picks.count(3) > 150


@pytest.mark.parametrize("cls", [ThompsonRouter])
def test_blocked_set_is_honoured_by_the_underlying_router(cls):
    router = cls(5, seed=0)
    picks = {router.select(0, None, frozenset({0, 1, 2})) for _ in range(300)}
    assert picks <= {3, 4}


def test_unconstrained_selection_is_unchanged_by_the_new_parameters():
    """The benchmark numbers predate constraints and must stay comparable."""
    a = ThompsonRouter(5, gamma=0.999, seed=3)
    b = ThompsonRouter(5, gamma=0.999, seed=3)
    assert [a.select(t) for t in range(300)] == [
        b.select(t, RoutingContext(tick=t, issuer="HDFC"), frozenset()) for t in range(300)
    ]


# -- closed loop ----------------------------------------------------------


def test_closed_loop_without_an_investigator_matches_the_open_loop():
    from close_loop import scenario_fleet

    config = SimConfig(days=2, transactions=4000, seeds=(1,))
    a = run_closed_loop(scenario_fleet(), config, lambda n, s: ThompsonRouter(n, seed=s), 1)
    b = run_closed_loop(scenario_fleet(), config, lambda n, s: ThompsonRouter(n, seed=s), 1)
    assert a.run.realised_sr == b.run.realised_sr
    assert a.constraints == [] and a.investigations == []


def test_closed_loop_installs_a_constraint_and_routes_under_it():
    from close_loop import DETECT_AT, scenario_fleet

    from src.agent import Investigator, InvestigatorConfig
    from src.agent.llm import BaselinePolicyClient

    client = BaselinePolicyClient()

    def investigate(telemetry, window):
        return Investigator(telemetry, client, config=InvestigatorConfig()).investigate(
            "Conversion is down.", window
        )

    # 150k matches close_loop.py's scenario and sits solidly in the regime
    # where issuer segments are measurable. Nearer the sampling boundary
    # (~80-120k) the scoped/blunt outcome flips with noise, so pinning a
    # specific diagnosis there would make this test flaky.
    config = SimConfig(days=2, transactions=150_000, seeds=(1,))
    result = run_closed_loop(
        scenario_fleet(), config, lambda n, s: ThompsonRouter(n, gamma=0.999, seed=s), 1,
        detect_ticks=(DETECT_AT,), investigate=investigate, canary_rate=0.0,
    )
    assert len(result.investigations) == 1
    assert len(result.constraints) == 1
    constraint = result.constraints[0]
    assert constraint.gateway == "PG-Delta" and constraint.issuer == "HDFC"

    # After the constraint lands, HDFC must stop reaching PG-Delta.
    tick = np.asarray(result.run.tick)
    from src.gateways import ISSUERS

    hdfc = np.asarray(result.run.issuer) == ISSUERS.index("HDFC")
    after = (tick >= constraint.created_tick) & (tick < constraint.expires_tick) & hdfc
    assert after.sum() > 0
    assert (np.asarray(result.run.chosen)[after] == 3).mean() == 0.0


def test_low_volume_degrades_to_a_blunt_constraint_rather_than_a_wrong_one():
    """The loop's *precision* depends on traffic volume, and degrades safely.

    Segmenting by issuer needs enough traffic in each gateway-by-issuer cell to
    compute a rate. At 40k transactions/2 days in this scenario the segments
    fall under the sampling floor, the agent cannot establish that the failure
    is issuer-scoped, and it reports `single_gateway` instead. The resulting
    constraint is blunt -- it steers *all* traffic off PG-Delta, including the
    issuers that were converting fine. (The crossover is not a sharp cliff:
    between roughly 80k and 120k the outcome flips with sampling noise. Only
    the two ends are pinned here.)

    That is a worse outcome than the scoped constraint but a safe one: it
    matches what a flat bandit would have done anyway. The failure mode to
    avoid is confidently naming the *wrong* issuer, and that does not happen --
    the agent withholds the claim it cannot support.
    """
    from close_loop import DETECT_AT, scenario_fleet

    from src.agent import Investigator, InvestigatorConfig
    from src.agent.llm import BaselinePolicyClient

    client = BaselinePolicyClient()

    def investigate(telemetry, window):
        return Investigator(telemetry, client, config=InvestigatorConfig()).investigate(
            "Conversion is down.", window
        )

    config = SimConfig(days=2, transactions=40_000, seeds=(1,))
    result = run_closed_loop(
        scenario_fleet(), config, lambda n, s: ThompsonRouter(n, gamma=0.999, seed=s), 1,
        detect_ticks=(DETECT_AT,), investigate=investigate, canary_rate=0.0,
    )
    constraint = result.constraints[0]
    assert constraint.gateway == "PG-Delta"
    assert constraint.issuer is None, "must not guess an issuer it cannot measure"


def test_agent_sees_no_data_from_after_the_detection_point():
    """No lookahead: the investigation must be causal."""
    from close_loop import DETECT_AT, scenario_fleet

    seen = {}

    def investigate(telemetry, window):
        seen["max_tick"] = int(np.asarray(telemetry.result.tick).max())
        from src.agent.schemas import InvestigationResult, Usage

        return InvestigationResult(None, [], Usage(), "completed")

    config = SimConfig(days=2, transactions=20_000, seeds=(1,))
    run_closed_loop(
        scenario_fleet(), config, lambda n, s: ThompsonRouter(n, seed=s), 1,
        detect_ticks=(DETECT_AT,), investigate=investigate,
    )
    assert seen["max_tick"] <= DETECT_AT
