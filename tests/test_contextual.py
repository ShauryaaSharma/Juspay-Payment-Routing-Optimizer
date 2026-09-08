"""Contextual bandit tests.

The interesting property is not "does Thompson sampling work" -- that is tested
in `test_routers.py`. It is whether conditioning on the issuer actually buys
resolution a flat bandit cannot have, and whether partial pooling behaves the
way the docstring claims at both ends of its range.
"""

import numpy as np
import pytest

from src.gateways import ISSUERS, Outcome, RoutingContext
from src.routers import ContextualThompsonRouter, ThompsonRouter

# Gateway 3 is the best for everyone except HDFC, whom it declines outright.
# This is the failure a flat bandit structurally cannot represent.
BASE_SR = [0.90, 0.90, 0.90, 0.95, 0.90]
BROKEN_GATEWAY = 3
BROKEN_ISSUER = "HDFC"


def _drive(router, n=20_000, seed=0, contextual=True):
    rng = np.random.default_rng(seed)
    for t in range(n):
        issuer = ISSUERS[rng.integers(len(ISSUERS))]
        context = RoutingContext(tick=t, issuer=issuer) if contextual else None
        gateway = router.select(t, context)
        sr = BASE_SR[gateway]
        if gateway == BROKEN_GATEWAY and issuer == BROKEN_ISSUER:
            sr = 0.05
        outcome = Outcome(
            gateway=gateway, success=bool(rng.random() < sr), latency_ms=1.0,
            tick=t, issuer=issuer if contextual else None,
        )
        router.update(outcome)
    return router


def _share(router, issuer, gateway, n=1000):
    picks = [router.select(0, RoutingContext(tick=0, issuer=issuer)) for _ in range(n)]
    return picks.count(gateway) / n


def test_learns_a_failure_confined_to_one_issuer():
    router = _drive(ContextualThompsonRouter(5, seed=0))
    assert _share(router, BROKEN_ISSUER, BROKEN_GATEWAY) < 0.05
    assert _share(router, "ICICI", BROKEN_GATEWAY) > 0.40


def test_the_flat_bandit_cannot_represent_the_same_thing():
    """The contrast that motivates the whole contextual arm.

    A flat router holds one number per gateway, so an issuer-specific failure
    is averaged into a mildly-degraded gateway. It cannot route HDFC away while
    keeping everyone else -- there is nowhere in its state to put that.
    """
    flat = ThompsonRouter(5, gamma=0.999, seed=0)
    _drive(flat, contextual=False)
    estimates = flat.estimated_sr()
    # The broken gateway looks merely soft, not broken: HDFC is one issuer in
    # five, so its collapse is diluted rather than visible.
    assert 0.70 < estimates[BROKEN_GATEWAY] < 0.92


def test_higher_pooling_narrows_the_gap_between_issuers():
    """Shrinkage works, but it is bounded -- and the bound is deliberate.

    An earlier version of this test asserted that a very large `k` collapses
    the router to a flat bandit. It does not, because prior strength is capped
    by available pooled evidence while each cell's own evidence keeps growing.
    That cap is what keeps unplayed gateways explorable, so the honest claim is
    the weaker one: more pooling means issuers look more alike, not identical.
    """
    weak = _drive(ContextualThompsonRouter(5, pooling=0.0, seed=0))
    strong = _drive(ContextualThompsonRouter(5, pooling=100_000.0, seed=0))

    def gap(router):
        return abs(
            router.estimated_sr_for("ICICI")[BROKEN_GATEWAY]
            - router.estimated_sr_for(BROKEN_ISSUER)[BROKEN_GATEWAY]
        )

    assert gap(strong) < gap(weak) / 2
    assert gap(strong) > 0.0  # bounded shrinkage, not collapse


def test_zero_pooling_is_pure_per_cell():
    router = ContextualThompsonRouter(5, pooling=0.0, seed=0)
    _drive(router)
    # With no shrinkage the two issuers' posteriors must diverge sharply.
    hdfc = router.estimated_sr_for(BROKEN_ISSUER)[BROKEN_GATEWAY]
    icici = router.estimated_sr_for("ICICI")[BROKEN_GATEWAY]
    assert icici - hdfc > 0.3


def test_unplayed_gateways_stay_explorable():
    """Regression test for a real bug found in development.

    The first implementation applied the full pooling strength to a gateway
    with no pooled evidence, producing a *confident* prior at the uninformative
    0.5 mean -- Beta(20, 20) at k=40. That can never out-sample an arm holding
    real evidence, so the router locked onto whichever gateway it tried first
    and never explored again. Capping prior strength by pooled evidence makes
    an unplayed gateway's prior uniform.
    """
    router = ContextualThompsonRouter(5, pooling=40.0, seed=0)
    # A cold router must not concentrate: every gateway should get traffic.
    picks = [router.select(t, RoutingContext(tick=t, issuer="HDFC")) for t in range(600)]
    assert len(set(picks)) == 5, f"cold start explored only {sorted(set(picks))}"

    router = _drive(router, n=8_000)
    observed = router.cell_observations("HDFC")
    assert (observed > 0).sum() >= 4, "most gateways should have been tried for HDFC"


def test_missing_context_falls_back_and_is_counted():
    """Silent degradation to a flat bandit must be visible, not invisible."""
    router = ContextualThompsonRouter(5, seed=0)
    for t in range(500):
        gateway = router.select(t)  # no context at all
        router.update(Outcome(gateway=gateway, success=True, latency_ms=1.0, tick=t))
    assert router.updates_without_context == 500
    assert all(c.total.sum() == 0 for c in router.cell_counts)
    assert router.global_counts.total.sum() > 0


def test_unknown_issuer_does_not_crash_or_mis_credit():
    router = ContextualThompsonRouter(5, seed=0)
    context = RoutingContext(tick=0, issuer="NOT-A-BANK")
    gateway = router.select(0, context)
    assert 0 <= gateway < 5
    router.update(Outcome(gateway=gateway, success=True, latency_ms=1.0,
                          tick=0, issuer="NOT-A-BANK"))
    assert router.updates_without_context == 1


def test_issuer_matching_is_case_insensitive():
    router = _drive(ContextualThompsonRouter(5, seed=0))
    upper = router.estimated_sr_for("hdfc")
    exact = router.estimated_sr_for("HDFC")
    np.testing.assert_allclose(upper, exact)


def test_honours_the_blocked_set():
    router = ContextualThompsonRouter(5, seed=0)
    context = RoutingContext(tick=0, issuer="HDFC")
    picks = {router.select(0, context, frozenset({0, 1, 2})) for _ in range(300)}
    assert picks <= {3, 4}


def test_negative_pooling_is_rejected():
    with pytest.raises(ValueError):
        ContextualThompsonRouter(5, pooling=-1.0)


def test_context_costs_statistical_power():
    """The tradeoff, made explicit.

    Conditioning on 5 issuers splits the same traffic across 5x the cells, so
    each holds far less evidence than the pooled per-gateway posterior. This is
    why a contextual bandit is not free, and why the agent can win early.
    """
    # gamma=1.0 on purpose: with discounting both counters saturate at
    # 1/(1-gamma) and the split stops being visible in the totals, which is
    # what made an earlier version of this test fail for the wrong reason.
    router = _drive(ContextualThompsonRouter(5, gamma=1.0, seed=0), n=10_000)
    pooled_total = router.global_counts.total.sum()
    cell_total = router.cell_observations(BROKEN_ISSUER).sum()
    # HDFC is one issuer in five, so its cells should hold roughly a fifth.
    assert cell_total < pooled_total / 3
