"""Behavioural tests: each router must actually do the thing it claims."""

import numpy as np
import pytest

from src.gateways import Outcome
from src.routers import (
    DiscountedCounts,
    EpsilonGreedyRouter,
    PIDThompsonRouter,
    StaticWeightedRouter,
    ThompsonRouter,
    UCB1Router,
)

ADAPTIVE = [EpsilonGreedyRouter, UCB1Router, ThompsonRouter, PIDThompsonRouter]


def _feed(router, gateway_srs, n=4000, seed=0):
    """Route n transactions through a stationary world; return chosen arms."""
    rng = np.random.default_rng(seed)
    chosen = []
    for t in range(n):
        g = router.select(t)
        ok = bool(rng.random() < gateway_srs[g])
        router.update(Outcome(gateway=g, success=ok, latency_ms=100.0, tick=t))
        chosen.append(g)
    return np.array(chosen)


@pytest.mark.parametrize("cls", ADAPTIVE)
def test_router_concentrates_on_the_best_arm(cls):
    chosen = _feed(cls(3, seed=0), [0.50, 0.95, 0.55])
    share_best = (chosen[-1000:] == 1).mean()
    assert share_best > 0.75, f"{cls.__name__} gave the best arm only {share_best:.2%}"


@pytest.mark.parametrize("cls", ADAPTIVE)
def test_router_always_returns_a_valid_gateway_index(cls):
    chosen = _feed(cls(4, seed=1), [0.6, 0.7, 0.8, 0.9], n=500)
    assert set(np.unique(chosen)).issubset({0, 1, 2, 3})


def test_static_router_ignores_feedback_entirely():
    router = StaticWeightedRouter(3, weights=[1, 1, 1], seed=0)
    before = router.estimated_sr().copy()
    _feed(router, [0.01, 0.99, 0.01], n=2000)
    np.testing.assert_allclose(router.estimated_sr(), before)


def test_static_router_respects_its_weights():
    chosen = _feed(
        StaticWeightedRouter(2, weights=[0.25, 0.75], seed=0), [0.9, 0.9], n=8000
    )
    assert (chosen == 1).mean() == pytest.approx(0.75, abs=0.02)


def test_static_router_rejects_degenerate_weights():
    with pytest.raises(ValueError):
        StaticWeightedRouter(2, weights=[0.0, 0.0])
    with pytest.raises(ValueError):
        StaticWeightedRouter(3, weights=[1.0, 1.0])


def test_discounting_lets_a_router_follow_a_regime_change():
    """The whole point of gamma: arm 0 is best, then abruptly is not."""
    router = ThompsonRouter(2, gamma=0.99, seed=0)
    _feed(router, [0.95, 0.40], n=3000, seed=0)
    after = _feed(router, [0.05, 0.95], n=3000, seed=1)
    assert (after[-500:] == 1).mean() > 0.8


def test_undiscounted_router_gets_stuck_after_a_regime_change():
    """Contrast case that motivates the design -- keep this gap documented."""
    router = ThompsonRouter(2, gamma=1.0, seed=0)
    _feed(router, [0.95, 0.40], n=20000, seed=0)
    after = _feed(router, [0.05, 0.95], n=2000, seed=1)
    assert (after[-500:] == 1).mean() < 0.8


def test_discounted_counts_forget_at_the_configured_rate():
    counts = DiscountedCounts(1, gamma=0.9, prior=1.0)
    for _ in range(500):
        counts.update(0, True)
    # Geometric ceiling: prior + sum(gamma^k) = 1 + 1/(1 - gamma) = 11
    assert counts.successes[0] == pytest.approx(11.0, abs=0.1)


def test_undiscounted_counts_grow_without_bound():
    counts = DiscountedCounts(1, gamma=1.0, prior=1.0)
    for _ in range(500):
        counts.update(0, True)
    assert counts.successes[0] == pytest.approx(501.0)


def test_discounted_counts_reject_invalid_gamma():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            DiscountedCounts(2, gamma=bad)


def test_pid_controller_idles_when_the_model_is_calibrated():
    """A calibrated posterior must cost nothing: inflation stays near 1.0."""
    router = PIDThompsonRouter(2, seed=0)
    _feed(router, [0.9, 0.9], n=6000)
    assert router.inflation == pytest.approx(1.0, abs=0.35)


def test_pid_controller_reacts_to_a_gateway_that_dies():
    """Prediction stays high while outcomes crash -> residual -> exploration."""
    router = PIDThompsonRouter(2, seed=0)
    _feed(router, [0.95, 0.30], n=6000)
    calm = router.inflation
    _feed(router, [0.02, 0.30], n=300, seed=3)
    assert router.inflation > calm + 1.0


def test_pid_inflation_is_bounded():
    router = PIDThompsonRouter(2, max_inflation=25.0, seed=0)
    _feed(router, [0.99, 0.99], n=2000)
    _feed(router, [0.01, 0.01], n=2000, seed=5)
    assert 1.0 <= router.inflation <= 25.0
