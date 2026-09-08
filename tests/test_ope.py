"""Off-policy evaluation tests.

Built on a small synthetic bandit where the true policy value is computable in
closed form, so unbiasedness is checked against an actual number rather than
asserted. The estimators are the part of this project most capable of being
confidently wrong -- IPS will happily return a success rate above 100% -- so
they get tested against ground truth rather than against each other.
"""

import numpy as np
import pytest

from src.gateways import Outcome, RoutingContext
from src.ope import (
    LoggedData,
    direct_method,
    doubly_robust,
    evaluate_all,
    fit_reward_model,
    ips,
    snips,
    true_value,
)
from src.routers import EpsilonGreedyRouter, StaticWeightedRouter, ThompsonRouter, UCB1Router

# A 2-context, 3-action world with known reward probabilities.
# Action 2 is best in context 0; action 0 is best in context 1.
TRUE_SR = np.array([
    [[0.50, 0.20],   # action 0: bad in ctx 0, good in ctx 1
     [0.60, 0.40],   # action 1
     [0.90, 0.30]],  # action 2: best in ctx 0
])[0]  # (n_actions, n_contexts)


def make_logs(logging_probs, n=40_000, seed=0):
    """Sample bandit feedback from a fixed stochastic logging policy."""
    rng = np.random.default_rng(seed)
    context = rng.integers(0, 2, size=n)
    actions = np.array([
        rng.choice(3, p=logging_probs[c]) for c in context
    ])
    propensity = logging_probs[context, actions]
    reward = (rng.random(n) < TRUE_SR[actions, context]).astype(float)
    return LoggedData(
        context=context, action=actions, reward=reward, propensity=propensity,
        tick=np.zeros(n, dtype=int), n_actions=3, n_contexts=2,
    )


def analytic_value(target_probs, logged):
    """Exact value of a fixed policy under the known reward probabilities."""
    per_row = TRUE_SR[:, logged.context].T  # (n, n_actions)
    return float(np.sum(target_probs[logged.context] * per_row, axis=1).mean())


EXPLORING = np.array([[0.4, 0.3, 0.3], [0.3, 0.35, 0.35]])


# -- validity -------------------------------------------------------------


def test_ips_is_unbiased():
    """The property that justifies IPS existing, checked against real truth."""
    logged = make_logs(EXPLORING, n=60_000, seed=1)
    target = np.array([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]])
    truth = analytic_value(target, logged)
    estimate = ips(logged, target[logged.context])
    assert abs(estimate.value - truth) < 0.02, f"{estimate.value:.4f} vs {truth:.4f}"


def test_evaluating_the_logging_policy_recovers_what_it_earned():
    """Sanity anchor: with target == logging policy every weight is 1."""
    logged = make_logs(EXPLORING, n=20_000, seed=2)
    estimate = ips(logged, EXPLORING[logged.context])
    np.testing.assert_allclose(estimate.value, logged.observed_value, rtol=1e-12)
    np.testing.assert_allclose(estimate.ess, logged.n, rtol=1e-9)


def test_all_estimators_agree_when_the_policies_are_close():
    logged = make_logs(EXPLORING, n=40_000, seed=3)
    target = EXPLORING * 0.9 + np.full((2, 3), 1 / 3) * 0.1
    target = target / target.sum(axis=1, keepdims=True)
    truth = analytic_value(target, logged)
    estimates, _ = evaluate_all(logged, target[logged.context], sr=None)
    for estimate in estimates:
        assert abs(estimate.value - truth) < 0.03, f"{estimate.estimator} was {estimate.value:.4f}"


def test_true_value_matches_the_analytic_calculation():
    logged = make_logs(EXPLORING, n=5_000, seed=4)
    target = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    sr = TRUE_SR[:, None, :]  # (n_actions, 1 tick, n_contexts)
    np.testing.assert_allclose(
        true_value(logged, target[logged.context], sr),
        analytic_value(target, logged),
        rtol=1e-12,
    )


# -- the failure modes that matter ----------------------------------------


def test_ips_is_not_bounded_by_the_reward_range():
    """Documents why nobody ships raw IPS.

    Rewards are 0/1, so a success rate above 1.0 is not merely wrong, it is
    incoherent -- yet nothing in the IPS formula prevents it. Constructed
    deterministically rather than fished for with a seed: every logged action
    had propensity 0.01 and the target would have taken it with probability 1,
    so every weight is 100 and the estimate is 100x the mean reward.
    """
    n = 100
    logged = LoggedData(
        context=np.zeros(n, dtype=int), action=np.zeros(n, dtype=int),
        reward=np.ones(n), propensity=np.full(n, 0.01),
        tick=np.zeros(n, dtype=int), n_actions=2, n_contexts=1,
    )
    target = np.tile(np.array([1.0, 0.0]), (n, 1))
    assert ips(logged, target).value == pytest.approx(100.0)
    # SNIPS divides by the summed weights instead of n, so it cannot escape
    # the range of rewards it actually observed.
    assert snips(logged, target).value == pytest.approx(1.0)


def test_snips_has_lower_variance_than_ips():
    """The practical version of the same point, on sampled data.

    Only the direction is asserted. The size of the gap depends on how much
    overlap there is between the policies -- measured at 0.035 vs 0.022 here,
    a factor of 1.6 rather than the order of magnitude an earlier version of
    this test assumed.
    """
    skewed = np.array([[0.95, 0.025, 0.025], [0.95, 0.025, 0.025]])
    logged = make_logs(skewed, n=20_000, seed=5)
    target = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])[logged.context]
    assert snips(logged, target).std_err < ips(logged, target).std_err


def test_snips_never_leaves_the_reward_range():
    """Self-normalisation makes an impossible estimate structurally impossible."""
    skewed = np.array([[0.98, 0.01, 0.01], [0.98, 0.01, 0.01]])
    logged = make_logs(skewed, n=20_000, seed=6)
    target = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    estimate = snips(logged, target[logged.context])
    assert 0.0 <= estimate.value <= 1.0


def test_doubly_robust_beats_ips_when_overlap_is_poor():
    """The reason DR is the default in practice."""
    skewed = np.array([[0.90, 0.05, 0.05], [0.90, 0.05, 0.05]])
    logged = make_logs(skewed, n=40_000, seed=7)
    target = np.array([[0.05, 0.05, 0.90], [0.05, 0.05, 0.90]])
    truth = analytic_value(target, logged)
    expanded = target[logged.context]
    assert abs(doubly_robust(logged, expanded).value - truth) < abs(
        ips(logged, expanded).value - truth
    )


def test_effective_sample_size_collapses_as_policies_diverge():
    """ESS is the diagnostic that tells you an estimate is decorative."""
    logged = make_logs(EXPLORING, n=20_000, seed=8)
    near = ips(logged, EXPLORING[logged.context])
    far = ips(logged, np.array([[0, 0, 1.0], [0, 0, 1.0]])[logged.context])
    assert near.ess > logged.n * 0.9
    assert far.ess < logged.n * 0.5


def test_off_support_actions_are_reported():
    """A target wanting actions the logs never contain is pure extrapolation."""
    n = 2_000
    rng = np.random.default_rng(9)
    logged = LoggedData(
        context=rng.integers(0, 3, n), action=rng.integers(0, 2, n),  # never action 2
        reward=(rng.random(n) < 0.9).astype(float), propensity=np.full(n, 0.5),
        tick=np.zeros(n, dtype=int), n_actions=3, n_contexts=3,
    )
    target = np.zeros((n, 3))
    target[:, 2] = 1.0
    estimate = ips(logged, target)
    assert estimate.off_support_fraction == pytest.approx(1.0)
    assert estimate.value == 0.0  # it knows nothing, and says so


def test_clipping_trades_bias_for_variance():
    skewed = np.array([[0.95, 0.025, 0.025], [0.95, 0.025, 0.025]])
    logged = make_logs(skewed, n=20_000, seed=10)
    target = np.array([[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]])[logged.context]
    unclipped, clipped = ips(logged, target), ips(logged, target, clip=10.0)
    assert clipped.std_err < unclipped.std_err
    assert clipped.clipped_fraction > 0
    assert clipped.ess > unclipped.ess


# -- input validation -----------------------------------------------------


def test_zero_propensity_is_rejected():
    """An action that was taken cannot have had probability zero."""
    with pytest.raises(ValueError, match="propensity"):
        LoggedData(
            context=np.zeros(3, dtype=int), action=np.zeros(3, dtype=int),
            reward=np.ones(3), propensity=np.array([0.5, 0.0, 0.5]),
            tick=np.zeros(3, dtype=int), n_actions=2, n_contexts=1,
        )


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="mismatched"):
        LoggedData(
            context=np.zeros(3, dtype=int), action=np.zeros(2, dtype=int),
            reward=np.ones(3), propensity=np.full(3, 0.5),
            tick=np.zeros(3, dtype=int), n_actions=2, n_contexts=1,
        )


def test_reward_model_falls_back_for_unseen_cells():
    n = 500
    rng = np.random.default_rng(11)
    logged = LoggedData(
        context=np.zeros(n, dtype=int), action=rng.integers(0, 2, n),
        reward=np.ones(n), propensity=np.full(n, 0.5),
        tick=np.zeros(n, dtype=int), n_actions=3, n_contexts=2,
    )
    model = fit_reward_model(logged)
    assert model.shape == (2, 3)
    assert np.isfinite(model).all()  # unseen context 1 and action 2 still filled


# -- propensities from the routers ---------------------------------------


def test_stochastic_routers_expose_usable_propensities():
    rng = np.random.default_rng(0)
    context = RoutingContext(tick=0, issuer="HDFC")
    for router in (
        ThompsonRouter(5, seed=0),
        EpsilonGreedyRouter(5, epsilon=0.1, seed=0),
        StaticWeightedRouter(5, weights=[1, 1, 2, 3, 3], seed=0),
    ):
        probs = router.action_probabilities(0, context, rng=rng)
        assert probs.shape == (5,)
        assert probs.sum() == pytest.approx(1.0)
        assert (probs >= 0).all()


def test_ucb1_is_deterministic_and_therefore_unusable_for_logging():
    """The constraint people discover too late.

    A deterministic logging policy assigns probability zero to every action it
    did not take, so importance weights are undefined and no volume of logged
    data can support off-policy evaluation. If you want to evaluate policies
    offline later, what you deploy today has to be stochastic.
    """
    router = UCB1Router(5, seed=0)
    rng = np.random.default_rng(0)

    # A cold router has every arm tied, and ties are broken uniformly -- so it
    # is briefly stochastic. That is not a reprieve: it lasts only until the
    # counts diverge.
    assert router.action_probabilities(0, rng=rng).max() == pytest.approx(0.2)

    for t in range(400):
        gateway = router.select(t)
        router.update(Outcome(gateway=gateway, success=gateway == 2,
                              latency_ms=1.0, tick=t))

    probs = router.action_probabilities(0, rng=rng)
    assert probs.max() == 1.0
    assert (probs == 0).sum() == 4


def test_propensity_computation_does_not_disturb_routing():
    """Measuring must not change what is being measured.

    The propensity Monte Carlo draws from a caller-supplied generator. If it
    used the router's own stream, every logged run would take a different path
    than the same router would take unlogged -- and every benchmark number in
    this repo would silently shift.
    """
    plain = ThompsonRouter(5, gamma=0.999, seed=3)
    probed = ThompsonRouter(5, gamma=0.999, seed=3)
    rng = np.random.default_rng(0)
    for t in range(300):
        probed.action_probabilities(t, rng=rng)
        assert plain.select(t) == probed.select(t)
