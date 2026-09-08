"""The simulator is the measuring instrument -- if it is wrong, every number
downstream is wrong. These tests pin its contract."""

import pytest

from src.gateways import (
    TICKS_PER_DAY,
    DegradationEvent,
    GatewayEnvironment,
    GatewaySpec,
    default_fleet,
)


def test_sr_stays_in_valid_probability_range():
    spec = GatewaySpec("x", base_sr=0.99, base_latency_ms=100, diurnal_amplitude=0.9)
    for t in range(0, TICKS_PER_DAY, 7):
        assert 0.0 < spec.true_sr(t) < 1.0


def test_degradation_event_applies_only_inside_window():
    spec = GatewaySpec(
        "x",
        base_sr=0.9,
        base_latency_ms=100,
        events=[DegradationEvent(start=100, end=200, sr_multiplier=0.1)],
    )
    assert spec.true_sr(99) == pytest.approx(0.9)
    assert spec.true_sr(100) == pytest.approx(0.09)
    assert spec.true_sr(199) == pytest.approx(0.09)
    assert spec.true_sr(200) == pytest.approx(0.9)


def test_diurnal_cycle_is_periodic():
    spec = GatewaySpec("x", base_sr=0.9, base_latency_ms=100, diurnal_amplitude=0.1)
    assert spec.true_sr(0) == pytest.approx(spec.true_sr(TICKS_PER_DAY), abs=1e-9)


def test_oracle_dominates_every_individual_gateway():
    env = GatewayEnvironment(default_fleet(), seed=0)
    for t in range(0, 7 * TICKS_PER_DAY, 331):
        best = env.oracle_sr(t)
        assert all(spec.true_sr(t) <= best + 1e-12 for spec in env.specs)
        assert env.specs[env.best_gateway(t)].true_sr(t) == pytest.approx(best)


def test_empirical_success_rate_matches_declared_probability():
    """The Bernoulli draw must honour true_sr -- otherwise regret is a fiction."""
    spec = GatewaySpec("x", base_sr=0.75, base_latency_ms=100)
    env = GatewayEnvironment([spec], seed=42)
    trials = 40_000
    hits = sum(env.attempt(0, 0).success for _ in range(trials))
    assert hits / trials == pytest.approx(0.75, abs=0.01)


def test_environment_is_reproducible_under_a_fixed_seed():
    a = GatewayEnvironment(default_fleet(), seed=7)
    b = GatewayEnvironment(default_fleet(), seed=7)
    assert [a.attempt(i % 5, i).success for i in range(500)] == [
        b.attempt(i % 5, i).success for i in range(500)
    ]


def test_fleet_leadership_actually_changes_over_the_week():
    """If one gateway were always best, the benchmark would prove nothing."""
    env = GatewayEnvironment(default_fleet(), seed=0)
    leaders = {env.best_gateway(t) for t in range(0, 7 * TICKS_PER_DAY, 60)}
    assert len(leaders) >= 3


def test_rejects_empty_fleet():
    with pytest.raises(ValueError):
        GatewayEnvironment([], seed=0)
