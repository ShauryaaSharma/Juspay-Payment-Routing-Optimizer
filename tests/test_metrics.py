"""Metrics tests -- regret and outage-response are the claims that get quoted."""

import numpy as np
import pytest

from src.gateways import DegradationEvent
from src.metrics import RunResult, _rolling_mean, outage_response_ticks


def _result(chosen, n_gateways=2):
    chosen = np.asarray(chosen)
    n = chosen.size
    return RunResult(
        router="t",
        chosen=chosen,
        success=np.ones(n, dtype=bool),
        latency_ms=np.full(n, 100.0),
        instant_regret=np.zeros(n),
        cost_bps=np.zeros(n),
        n_gateways=n_gateways,
    )


def test_rolling_mean_is_trailing_and_length_preserving():
    x = np.array([1.0, 0.0, 1.0, 1.0])
    out = _rolling_mean(x, 2)
    assert out.size == x.size
    np.testing.assert_allclose(out, [1.0, 0.5, 0.5, 1.0])


def test_allocation_rows_sum_to_one_per_transaction():
    alloc = _result([0, 1, 1, 0, 1]).allocation()
    np.testing.assert_allclose(alloc.sum(axis=0), np.ones(5))


def test_outage_response_detects_a_router_that_pulls_away():
    chosen = np.array([1] * 100 + [0] * 900)
    event = DegradationEvent(start=0, end=1000, sr_multiplier=0.1)
    resp = outage_response_ticks(_result(chosen), 1, event, tx_per_tick=1.0, window_tx=20)
    assert np.isfinite(resp)
    assert 100 <= resp <= 140


def test_outage_response_is_infinite_for_a_router_that_never_reacts():
    chosen = np.array([1] * 1000)
    event = DegradationEvent(start=0, end=1000, sr_multiplier=0.1)
    resp = outage_response_ticks(_result(chosen), 1, event, tx_per_tick=1.0, window_tx=20)
    assert np.isinf(resp)


def test_outage_response_converts_tick_space_to_transaction_space():
    """Events are in ticks, results are in transactions. Conflating the two
    silently reported 'never' for every router and hid a broken metric."""
    chosen = np.array([1] * 400 + [0] * 600)
    event = DegradationEvent(start=100, end=400, sr_multiplier=0.1)  # ticks
    resp = outage_response_ticks(_result(chosen), 1, event, tx_per_tick=2.0, window_tx=20)
    assert np.isfinite(resp)  # tx 200..800 spans the switch at tx 400


def test_realised_sr_and_regret_aggregate_correctly():
    r = RunResult(
        router="t",
        chosen=np.array([0, 0, 1, 1]),
        success=np.array([True, False, True, True]),
        latency_ms=np.array([10.0, 20.0, 30.0, 40.0]),
        instant_regret=np.array([0.1, 0.2, 0.0, 0.0]),
        cost_bps=np.array([10.0, 10.0, 20.0, 20.0]),
        n_gateways=2,
    )
    assert r.realised_sr == pytest.approx(0.75)
    assert r.total_regret == pytest.approx(0.3)
    assert r.mean_cost_bps == pytest.approx(15.0)
    np.testing.assert_allclose(r.cumulative_regret, [0.1, 0.3, 0.3, 0.3])
