"""Regression tests for shortcomings found reviewing the service layer.

Each test here corresponds to a real defect, and names it. They are separated
from `test_service.py` because that file tests the feature; these test the bug.
"""

import threading

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="service layer needs FastAPI")
from fastapi.testclient import TestClient  # noqa: E402

from service.state import RedisConstraintStore  # noqa: E402
from src.agent.constraints import RoutingConstraint  # noqa: E402
from src.gateways import ISSUERS, RoutingContext  # noqa: E402
from src.routers import ContextualThompsonRouter, ThompsonRouter  # noqa: E402

FLEET = ["PG-Alpha", "PG-Bravo", "PG-Charlie", "PG-Delta", "PG-Echo"]


class TtlFakeRedis:
    """A fake that expires keys, because the real one does.

    An earlier fake ignored TTL and therefore re-served expired constraints the
    instant they were pruned, which made a working fix look broken. A fake that
    is easier than the real thing tests nothing.
    """

    def __init__(self):
        self.kv, self.exp, self.now = {}, {}, 0

    def setex(self, key, ttl, value):
        self.kv[key] = value
        self.exp[key] = self.now + ttl

    def scan_iter(self, match=None, count=None):
        return [k for k, e in self.exp.items() if e > self.now]

    def get(self, key):
        return self.kv.get(key) if self.exp.get(key, 0) > self.now else None


# -- decide(): one draw for decision and propensity -----------------------


@pytest.mark.parametrize("cls", [ThompsonRouter, ContextualThompsonRouter])
def test_decide_returns_a_valid_gateway_and_a_distribution(cls):
    router = cls(5, seed=0)
    rng = np.random.default_rng(0)
    context = RoutingContext(tick=0, issuer="HDFC")
    for _ in range(50):
        gateway, probabilities = router.decide(0, context, n_samples=64, rng=rng)
        assert 0 <= gateway < 5
        assert probabilities.shape == (5,)
        assert probabilities.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("cls", [ThompsonRouter, ContextualThompsonRouter])
def test_decide_honours_the_blocked_set(cls):
    router = cls(5, seed=0)
    rng = np.random.default_rng(0)
    blocked = frozenset({0, 1, 2})
    for _ in range(200):
        gateway, probabilities = router.decide(0, None, blocked, n_samples=32, rng=rng)
        assert gateway in {3, 4}
        assert probabilities[list(blocked)].sum() == 0.0


def test_decide_excludes_its_own_decision_draw_from_the_propensity():
    """Otherwise the chosen arm's propensity is biased upward by ~1/n.

    Importance weights divide by that number, so a systematically inflated
    propensity systematically shrinks the weights of exactly the actions that
    were taken -- a bias that would be invisible and would corrupt every
    off-policy estimate built on these logs.
    """
    router = ThompsonRouter(5, seed=0)
    rng = np.random.default_rng(0)
    n = 8  # small n makes the 1/n bias large enough to detect
    chosen_propensities = []
    for _ in range(4000):
        gateway, probabilities = router.decide(0, None, n_samples=n, rng=rng)
        chosen_propensities.append(probabilities[gateway])
    # With five arms at a uniform posterior the mean propensity of the chosen
    # arm should sit near 1/5. Including the decision draw would pull it toward
    # (1 + (n-1)/5) / n = 0.30 at n=8.
    assert np.mean(chosen_propensities) < 0.26


def test_decide_is_cheaper_than_two_separate_passes():
    """The defect: propensity cost 2.1x the decision it was describing."""
    import time

    router = ThompsonRouter(5, seed=0)
    rng = np.random.default_rng(0)
    context = RoutingContext(tick=0, issuer="HDFC")
    for _ in range(500):
        router.decide(0, context, n_samples=64, rng=rng)
        router.select(0, context)

    started = time.perf_counter()
    for _ in range(3000):
        router.decide(0, context, n_samples=64, rng=rng)
    combined = time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(3000):
        router.select(0, context)
        router.action_probabilities(0, context, n_samples=64, rng=rng)
    separate = time.perf_counter() - started

    assert combined < separate


# -- thread safety --------------------------------------------------------


def test_concurrent_requests_keep_the_history_arrays_aligned():
    """FastAPI runs `def` handlers in a threadpool.

    `_append_history` writes five lists and `decisions += 1` is a
    read-modify-write. Interleaved, they can leave the arrays at different
    lengths, and RunResult would then pair a tick with someone else's outcome.
    """
    from service.api import app

    with TestClient(app) as client:
        errors: list[Exception] = []

        def hammer():
            try:
                for i in range(60):
                    issuer = ISSUERS[i % len(ISSUERS)]
                    routed = client.post("/route", json={"issuer": issuer}).json()
                    client.post("/outcome", json={
                        "transaction_id": routed["transaction_id"],
                        "gateway": routed["gateway"], "success": i % 3 != 0,
                        "latency_ms": 1.0, "issuer": issuer,
                        "propensity": routed["propensity"],
                    })
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert client.get("/health").json()["decisions"] == 480

        from service.api import SERVICE

        history = SERVICE.history()
        lengths = {
            history.tick.size, history.chosen.size, history.success.size,
            history.latency_ms.size, history.issuer.size,
        }
        assert len(lengths) == 1, f"history arrays misaligned: {lengths}"
        assert history.tick.size == 480


# -- constraint store growth ---------------------------------------------


def test_expired_constraints_leave_memory():
    """`active()` filtered them from its result but never from the list.

    A long-running replica accumulated every constraint any peer had ever
    written, because the merge re-added them and nothing removed them.
    """
    redis = TtlFakeRedis()
    store = RedisConstraintStore(FLEET, redis, refresh_seconds=0.0)
    for gateway, expires in [("PG-Alpha", 50), ("PG-Bravo", 200), ("PG-Delta", 400)]:
        store.add(RoutingConstraint(
            gateway=gateway, issuer="HDFC", reason="r", created_tick=0,
            expires_tick=expires, confidence=0.8,
            source_scope="issuer_specific", canary_rate=0.0,
        ))

    for tick, expected in [(10, 3), (100, 2), (300, 1), (500, 0)]:
        redis.now = tick
        assert len(store.active(tick)) == expected
        assert len(store.constraints) == expected, "pruning did not reclaim memory"


def test_pruning_keeps_every_live_constraint():
    """Regression for a fix that was itself wrong.

    The first attempt pruned against `max(expires_tick)` rather than the
    current tick, which kept only the longest-lived constraint and silently
    discarded every other live one -- strictly worse than the leak it replaced.
    """
    redis = TtlFakeRedis()
    store = RedisConstraintStore(FLEET, redis, refresh_seconds=0.0)
    for gateway, expires in [("PG-Alpha", 100), ("PG-Bravo", 500), ("PG-Delta", 900)]:
        store.add(RoutingConstraint(
            gateway=gateway, issuer=None, reason="r", created_tick=0,
            expires_tick=expires, confidence=0.8,
            source_scope="single_gateway", canary_rate=0.0,
        ))
    redis.now = 50
    assert len(store.active(50)) == 3, "a live constraint was pruned"


# -- configuration is honoured -------------------------------------------


def test_investigate_uses_the_configured_constraint_settings(monkeypatch):
    """The endpoint hardcoded the defaults, ignoring CONSTRAINT_* entirely."""
    import inspect

    import service.api as api

    source = inspect.getsource(api.investigate)
    assert "settings().constraints" in source
    assert "ttl_minutes=" in source and "canary_rate=" in source


def test_investigate_closes_its_trace_store():
    """Langfuse holds a background thread and Postgres a connection.

    One leaked per investigation would accumulate silently for as long as the
    service stayed up.
    """
    import inspect

    import service.api as api

    source = inspect.getsource(api.investigate)
    assert "traces.close()" in source
    assert "finally:" in source
