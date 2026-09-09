"""Service-layer tests: the HTTP surface, metrics, and graceful degradation.

The property under test more than any other is that **every backend is
optional**. Redis, Kafka and ClickHouse are all absent in CI, and the service
must start, route, and answer correctly without them. A router that stops
routing because an analytics sink is down has inverted its own priorities.
"""

import json

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="service layer needs FastAPI")
from fastapi.testclient import TestClient  # noqa: E402

from service import metrics  # noqa: E402
from service.events import ClickHouseSink, EventPipeline, KafkaSink, RoutingEvent  # noqa: E402
from service.state import PosteriorSnapshot, connect  # noqa: E402


@pytest.fixture
def client():
    from service.api import app

    with TestClient(app) as c:
        yield c


def _route_and_report(client, issuer="HDFC", success=True):
    routed = client.post("/route", json={"issuer": issuer}).json()
    client.post("/outcome", json={
        "transaction_id": routed["transaction_id"], "gateway": routed["gateway"],
        "success": success, "latency_ms": 150.0, "issuer": issuer,
        "propensity": routed["propensity"],
    })
    return routed


# -- the hot path ---------------------------------------------------------


def test_health_reports_which_backends_actually_connected(client):
    """An unhealthy service must be distinguishable from one running bare."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["backends"]) == {
        "redis", "posterior_restored_from_snapshot", "kafka", "clickhouse"
    }
    # None are configured in CI, and that is a valid state, not a failure.
    assert body["backends"]["redis"] is False


def test_route_returns_a_real_gateway_and_its_propensity(client):
    body = client.post("/route", json={"issuer": "HDFC"}).json()
    assert body["gateway"] in client.get("/health").json()["gateways"]
    assert 0.0 <= body["propensity"] <= 1.0
    assert body["decision_micros"] > 0


def test_propensity_is_recorded_because_off_policy_evaluation_needs_it(client):
    """The field nobody logs until it is too late.

    Without P(action | context) at decision time, src/ope.py cannot run and the
    logs only measure the policy that produced them.
    """
    body = client.post("/route", json={"issuer": "ICICI"}).json()
    assert "propensity" in body and body["propensity"] > 0


def test_unknown_issuer_is_rejected(client):
    assert client.post("/route", json={"issuer": "NOT-A-BANK"}).status_code == 400


def test_unknown_gateway_in_an_outcome_is_rejected(client):
    response = client.post("/outcome", json={
        "transaction_id": "x", "gateway": "PG-Imaginary", "success": True,
    })
    assert response.status_code == 404


def test_routing_learns_from_reported_outcomes(client):
    """Feeding one gateway nothing but failures should drive traffic away."""
    health = client.get("/health").json()
    for _ in range(60):
        routed = client.post("/route", json={"issuer": "HDFC"}).json()
        client.post("/outcome", json={
            "transaction_id": routed["transaction_id"], "gateway": routed["gateway"],
            "success": routed["gateway"] != health["gateways"][0],
            "latency_ms": 10.0, "issuer": "HDFC", "propensity": routed["propensity"],
        })
    picks = [client.post("/route", json={"issuer": "HDFC"}).json()["gateway"]
             for _ in range(60)]
    assert picks.count(health["gateways"][0]) < 30


def test_decision_latency_stays_far_inside_the_budget(client):
    """The JusTrust brief quotes 100ms. The decision should not be close to it."""
    for _ in range(50):
        _route_and_report(client)
    samples = [client.post("/route", json={"issuer": "SBI"}).json()["decision_micros"]
               for _ in range(200)]
    assert float(np.percentile(samples, 95)) < 5_000  # 5ms, generous for CI


# -- the cold path --------------------------------------------------------


def test_investigate_refuses_when_there_is_no_telemetry(client):
    """A clear 409 beats inventing a diagnosis from nothing."""
    response = client.post("/investigate")
    assert response.status_code == 409
    assert "no telemetry" in response.json()["detail"]


def test_investigate_runs_once_traffic_exists(client):
    for i in range(200):
        _route_and_report(client, issuer=["HDFC", "ICICI", "SBI"][i % 3])
    body = client.post("/investigate").json()
    assert body["stop_reason"] == "completed"
    assert body["diagnosis"] is not None
    # Healthy synthetic traffic has no planted incident, so no constraint
    # should be installed. Refusing to act is the correct outcome.
    assert body["constraint_installed"] is None


def test_constraints_endpoint_is_readable(client):
    body = client.get("/constraints").json()
    assert body["active"] == []
    assert "canary_releases" in body["stats"]


# -- metrics --------------------------------------------------------------


def test_metrics_endpoint_exposes_the_prometheus_format(client):
    _route_and_report(client)
    text = client.get("/metrics").text
    assert "# HELP routing_decision_duration_seconds" in text
    assert "# TYPE routing_decision_duration_seconds histogram" in text
    assert 'routing_decision_duration_seconds_bucket{le="+Inf"}' in text
    assert "routing_decisions_total{" in text


def test_histogram_buckets_are_cumulative():
    """The part hand-rolled implementations usually get wrong."""
    h = metrics.Histogram("t_seconds", "help", (0.001, 0.01, 0.1))
    for value in (0.0005, 0.005, 0.05, 5.0):
        h.observe(value)
    lines = list(h.samples())
    counts = {
        line.split("le=\"")[1].split("\"")[0]: int(line.split()[-1])
        for line in lines if "_bucket" in line
    }
    assert counts["0.001"] == 1
    assert counts["0.01"] == 2      # includes the 0.0005 observation
    assert counts["0.1"] == 3
    assert counts["+Inf"] == 4      # every observation, including the 5.0
    assert [l for l in lines if l.startswith("t_seconds_count")][0].endswith(" 4")


def test_counter_labels_render_and_escape():
    c = metrics.Counter("things_total", "help")
    c.inc(2, gateway="PG-A", outcome="success")
    c.inc(gateway='weird"name')
    rendered = "\n".join(c.samples())
    assert 'things_total{gateway="PG-A",outcome="success"} 2' in rendered
    assert '\\"' in rendered  # the quote is escaped, not emitted raw


# -- graceful degradation -------------------------------------------------


def test_every_sink_is_a_no_op_when_unconfigured():
    pipeline = EventPipeline()
    event = RoutingEvent("t1", 0, "HDFC", "PG-Alpha", True, 10.0, 0.3, "test")
    pipeline.publish(event)  # must not raise
    pipeline.flush()
    assert pipeline.kafka.enabled is False
    assert pipeline.clickhouse.enabled is False


def test_redis_connect_returns_none_rather_than_raising():
    assert connect(None) is None
    # An unroutable address: must give up on the timeout, not propagate.
    assert connect("redis://10.255.255.1:6379/0", timeout=0.25) is None


def test_posterior_snapshot_is_inert_without_redis():
    from src.routers import ThompsonRouter

    snapshot = PosteriorSnapshot(None)
    router = ThompsonRouter(5, seed=0)
    assert snapshot.enabled is False
    assert snapshot.maybe_save(router) is False
    assert snapshot.restore(router) is False


def test_clickhouse_sink_buffers_then_drops_when_unreachable():
    """A lost analytics row is cheaper than a stalled payment."""
    sink = ClickHouseSink("http://10.255.255.1:8123", batch_size=2, timeout=0.25)
    event = RoutingEvent("t1", 0, "HDFC", "PG-Alpha", True, 10.0, 0.3, "test")
    assert sink.publish(event) is True          # buffered
    assert sink.publish(event) is False         # flush attempted and failed
    assert sink.failures == 2
    assert sink.last_error is not None


def test_kafka_sink_reports_a_missing_broker_without_raising():
    sink = KafkaSink(None)
    assert sink.enabled is False
    assert sink.publish(
        RoutingEvent("t1", 0, "HDFC", "PG-Alpha", True, 10.0, 0.3, "test")
    ) is False


def test_routing_event_serialises_for_clickhouse():
    event = RoutingEvent("t1", 42, "HDFC", "PG-Alpha", True, 12.5, 0.31, "thompson")
    row = event.to_dict()
    assert row["success"] == 1 and row["constrained"] == 0  # UInt8, not bool
    assert row["ts"].count(":") == 2 and "." in row["ts"]   # DateTime64(3)
    json.dumps(row)  # must be JSONEachRow-serialisable
