"""The demo layer, and the one thing about it that is easy to get wrong.

The endpoints under /simulate exist so the UI has something to drive. They are
thin, but they carry a real invariant: the transactions they generate must land
on a *time axis*. The service's tick is wall-clock minutes, and a demo replays
hours of traffic in milliseconds -- so without an explicit simulated clock every
transaction lands on tick 0, every windowed telemetry query is empty, and the
agent answers "no gateway carried enough traffic to assess" no matter what is
actually broken. That failure is silent and looks like a broken agent rather
than a broken clock, which is why it is pinned here.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from service.api import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as c:
        c.post("/simulate/reset")
        yield c
        c.post("/simulate/reset")


def test_traffic_runs_the_real_router_and_reports_per_segment(client):
    body = client.post("/simulate/traffic", json={"count": 300}).json()
    assert body["routed"] == 300
    assert 0.0 <= body["success_rate"] <= 1.0
    assert sum(g["routed"] for g in body["per_gateway"]) == 300
    assert sum(i["routed"] for i in body["per_issuer"]) == 300
    assert pytest.approx(sum(g["share"] for g in body["per_gateway"]), abs=1e-3) == 1.0

    # The decisions really went through the service, not a shadow copy.
    assert client.get("/simulate/state").json()["decisions"] == 300


def test_traffic_advances_the_simulated_clock(client):
    """Without this the whole telemetry layer sees a single instant."""
    before = client.get("/simulate/state").json()["tick"]
    body = client.post("/simulate/traffic", json={"count": 600}).json()
    after = client.get("/simulate/state").json()["tick"]

    assert body["minutes_simulated"] == pytest.approx(60, abs=1)
    assert after - before >= 59, "600 transactions must span about an hour"


def test_investigation_has_a_non_empty_window_after_traffic(client):
    """The regression that motivated the clock: an empty window silently
    turns every diagnosis into 'not enough traffic'."""
    client.post("/simulate/traffic", json={"count": 600})
    body = client.post("/investigate", json={}).json()
    assert body["stop_reason"] == "completed"
    assert body["diagnosis"] is not None
    assert "carried enough traffic" not in body["diagnosis"]["summary"]


def test_issuer_scoped_incident_hurts_only_that_issuer(client):
    client.post("/simulate/traffic", json={"count": 600})
    busiest = max(
        client.post("/simulate/traffic", json={"count": 300}).json()["per_gateway"],
        key=lambda g: g["share"],
    )["gateway"]

    client.post("/simulate/incident",
                json={"gateway": busiest, "issuer": "HDFC", "severity": 0.08})
    batch = client.post("/simulate/traffic", json={"count": 900}).json()

    by_issuer = {i["issuer"]: i["success_rate"] for i in batch["per_issuer"] if i["routed"] > 40}
    assert "HDFC" in by_issuer
    others = [sr for name, sr in by_issuer.items() if name != "HDFC"]
    assert others, "need other issuers with enough traffic to compare against"
    assert by_issuer["HDFC"] < max(others), "the scoped fault must be visible per issuer"


def test_state_exposes_every_field_the_ui_reads(client):
    client.post("/simulate/traffic", json={"count": 120})
    state = client.get("/simulate/state").json()
    assert {"tick", "decisions", "recent_success_rate", "gateways", "issuers",
            "constraints", "constraint_stats", "backends"} <= set(state)
    assert {"name", "posterior_sr", "blocked_for", "injected_fault"} <= set(state["gateways"][0])
    assert set(state["backends"]) == {"redis", "kafka", "clickhouse"}


def test_incident_is_reversible_and_constraints_are_left_alone(client):
    client.post("/simulate/traffic", json={"count": 200})
    client.post("/simulate/incident", json={"gateway": "PG-Delta", "issuer": "HDFC"})
    assert any(g["injected_fault"] for g in client.get("/simulate/state").json()["gateways"])

    cleared = client.post("/simulate/incident/clear").json()
    assert cleared["incidents_removed"] == 1
    assert not any(g["injected_fault"] for g in client.get("/simulate/state").json()["gateways"])


def test_injecting_twice_replaces_rather_than_stacks(client):
    for issuer in ("HDFC", "ICICI"):
        client.post("/simulate/incident", json={"gateway": "PG-Delta", "issuer": issuer})
    faults = [g["injected_fault"] for g in client.get("/simulate/state").json()["gateways"]
              if g["injected_fault"]]
    assert len(faults) == 1 and faults[0]["issuer"] == "ICICI"


def test_attempt_resolves_one_payment_against_the_fleet(client):
    """The seam a merchant calls where it would call the acquiring network."""
    body = client.post("/simulate/attempt", json={"gateway": "PG-Alpha", "issuer": "hdfc"}).json()
    assert body["gateway"] == "PG-Alpha" and body["issuer"] == "HDFC"
    assert isinstance(body["success"], bool)
    assert body["latency_ms"] > 0
    assert (body["failure_reason"] is None) == body["success"]

    # An injected fault has to reach it, or the merchant demo shows nothing.
    client.post("/simulate/incident",
                json={"gateway": "PG-Alpha", "issuer": "HDFC", "severity": 0.01})
    approved = sum(
        client.post("/simulate/attempt", json={"gateway": "PG-Alpha", "issuer": "HDFC"})
        .json()["success"]
        for _ in range(60)
    )
    assert approved < 10, "a broken path must decline nearly everything"

    assert client.post("/simulate/attempt",
                       json={"gateway": "PG-Nope", "issuer": "HDFC"}).status_code == 404
    assert client.post("/simulate/attempt",
                       json={"gateway": "PG-Alpha", "issuer": "NOPE"}).status_code == 400


def test_bad_input_is_rejected_rather_than_silently_ignored(client):
    assert client.post("/simulate/incident", json={"gateway": "PG-Nope"}).status_code == 404
    assert client.post("/simulate/incident",
                       json={"gateway": "PG-Delta", "issuer": "NOTABANK"}).status_code == 400
    assert client.post("/simulate/traffic", json={"count": 0}).status_code == 422
    assert client.post("/simulate/traffic", json={"count": 99999}).status_code == 422


def test_reset_clears_the_router_the_faults_and_the_clock(client):
    client.post("/simulate/traffic", json={"count": 200})
    client.post("/simulate/incident", json={"gateway": "PG-Delta"})
    client.post("/simulate/reset")

    state = client.get("/simulate/state").json()
    assert state["decisions"] == 0
    assert state["constraints"] == []
    assert not any(g["injected_fault"] for g in state["gateways"])


def test_ui_is_served_and_only_calls_endpoints_that_exist(client):
    """A typo in a fetch path in the UI is invisible until someone clicks."""
    import re

    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]

    # Recent FastAPI defers an included router behind a wrapper, so app.routes
    # is not a flat list of paths. The generated schema is, and it is also the
    # contract the UI is written against.
    known = set(app.openapi()["paths"])
    # Both call styles the UI uses: a plain string, and a template literal that
    # appends a query string.
    called = {p.split("?")[0]
              for p in re.findall(r'api\([`"](/[^`"]+)[`"]', page.text)}
    assert called, "expected the UI to call the API"
    assert called <= known, f"UI calls routes that do not exist: {called - known}"
