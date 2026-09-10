"""The merchant checkout, tested against the real routing service.

The point of this service is that it is a *caller* -- so testing it against a
mocked router would test nothing worth testing. Instead the fixture wires the
merchant's two HTTP helpers to a TestClient holding the real router app, so
every assertion here crosses the actual public contract: `/route` for the
decision, `/simulate/attempt` for the acquirer, `/outcome` for the feedback.

What is deliberately *not* mocked is the failure path. `test_router_down_*`
points the merchant at a closed port and exercises real urllib, because "the
dependency is unreachable" is the one failure a merchant integration must
handle well, and a mock would have proved nothing about it.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="the checkout service needs FastAPI")
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from checkout import app as merchant  # noqa: E402


@pytest.fixture
def wired(monkeypatch):
    """Merchant on top of a live in-process router, over the real API shapes."""
    from service.api import app as router_app

    with TestClient(router_app) as router:
        router.post("/simulate/reset")

        def post(path, payload=None):
            response = router.post(path, json=payload or {})
            if response.status_code >= 400:
                raise HTTPException(502, f"router returned {response.status_code} for {path}")
            return response.json()

        def get(path):
            response = router.get(path)
            if response.status_code >= 400:
                raise HTTPException(502, f"router returned {response.status_code} for {path}")
            return response.json()

        monkeypatch.setattr(merchant, "_post", post)
        monkeypatch.setattr(merchant, "_get", get)
        with TestClient(merchant.app) as client:
            yield client, router


def test_cart_total_is_the_sum_of_its_lines(wired):
    client, _ = wired
    cart = client.get("/cart").json()
    assert cart["total_paise"] == sum(i["amount_paise"] for i in cart["items"])
    assert [b["code"] for b in cart["banks"]] == ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK"]


def test_a_payment_crosses_the_whole_contract(wired):
    client, router = wired
    before = router.get("/simulate/state").json()["decisions"]

    body = client.post("/pay", json={"issuer": "HDFC", "amount_paise": 90900}).json()

    assert body["gateway"] in {g["name"] for g in router.get("/simulate/state").json()["gateways"]}
    assert body["issuer"] == "HDFC"
    assert 0.0 < body["propensity"] <= 1.0
    assert body["decision_micros"] > 0
    assert isinstance(body["success"], bool)
    assert (body["failure_reason"] is None) == body["success"]

    # /route was called and /outcome came back: the router counted the decision
    # and learned from the result.
    assert router.get("/simulate/state").json()["decisions"] == before + 1


def test_payments_teach_the_router(wired):
    """The feedback leg is easy to leave out and impossible to notice without this."""
    client, router = wired
    for _ in range(40):
        client.post("/pay", json={"issuer": "ICICI", "amount_paise": 5000})
    assert router.get("/simulate/state").json()["decisions"] == 40


def test_a_constraint_on_the_router_reaches_the_merchant(wired):
    """The whole reason for two services: a decision made in one is visible in the other.

    The constraint is installed directly rather than by running an investigation.
    Whether the agent earns one on any given run is stochastic -- that is graded
    in `test_agent_evals.py`, and letting it decide here would make this test
    skip itself at random while claiming to check the merchant.
    """
    import service.api as router_api
    from src.agent.constraints import RoutingConstraint

    client, _ = wired
    # Read the module attribute rather than importing the name: /simulate/reset
    # rebinds it, so a `from ... import SERVICE` above would pin the discarded one.
    service = router_api.SERVICE
    tick = service.tick
    service.constraints.add(RoutingConstraint(
        gateway="PG-Delta", issuer="HDFC", reason="planted by a test",
        created_tick=tick, expires_tick=tick + 600, confidence=0.9,
        source_scope="issuer_specific", canary_rate=0.0,
    ))

    for _ in range(6):
        trace = client.post("/pay", json={"issuer": "HDFC", "amount_paise": 5000}).json()
        assert trace["blocked"] == ["PG-Delta"], "the merchant must see the router's constraint"
        assert trace["gateway"] != "PG-Delta", "and must never be routed to a blocked gateway"

    # Scoped, not fleet-wide: another issuer is unaffected.
    other = client.post("/pay", json={"issuer": "ICICI", "amount_paise": 5000}).json()
    assert other["blocked"] == []


def test_unsupported_issuer_is_refused_before_the_router_is_called(wired):
    client, _ = wired
    assert client.post("/pay", json={"issuer": "NOTABANK", "amount_paise": 5000}).status_code == 400
    assert client.post("/pay", json={"issuer": "HDFC", "amount_paise": 0}).status_code == 422


def test_issuer_case_is_normalised(wired):
    client, _ = wired
    assert client.post("/pay", json={"issuer": "hdfc", "amount_paise": 5000}).json()["issuer"] == "HDFC"


def test_background_traffic_moves_the_router_and_not_the_merchant(wired):
    client, router = wired
    body = client.post("/background", json={"count": 300}).json()
    assert body["routed"] == 300
    assert router.get("/simulate/state").json()["decisions"] == 300


# -- the failure path, against real urllib and a closed port -------------------

@pytest.fixture
def unreachable(monkeypatch):
    # Port 1 is reserved and never listening; connecting fails immediately.
    monkeypatch.setattr(merchant, "ROUTER_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(merchant, "ROUTER_TIMEOUT", 2.0)
    with TestClient(merchant.app) as client:
        yield client


def test_router_down_is_502_with_something_actionable(unreachable):
    """A merchant that returns 500 when its dependency is down has lied about whose fault it is."""
    response = unreachable.post("/pay", json={"issuer": "HDFC", "amount_paise": 5000})
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "cannot reach the routing service" in detail
    assert "uvicorn service.api:app" in detail, "the error should say how to fix it"


def test_router_down_leaves_the_merchant_healthy(unreachable):
    """Reported separately so the page can explain the outage instead of looking broken."""
    body = unreachable.get("/health").json()
    assert body["status"] == "ok"
    assert body["router_reachable"] is False
    assert body["router_url"] == "http://127.0.0.1:1"


def test_the_page_is_served_and_only_calls_endpoints_that_exist(wired):
    import re

    client, _ = wired
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]

    known = set(merchant.app.openapi()["paths"])
    called = {p.split("?")[0] for p in re.findall(r'api\([`"](/[^`"]+)[`"]', page.text)}
    assert called, "expected the page to call the merchant API"
    assert called <= known, f"page calls routes that do not exist: {called - known}"


def test_the_page_collects_no_card_details(wired):
    """A demo checkout has no business rendering a field that looks like a card number."""
    client, _ = wired
    page = client.get("/").text.lower()
    for pattern in ('type="password"', 'autocomplete="cc-', "cardnumber", "card-number", "cvv"):
        assert pattern not in page, f"unexpected credential input: {pattern}"
