"""The merchant side of the integration.

The routing service has, until now, had no caller. `/simulate/traffic` plays
both the merchant and the gateway, which is honest for a volume demo but means
nothing in the repo exercises the public contract the way an integrator would.

This does. It is a separate process, on a separate port, sharing no memory with
the router -- it speaks the same HTTP API a real merchant backend would speak,
in the same order:

    1. POST /route             "where should this payment go?"
    2. POST /simulate/attempt  the acquiring network, simulated
    3. POST /outcome           "here is what happened"

Step 2 is the only fiction. Steps 1 and 3 are the production contract.

Two design notes, both decisions rather than accidents:

**The browser never talks to the router.** A routing API decides where money
goes; that is a backend concern. The page posts here, and this service calls
the router server-to-server, which is how a real integration works and why
there is no CORS configuration anywhere in this repo.

**No HTTP client dependency.** urllib is enough for three JSON POSTs, and the
rule in this project is that a dependency has to earn its place.
`service/events.py` posts to ClickHouse the same way.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError("the checkout service needs FastAPI") from exc

# 127.0.0.1 rather than "localhost" deliberately. On Windows the name resolves
# to ::1 first, and urllib waits for that connection to fail before retrying
# IPv4 -- which turned a 5 ms round trip into 6 seconds and made the routing
# decision look slow when the router was answering in microseconds.
ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8000").rstrip("/")
ROUTER_TIMEOUT = float(os.environ.get("ROUTER_TIMEOUT_SECONDS", "10"))

_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# A fictional storefront. The point is to have a caller, not to imitate anyone.
MERCHANT = {
    "name": "Kirana Cart",
    "order_id": "ORD-4417-2290",
    "items": [
        {"label": "Basmati rice, 5 kg", "amount_paise": 62000},
        {"label": "Cold-pressed mustard oil, 1 L", "amount_paise": 24900},
        {"label": "Delivery", "amount_paise": 4000},
    ],
}
BANKS = [
    {"code": "HDFC", "name": "HDFC Bank"},
    {"code": "ICICI", "name": "ICICI Bank"},
    {"code": "SBI", "name": "State Bank of India"},
    {"code": "AXIS", "name": "Axis Bank"},
    {"code": "KOTAK", "name": "Kotak Mahindra Bank"},
]
BANK_CODES = {bank["code"] for bank in BANKS}


def _post(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """POST JSON to the router and return the decoded reply.

    Router failures surface as 502 rather than 500: the merchant is fine, its
    dependency is not, and whoever reads the log should not have to guess which.
    """
    body = json.dumps(payload or {}).encode()
    request = urllib.request.Request(
        f"{ROUTER_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=ROUTER_TIMEOUT) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        raise HTTPException(502, f"router returned {exc.code} for {path}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HTTPException(
            502,
            f"cannot reach the routing service at {ROUTER_URL} ({exc}). "
            f"Start it with: uvicorn service.api:app --port 8000",
        ) from exc


def _get(path: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{ROUTER_URL}{path}", timeout=ROUTER_TIMEOUT) as response:
            return json.loads(response.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HTTPException(
            502, f"cannot reach the routing service at {ROUTER_URL} ({exc})"
        ) from exc


class PayRequest(BaseModel):
    issuer: str = Field(..., description="Issuing bank selected at checkout.")
    amount_paise: int = Field(..., ge=100, le=10_000_000)


class BackgroundRequest(BaseModel):
    count: int = Field(600, ge=1, le=20000, description="Background transactions to generate.")


app = FastAPI(
    title="Kirana Cart checkout",
    description=(
        "A merchant integrating against the payment routing service. Separate "
        "process, separate port, the same public API a real integrator uses."
    ),
)


@app.get("/", include_in_schema=False)
def index():
    page = os.path.join(_STATIC, "index.html")
    if not os.path.exists(page):
        return JSONResponse({"service": "checkout", "ui": "not installed", "docs": "/docs"})
    return FileResponse(page)


@app.get("/health")
def health() -> dict[str, Any]:
    """Our health and, separately, whether the router is reachable.

    Kept distinct on purpose: this service is up even when its dependency is
    down, and the page uses that to explain the failure rather than showing a
    dead button.
    """
    reachable, detail = True, "ok"
    try:
        _get("/health")
    except HTTPException as exc:
        reachable, detail = False, str(exc.detail)
    return {
        "status": "ok",
        "router_url": ROUTER_URL,
        "router_reachable": reachable,
        "router_detail": detail,
    }


@app.get("/cart")
def cart() -> dict[str, Any]:
    total = sum(item["amount_paise"] for item in MERCHANT["items"])
    return {**MERCHANT, "banks": BANKS, "total_paise": total}


@app.post("/pay")
def pay(request: PayRequest) -> dict[str, Any]:
    """One payment, the whole way through, with the trace kept.

    The trace is why this page exists: it shows the routing decision the
    merchant received, which gateways the constraint layer removed before the
    bandit ever saw them, and the propensity that makes the decision replayable
    for off-policy evaluation later. A real merchant would log exactly this.
    """
    issuer = request.issuer.upper()
    if issuer not in BANK_CODES:
        raise HTTPException(400, f"unsupported issuer {request.issuer!r}")

    transaction_id = uuid.uuid4().hex
    started = time.perf_counter()

    decision = _post("/route", {
        "issuer": issuer,
        "amount_paise": request.amount_paise,
        "transaction_id": transaction_id,
    })
    gateway = decision["gateway"]

    # Where a real merchant would call the acquirer.
    result = _post("/simulate/attempt", {"gateway": gateway, "issuer": issuer})

    _post("/outcome", {
        "transaction_id": transaction_id,
        "gateway": gateway,
        "success": result["success"],
        "latency_ms": result["latency_ms"],
        "issuer": issuer,
        "propensity": decision["propensity"],
    })

    return {
        "transaction_id": transaction_id,
        "issuer": issuer,
        "amount_paise": request.amount_paise,
        "gateway": gateway,
        "success": result["success"],
        "failure_reason": result["failure_reason"],
        "gateway_latency_ms": result["latency_ms"],
        "propensity": decision["propensity"],
        "blocked": decision.get("blocked", []),
        "decision_micros": decision["decision_micros"],
        "merchant_round_trip_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@app.post("/background")
def background(request: BackgroundRequest) -> dict[str, Any]:
    """Generate background traffic on the router.

    A single shopper cannot teach a bandit anything -- a posterior needs
    hundreds of transactions to move. This is the rest of the merchant's
    traffic, which in production would be arriving anyway.
    """
    return _post("/simulate/traffic", {"count": request.count})


@app.get("/router/state")
def router_state() -> dict[str, Any]:
    """Proxy the router's view, so the page can show what its payments did."""
    return _get("/simulate/state")
