"""Tool surface for the investigation agent.

Six tools, all read-only, all backed by the same telemetry store. The surface
is deliberately small: every extra tool is a chance for the model to pick the
wrong one, and each of these answers a question an on-call engineer actually
asks. `segment_failures` is the one that earns its place -- it is the only way
to see a degradation that the fleet-level success rate averages into invisibility.

Every tool is `strict: true`, so arguments are schema-valid before they reach
the dispatcher. That does not make them *sensible* -- a valid call can still
name a gateway that does not exist or invert a window -- so the dispatcher
returns a structured error the model can read and retry from, rather than
raising.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .schemas import ToolCall
from .telemetry import TelemetryStore

_WINDOW_PROPS = {
    "start_minute": {"type": "integer", "description": "Window start, minutes since the week began."},
    "end_minute": {"type": "integer", "description": "Window end (exclusive), minutes since the week began."},
}


def tool_definitions() -> list[dict[str, Any]]:
    """The tool list sent to the API.

    Ordering is fixed and the content is static -- both matter for prompt
    caching, since tools are rendered before system and messages, so any
    churn here invalidates the cached prefix for every request.
    """
    return [
        {
            "name": "list_gateways",
            "description": (
                "List every payment gateway in the fleet with its cost and typical latency, "
                "plus the issuing banks that appear in traffic and the time range for which "
                "telemetry exists. Call this first if you do not already know the fleet."
            ),
            "strict": True,
            "input_schema": {
                "type": "object", "properties": {}, "required": [], "additionalProperties": False,
            },
        },
        {
            "name": "get_fleet_health",
            "description": (
                "Success rate, transaction count, and traffic share for every gateway over a "
                "time window, plus the fleet-wide success rate. Use this to see whether a "
                "problem is isolated to one gateway or affects all of them."
            ),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": dict(_WINDOW_PROPS),
                "required": ["start_minute", "end_minute"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_gateway_health",
            "description": (
                "Success rate, transaction volume, and p50/p95 latency for ONE gateway over a "
                "time window. Returns success_rate=null when too little traffic was routed "
                "there to compute a reliable rate."
            ),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "gateway": {"type": "string", "description": "Gateway name, e.g. PG-Alpha."},
                    **_WINDOW_PROPS,
                },
                "required": ["gateway", "start_minute", "end_minute"],
                "additionalProperties": False,
            },
        },
        {
            "name": "segment_failures",
            "description": (
                "Break one gateway's traffic down by issuing bank over a window, giving the "
                "success rate per issuer. Use this when a gateway's overall success rate is "
                "soft but not catastrophic -- a failure confined to a single issuer looks mild "
                "in aggregate and severe once segmented."
            ),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "gateway": {"type": "string", "description": "Gateway name, e.g. PG-Alpha."},
                    **_WINDOW_PROPS,
                    "by": {"type": "string", "enum": ["issuer"], "description": "Segmentation dimension."},
                },
                "required": ["gateway", "start_minute", "end_minute", "by"],
                "additionalProperties": False,
            },
        },
        {
            "name": "compare_windows",
            "description": (
                "Compare one gateway's health between a baseline window and a current window, "
                "returning the success-rate delta in basis points. Use this to establish that "
                "something changed, and when."
            ),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "gateway": {"type": "string"},
                    "baseline_start": {"type": "integer"},
                    "baseline_end": {"type": "integer"},
                    "current_start": {"type": "integer"},
                    "current_end": {"type": "integer"},
                },
                "required": [
                    "gateway", "baseline_start", "baseline_end", "current_start", "current_end",
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_traffic_shift",
            "description": (
                "Show how the router redistributed traffic between two windows, sorted by "
                "largest loss of share first. A gateway the router pulled away from is "
                "evidence in itself -- and explains why its recent sample may be too small "
                "to measure."
            ),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "baseline_start": {"type": "integer"},
                    "baseline_end": {"type": "integer"},
                    "current_start": {"type": "integer"},
                    "current_end": {"type": "integer"},
                },
                "required": ["baseline_start", "baseline_end", "current_start", "current_end"],
                "additionalProperties": False,
            },
        },
    ]


TOOL_NAMES: tuple[str, ...] = tuple(t["name"] for t in tool_definitions())


class ToolDispatcher:
    """Routes a tool call to the telemetry store and formats the result.

    Never raises on bad input. A tool error is a normal event in an agent loop
    -- the model reads the message and corrects itself -- so errors come back
    as content with ``is_error`` set, and the loop counts them.
    """

    def __init__(self, store: TelemetryStore) -> None:
        self.store = store

    def dispatch(self, name: str, arguments: dict[str, Any]) -> ToolCall:
        started = time.perf_counter()
        try:
            payload = self._invoke(name, arguments)
            is_error = False
        except (KeyError, ValueError, TypeError) as exc:
            payload = {"error": str(exc)}
            is_error = True
        return ToolCall(
            name=name,
            arguments=arguments,
            result=payload,
            is_error=is_error,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def _invoke(self, name: str, args: dict[str, Any]) -> Any:
        store = self.store
        if name == "list_gateways":
            return store.list_gateways()
        if name == "get_fleet_health":
            return store.fleet_health(int(args["start_minute"]), int(args["end_minute"]))
        if name == "get_gateway_health":
            return store.gateway_health(
                str(args["gateway"]), int(args["start_minute"]), int(args["end_minute"])
            )
        if name == "segment_failures":
            return store.segment_failures(
                str(args["gateway"]),
                int(args["start_minute"]),
                int(args["end_minute"]),
                str(args.get("by", "issuer")),
            )
        if name == "compare_windows":
            return store.compare_windows(
                str(args["gateway"]),
                int(args["baseline_start"]), int(args["baseline_end"]),
                int(args["current_start"]), int(args["current_end"]),
            )
        if name == "get_traffic_shift":
            return store.traffic_shift(
                int(args["baseline_start"]), int(args["baseline_end"]),
                int(args["current_start"]), int(args["current_end"]),
            )
        raise KeyError(f"unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}")

    @staticmethod
    def render(call: ToolCall) -> str:
        return json.dumps(call.result, separators=(",", ":"), sort_keys=True)
