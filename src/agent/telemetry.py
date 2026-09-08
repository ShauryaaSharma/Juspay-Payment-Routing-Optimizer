"""The queryable view of what actually happened -- what the agent's tools read.

This is deliberately the *observable* surface: counts of transactions that were
routed and their outcomes. It never exposes ground-truth success rates, which
the simulator knows and a production system could not. If the agent could read
`true_sr`, the eval would be measuring nothing.

That constraint has teeth. A gateway the router pulled traffic away from has
almost no observations, so its recent success rate is genuinely unknowable --
the agent has to reason about survivorship, not just read a number off a
dashboard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..gateways import ISSUERS, GatewaySpec
from ..metrics import RunResult

MIN_SAMPLES_FOR_SR = 30  # below this, a success rate is noise dressed as signal


def format_clock(minute: int) -> str:
    """Render a tick as ``day 3, 14:05`` -- how an on-call engineer reads time."""
    day, rem = divmod(int(minute), 24 * 60)
    return f"day {day + 1}, {rem // 60:02d}:{rem % 60:02d}"


class TelemetryStore:
    """Windowed aggregate queries over one simulated week of transactions."""

    def __init__(self, result: RunResult, specs: list[GatewaySpec]) -> None:
        if result.tick is None or result.issuer is None:
            raise ValueError("RunResult must carry tick and issuer arrays")
        self.result = result
        self.specs = specs
        self.names = [s.name for s in specs]
        self._tick = np.asarray(result.tick)
        self._issuer = np.asarray(result.issuer)
        self._chosen = np.asarray(result.chosen)
        self._success = np.asarray(result.success)
        self._latency = np.asarray(result.latency_ms)

    # -- helpers -----------------------------------------------------------

    def gateway_index(self, name: str) -> int:
        """Resolve a gateway name, case-insensitively.

        Raises KeyError with the valid options; the loop turns that into a
        tool error the model can recover from rather than a crash.
        """
        lookup = {n.lower(): i for i, n in enumerate(self.names)}
        key = str(name).strip().lower()
        if key not in lookup:
            raise KeyError(f"unknown gateway {name!r}; valid: {', '.join(self.names)}")
        return lookup[key]

    def _slice(self, start_min: int, end_min: int) -> slice:
        if end_min <= start_min:
            raise ValueError(f"end_minute ({end_min}) must be greater than start_minute ({start_min})")
        lo = int(np.searchsorted(self._tick, start_min, side="left"))
        hi = int(np.searchsorted(self._tick, end_min, side="left"))
        return slice(lo, hi)

    @staticmethod
    def _sr(success: np.ndarray) -> float | None:
        """Success rate, or None when there is not enough traffic to say."""
        if success.size < MIN_SAMPLES_FOR_SR:
            return None
        return round(float(success.mean()), 4)

    # -- tool-facing queries ----------------------------------------------

    def list_gateways(self) -> dict[str, Any]:
        return {
            "gateways": [
                {"name": s.name, "cost_bps": s.cost_bps, "typical_latency_ms": s.base_latency_ms}
                for s in self.specs
            ],
            "issuers": list(ISSUERS),
            "window_available_minutes": [int(self._tick[0]), int(self._tick[-1]) + 1],
        }

    def gateway_health(self, gateway: str, start_minute: int, end_minute: int) -> dict[str, Any]:
        idx = self.gateway_index(gateway)
        sl = self._slice(start_minute, end_minute)
        mask = self._chosen[sl] == idx
        success = self._success[sl][mask]
        latency = self._latency[sl][mask]

        out: dict[str, Any] = {
            "gateway": self.names[idx],
            "window": [start_minute, end_minute],
            "window_readable": f"{format_clock(start_minute)} to {format_clock(end_minute)}",
            "transactions": int(mask.sum()),
            "success_rate": self._sr(success),
        }
        if latency.size:
            out["p50_latency_ms"] = round(float(np.percentile(latency, 50)), 1)
            out["p95_latency_ms"] = round(float(np.percentile(latency, 95)), 1)
        if out["success_rate"] is None:
            out["note"] = (
                f"only {int(mask.sum())} transactions were routed here in this window "
                f"(minimum {MIN_SAMPLES_FOR_SR} for a reliable rate). The router may "
                f"have already shifted traffic away."
            )
        return out

    def fleet_health(self, start_minute: int, end_minute: int) -> dict[str, Any]:
        sl = self._slice(start_minute, end_minute)
        chosen, success = self._chosen[sl], self._success[sl]
        per_gateway = []
        for i, name in enumerate(self.names):
            mask = chosen == i
            per_gateway.append({
                "gateway": name,
                "transactions": int(mask.sum()),
                "traffic_share": round(float(mask.mean()), 4) if chosen.size else 0.0,
                "success_rate": self._sr(success[mask]),
            })
        return {
            "window": [start_minute, end_minute],
            "window_readable": f"{format_clock(start_minute)} to {format_clock(end_minute)}",
            "fleet_success_rate": self._sr(success),
            "total_transactions": int(chosen.size),
            "per_gateway": per_gateway,
        }

    def segment_failures(
        self, gateway: str, start_minute: int, end_minute: int, by: str = "issuer"
    ) -> dict[str, Any]:
        """Break one gateway's traffic down by a dimension.

        The tool that finds what a fleet-level success rate hides.
        """
        if by != "issuer":
            raise ValueError(f"unsupported segmentation {by!r}; only 'issuer' is available")
        idx = self.gateway_index(gateway)
        sl = self._slice(start_minute, end_minute)
        mask = self._chosen[sl] == idx
        issuers, success = self._issuer[sl][mask], self._success[sl][mask]

        segments = []
        for j, issuer_name in enumerate(ISSUERS):
            sub = issuers == j
            segments.append({
                "issuer": issuer_name,
                "transactions": int(sub.sum()),
                "success_rate": self._sr(success[sub]),
            })
        return {
            "gateway": self.names[idx],
            "window": [start_minute, end_minute],
            "segmented_by": "issuer",
            "overall_success_rate": self._sr(success),
            "segments": segments,
        }

    def compare_windows(
        self,
        gateway: str,
        baseline_start: int,
        baseline_end: int,
        current_start: int,
        current_end: int,
    ) -> dict[str, Any]:
        """Before/after on one gateway -- the shape of 'what changed'."""
        before = self.gateway_health(gateway, baseline_start, baseline_end)
        after = self.gateway_health(gateway, current_start, current_end)
        delta = None
        if before["success_rate"] is not None and after["success_rate"] is not None:
            delta = round(after["success_rate"] - before["success_rate"], 4)
        return {
            "gateway": before["gateway"],
            "baseline": before,
            "current": after,
            "success_rate_delta": delta,
            "delta_bps": None if delta is None else round(delta * 10_000, 1),
        }

    def traffic_shift(self, baseline_start: int, baseline_end: int,
                      current_start: int, current_end: int) -> dict[str, Any]:
        """How the router redistributed traffic between two windows.

        A large shift away from a gateway is itself evidence: the router
        detected something before anyone opened a dashboard.
        """
        before = self.fleet_health(baseline_start, baseline_end)["per_gateway"]
        after = self.fleet_health(current_start, current_end)["per_gateway"]
        shifts = [
            {
                "gateway": b["gateway"],
                "baseline_share": b["traffic_share"],
                "current_share": a["traffic_share"],
                "share_change": round(a["traffic_share"] - b["traffic_share"], 4),
            }
            for b, a in zip(before, after)
        ]
        shifts.sort(key=lambda s: s["share_change"])
        return {"shifts": shifts}
