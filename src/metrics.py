"""Metrics that distinguish a good router from a lucky one.

Realised success rate is the number the business cares about, but on its own
it hides everything interesting: two routers can post the same SR while one
of them spent an outage bleeding traffic into a dead gateway and the other
did not. Regret and outage-response time are what expose that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gateways import DegradationEvent, GatewayEnvironment


@dataclass
class RunResult:
    """Per-transaction record of one router's pass over one environment."""

    router: str
    chosen: np.ndarray  # (horizon,) int  -- gateway index per transaction
    success: np.ndarray  # (horizon,) bool
    latency_ms: np.ndarray  # (horizon,) float
    instant_regret: np.ndarray  # (horizon,) float -- oracle SR minus chosen SR
    cost_bps: np.ndarray  # (horizon,) float
    n_gateways: int
    tick: np.ndarray | None = None  # (horizon,) int -- simulated minute
    issuer: np.ndarray | None = None  # (horizon,) int -- index into ISSUERS

    @property
    def horizon(self) -> int:
        return self.chosen.size

    @property
    def realised_sr(self) -> float:
        return float(self.success.mean())

    @property
    def cumulative_regret(self) -> np.ndarray:
        return np.cumsum(self.instant_regret)

    @property
    def total_regret(self) -> float:
        return float(self.instant_regret.sum())

    @property
    def mean_cost_bps(self) -> float:
        return float(self.cost_bps.mean())

    def rolling_sr(self, window: int = 500) -> np.ndarray:
        return _rolling_mean(self.success.astype(float), window)

    def allocation(self) -> np.ndarray:
        """(n_gateways, horizon) one-hot traffic allocation."""
        alloc = np.zeros((self.n_gateways, self.horizon))
        alloc[self.chosen, np.arange(self.horizon)] = 1.0
        return alloc

    def latency_percentile(self, p: float) -> float:
        return float(np.percentile(self.latency_ms, p))


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean, left-padded so the output aligns with the input."""
    if window <= 1:
        return x.astype(float)
    csum = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = np.empty_like(x, dtype=float)
    idx = np.arange(x.size)
    lo = np.maximum(0, idx - window + 1)
    out = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)
    return out


def outage_response_ticks(
    result: RunResult,
    gateway: int,
    event: DegradationEvent,
    tx_per_tick: float,
    share_threshold: float = 0.05,
    window_tx: int = 400,
) -> float:
    """How long the router kept feeding a degraded gateway, in simulated minutes.

    ``result`` is indexed by transaction while ``event`` is expressed in ticks,
    so the window is converted into transaction space before scanning. Returns
    ``inf`` if the router never pulled below ``share_threshold`` before the
    event ended -- which for a static router is the expected, damning result.
    """
    share = _rolling_mean((result.chosen == gateway).astype(float), window_tx)
    lo = int(event.start * tx_per_tick)
    hi = min(int(event.end * tx_per_tick), result.horizon)
    for i in range(lo, hi):
        if share[i] < share_threshold:
            return float((i - lo) / tx_per_tick)
    return float("inf")


def summarise(result: RunResult, env: GatewayEnvironment) -> dict[str, float]:
    """Flat dict of headline numbers, ready for a CSV row."""
    return {
        "realised_sr": result.realised_sr,
        "total_regret": result.total_regret,
        "mean_cost_bps": result.mean_cost_bps,
        "p50_latency_ms": result.latency_percentile(50),
        "p95_latency_ms": result.latency_percentile(95),
        "p99_latency_ms": result.latency_percentile(99),
    }
