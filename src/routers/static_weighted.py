"""Fixed-weight routing -- the baseline every PSP starts with."""

from __future__ import annotations

import numpy as np

from ..gateways import Outcome, RoutingContext
from .base import Router


class StaticWeightedRouter(Router):
    """Splits traffic by fixed weights and never learns.

    This is not a strawman. Weighted round-robin negotiated in a commercial
    agreement is how a large share of production payment traffic is actually
    routed. It is the number every adaptive strategy has to beat, and it fails
    in exactly one way: when a gateway degrades, it keeps sending traffic.
    """

    name = "static-weighted"

    def __init__(
        self, n_gateways: int, weights: list[float] | None = None, seed: int = 0
    ) -> None:
        super().__init__(n_gateways, seed)
        if weights is None:
            weights = [1.0] * n_gateways
        if len(weights) != n_gateways:
            raise ValueError("weights length must match gateway count")
        w = np.asarray(weights, dtype=float)
        if w.min() < 0 or w.sum() <= 0:
            raise ValueError("weights must be non-negative and sum to > 0")
        self.weights = w / w.sum()

    def select(
        self,
        tick: int,
        context: RoutingContext | None = None,
        blocked: frozenset[int] = frozenset(),
    ) -> int:
        if not blocked:
            return int(self.rng.choice(self.n_gateways, p=self.weights))
        weights = self.weights.copy()
        weights[list(blocked)] = 0.0
        total = weights.sum()
        if total <= 0:  # everything blocked -- fall back rather than strand
            return int(self.rng.choice(self.n_gateways, p=self.weights))
        return int(self.rng.choice(self.n_gateways, p=weights / total))

    def action_probabilities(
        self,
        tick: int,
        context: RoutingContext | None = None,
        blocked: frozenset[int] = frozenset(),
        n_samples: int = 128,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Exactly the configured weights, renormalised over allowed gateways."""
        weights = self.weights.copy()
        if blocked:
            weights[list(blocked)] = 0.0
        total = weights.sum()
        return self.weights.copy() if total <= 0 else weights / total

    def update(self, outcome: Outcome) -> None:
        pass  # Deliberately blind.

    def estimated_sr(self) -> np.ndarray:
        return self.weights.copy()
