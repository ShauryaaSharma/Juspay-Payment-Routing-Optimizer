"""epsilon-greedy: the simplest thing that adapts at all."""

from __future__ import annotations

import numpy as np

from ..gateways import Outcome, RoutingContext
from .base import DiscountedCounts, Router


class EpsilonGreedyRouter(Router):
    """Exploit the best-known gateway, explore uniformly ``epsilon`` of the time.

    Its weakness is that exploration cost is paid at a constant rate forever,
    and that the exploration is undirected -- it is as likely to probe an arm
    it already knows is terrible as one it is genuinely uncertain about.
    """

    name = "epsilon-greedy"

    def __init__(
        self,
        n_gateways: int,
        epsilon: float = 0.10,
        gamma: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(n_gateways, seed)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
        self.counts = DiscountedCounts(n_gateways, gamma=gamma)

    def select(
        self,
        tick: int,
        context: RoutingContext | None = None,
        blocked: frozenset[int] = frozenset(),
    ) -> int:
        explore = self.rng.random() < self.epsilon
        if not blocked:
            # Original path, preserved exactly so RNG consumption is unchanged.
            if explore:
                return int(self.rng.integers(self.n_gateways))
            means = self.counts.mean
            # Random tie-break so a cold start does not lock onto index 0.
            best = np.flatnonzero(means == means.max())
            return int(self.rng.choice(best))

        allowed = [i for i in range(self.n_gateways) if i not in blocked]
        if not allowed:  # every gateway blocked: the constraint layer must not
            return int(self.rng.integers(self.n_gateways))  # strand a transaction
        if explore:
            return int(self.rng.choice(allowed))
        means = self.counts.mean.copy()
        means[list(blocked)] = -np.inf
        best = np.flatnonzero(means == means.max())
        return int(self.rng.choice(best))

    def action_probabilities(
        self,
        tick: int,
        context: RoutingContext | None = None,
        blocked: frozenset[int] = frozenset(),
        n_samples: int = 128,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Closed form -- no Monte Carlo error, unlike the Thompson variants.

        epsilon spread uniformly over the allowed arms, the remainder split
        evenly among those tied for the best estimate.
        """
        allowed = np.array([i not in blocked for i in range(self.n_gateways)])
        if not allowed.any():
            return np.full(self.n_gateways, 1.0 / self.n_gateways)

        probs = np.where(allowed, self.epsilon / allowed.sum(), 0.0)
        means = np.where(allowed, self.counts.mean, -np.inf)
        best = means == means.max()
        probs = probs + np.where(best, (1.0 - self.epsilon) / best.sum(), 0.0)
        return probs / probs.sum()

    def update(self, outcome: Outcome) -> None:
        self.counts.update(outcome.gateway, outcome.success)

    def estimated_sr(self) -> np.ndarray:
        return self.counts.mean
