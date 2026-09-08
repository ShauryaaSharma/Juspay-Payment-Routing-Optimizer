"""Thompson sampling over Beta posteriors."""

from __future__ import annotations

import numpy as np

from ..gateways import Outcome, RoutingContext
from .base import DiscountedCounts, Router


class ThompsonRouter(Router):
    """Sample a plausible SR per gateway, route to the winner.

    Probability matching gives smoother traffic allocation than UCB1's
    argmax -- allocation degrades gracefully as evidence accumulates instead
    of flipping hard between arms, which matters when downstream gateways
    have capacity commitments that hate step changes.
    """

    name = "thompson"

    def __init__(self, n_gateways: int, gamma: float = 1.0, seed: int = 0) -> None:
        super().__init__(n_gateways, seed)
        self.counts = DiscountedCounts(n_gateways, gamma=gamma)

    def _sample(self, inflation: float = 1.0) -> np.ndarray:
        # Dividing both Beta parameters by `inflation` preserves the posterior
        # mean while widening its variance -- deliberate, controllable doubt.
        alpha = np.maximum(self.counts.successes / inflation, 1e-3)
        beta = np.maximum(self.counts.failures / inflation, 1e-3)
        return self.rng.beta(alpha, beta)

    def select(
        self,
        tick: int,
        context: RoutingContext | None = None,
        blocked: frozenset[int] = frozenset(),
    ) -> int:
        samples = self._sample()
        if blocked:
            samples[list(blocked)] = -np.inf
        return int(np.argmax(samples))

    def action_probabilities(
        self,
        tick: int,
        context: RoutingContext | None = None,
        blocked: frozenset[int] = frozenset(),
        n_samples: int = 128,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Estimated by Monte Carlo: Thompson has no closed form for P(argmax).

        Uses the caller's rng so the router's own decision stream is untouched.
        """
        rng = rng if rng is not None else np.random.default_rng(0)
        alpha = np.maximum(self.counts.successes, 1e-3)
        beta = np.maximum(self.counts.failures, 1e-3)
        draws = rng.beta(alpha, beta, size=(n_samples, self.n_gateways))
        if blocked:
            draws[:, list(blocked)] = -np.inf
        winners = np.argmax(draws, axis=1)
        return np.bincount(winners, minlength=self.n_gateways) / n_samples

    def update(self, outcome: Outcome) -> None:
        self.counts.update(outcome.gateway, outcome.success)

    def estimated_sr(self) -> np.ndarray:
        return self.counts.mean
