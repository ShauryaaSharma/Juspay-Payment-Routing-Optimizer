"""UCB1: optimism in the face of uncertainty."""

from __future__ import annotations

import numpy as np

from ..gateways import Outcome, RoutingContext
from .base import DiscountedCounts, Router


class UCB1Router(Router):
    """Pick the arm with the highest upper confidence bound.

    Exploration is *directed*: the bonus term shrinks as an arm is played, so
    attention flows to genuinely uncertain arms rather than to random ones.
    The vanilla version assumes stationarity -- the ``gamma`` discount is what
    keeps the confidence intervals from collapsing shut and blinding the
    router to a gateway that changed behaviour after a million observations.
    """

    name = "ucb1"

    def __init__(
        self, n_gateways: int, c: float = 1.0, gamma: float = 1.0, seed: int = 0
    ) -> None:
        super().__init__(n_gateways, seed)
        self.c = c
        self.counts = DiscountedCounts(n_gateways, gamma=gamma)

    def select(
        self,
        tick: int,
        context: RoutingContext | None = None,
        blocked: frozenset[int] = frozenset(),
    ) -> int:
        totals = self.counts.total
        horizon = totals.sum()
        bonus = self.c * np.sqrt(2.0 * np.log(max(horizon, 2.0)) / totals)
        scores = self.counts.mean + bonus
        if blocked:
            scores = scores.copy()
            scores[list(blocked)] = -np.inf
        best = np.flatnonzero(scores == scores.max())
        return int(self.rng.choice(best))

    def action_probabilities(
        self,
        tick: int,
        context: RoutingContext | None = None,
        blocked: frozenset[int] = frozenset(),
        n_samples: int = 128,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """One-hot: UCB1 is deterministic given its state.

        Worth stating plainly, because it has a consequence people discover too
        late: **you cannot do off-policy evaluation from a deterministic logging
        policy.** Every unchosen action has propensity zero, so importance
        weights are undefined and no amount of logged data can tell you what a
        different policy would have earned. If you intend to evaluate policies
        offline later, the policy you deploy today has to be stochastic.
        """
        totals = self.counts.total
        bonus = self.c * np.sqrt(2.0 * np.log(max(totals.sum(), 2.0)) / totals)
        scores = self.counts.mean + bonus
        if blocked:
            scores = scores.copy()
            scores[list(blocked)] = -np.inf
        best = scores == scores.max()
        return best / best.sum()

    def update(self, outcome: Outcome) -> None:
        self.counts.update(outcome.gateway, outcome.success)

    def estimated_sr(self) -> np.ndarray:
        return self.counts.mean
