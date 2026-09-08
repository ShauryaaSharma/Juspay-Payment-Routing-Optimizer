"""Router interface and the shared statistics primitive.

Every strategy in this package is a bandit over gateways. The only thing they
share is how they remember the past -- and that turns out to be the whole
ballgame under non-stationarity, which is why ``DiscountedCounts`` lives here
rather than inside any one strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..gateways import Outcome, RoutingContext  # noqa: F401  (RoutingContext used in annotations)


class DiscountedCounts:
    """Beta-style success/failure counts with geometric forgetting.

    With ``gamma == 1.0`` this is an ordinary running tally: evidence from
    100k transactions ago carries exactly as much weight as evidence from the
    last second. That is the correct thing to do in a stationary world and the
    wrong thing to do in a payments network, where a gateway that was healthy
    an hour ago tells you very little about the gateway right now.

    With ``gamma < 1.0`` every observation decays the accumulated history, so
    the counts track an exponentially-weighted recent window. The effective
    memory is roughly ``1 / (1 - gamma)`` observations per gateway.
    """

    def __init__(self, n: int, gamma: float = 1.0, prior: float = 1.0) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.n = n
        self.gamma = gamma
        self.prior = prior
        self.successes = np.full(n, prior, dtype=float)
        self.failures = np.full(n, prior, dtype=float)

    def update(self, gateway: int, success: bool) -> None:
        if self.gamma < 1.0:
            # Decay everything, then credit the observed arm. Decaying all arms
            # (not just the played one) keeps unplayed arms drifting back
            # toward the prior, which is what re-opens them for exploration
            # after a long absence.
            self.successes = self.prior + self.gamma * (self.successes - self.prior)
            self.failures = self.prior + self.gamma * (self.failures - self.prior)
        if success:
            self.successes[gateway] += 1.0
        else:
            self.failures[gateway] += 1.0

    @property
    def total(self) -> np.ndarray:
        return self.successes + self.failures

    @property
    def mean(self) -> np.ndarray:
        return self.successes / self.total


class Router(ABC):
    """Chooses a gateway per transaction and learns from the outcome."""

    name: str = "router"

    def __init__(self, n_gateways: int, seed: int = 0) -> None:
        self.n_gateways = n_gateways
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select(
        self,
        tick: int,
        context: RoutingContext | None = None,
        blocked: frozenset[int] = frozenset(),
    ) -> int:
        """Return the gateway index to route this transaction to.

        ``context`` carries per-transaction attributes (currently the issuing
        bank). Every strategy in this package ignores it -- they are flat
        bandits over gateways -- but it is threaded through so that
        `ConstrainedRouter` can act on it without a parallel call path.

        ``blocked`` is a set of gateway indices this transaction must not use,
        supplied by the constraint layer. Implementations MUST leave the
        unblocked path byte-identical, including random-number consumption:
        the benchmark numbers were measured before constraints existed and
        must stay comparable. In practice that means an early return on the
        original code path when ``blocked`` is empty.
        """

    @abstractmethod
    def update(self, outcome: Outcome) -> None:
        """Fold a realised outcome back into the router's state."""

    def estimated_sr(self) -> np.ndarray:
        """Current per-gateway SR estimate. Used for introspection only."""
        return np.full(self.n_gateways, float("nan"))

    def action_probabilities(
        self,
        tick: int,
        context: "RoutingContext | None" = None,
        blocked: frozenset[int] = frozenset(),
        n_samples: int = 128,
        rng: "np.random.Generator | None" = None,
    ) -> np.ndarray:
        """P(select each gateway) in this router's current state -- the propensity.

        Off-policy evaluation is impossible without this. To estimate what a new
        policy would have earned from logs of an old one, you need the
        probability the old policy assigned to each action it took; that is the
        importance weight's denominator.

        ``rng`` is supplied by the caller, never taken from ``self.rng``.
        Sampling from the router's own stream to compute a propensity would
        change the sequence of routing decisions -- the act of measuring would
        alter what is being measured, and every benchmark number with it.

        Implementations that sample (Thompson) estimate this by Monte Carlo;
        implementations with a closed form (epsilon-greedy, static weights)
        return it exactly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose action probabilities, so it "
            f"cannot be used as a logging policy for off-policy evaluation."
        )
