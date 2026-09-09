"""Contextual Thompson sampling with partial pooling across issuers.

This is the answer to the obvious objection to the whole investigation-agent
layer: *why build an agent to discover that PG-Delta is declining HDFC cards,
when a bandit conditioned on the issuer would learn it on its own?*

It would. The question is how fast, and that is worth measuring rather than
arguing about -- see `contextual_vs_agent.py`.

## The cost of context

A flat bandit over 5 gateways tracks 5 posteriors. Conditioning on 5 issuers
tracks 25, so every cell sees roughly a fifth of the traffic. Statistical power
per cell drops with it, and a naive contextual bandit is therefore *worse* than
a flat one until volume catches up -- it has fragmented its evidence in
exchange for resolution it cannot yet support.

## Partial pooling

The fix is not to choose between the two. Each cell's posterior is shrunk
toward its gateway's pooled posterior:

    alpha = k * m_g + successes[g, i]
    beta  = k * (1 - m_g) + failures[g, i]

where ``m_g`` is gateway g's success rate across all issuers and ``k`` is a
prior strength in pseudo-observations. A cell with no data of its own inherits
the gateway's overall behaviour; as it accumulates evidence, that evidence
takes over.

``k`` scales how hard each cell is pulled toward its gateway's pooled
behaviour:

* ``k = 0``    -- pure contextual. Maximum resolution, worst cold start.
* larger ``k`` -- progressively stronger shrinkage; issuers look more alike.

It does **not** interpolate all the way to a flat bandit, and that is worth
being precise about. The prior strength is capped by the pooled evidence that
exists (see `_posterior`), so raising ``k`` past that ceiling changes nothing,
while a cell's own evidence keeps accumulating and eventually rivals the prior.
Measured at k=100000 on a 20k-transaction run, the broken cell still separates
from its neighbours by ~6 percentage points rather than collapsing to zero
difference. The cap is what keeps unplayed gateways explorable, so this is a
deliberate trade: bounded shrinkage in exchange for a router that does not lock
onto the first gateway it tries.
"""

from __future__ import annotations

import numpy as np

from ..gateways import ISSUERS, Outcome, RoutingContext
from .base import DiscountedCounts, Router

DEFAULT_POOLING = 40.0


class ContextualThompsonRouter(Router):
    """Thompson sampling over (gateway, issuer) cells, pooled per gateway."""

    name = "contextual-thompson"

    def __init__(
        self,
        n_gateways: int,
        issuers: tuple[str, ...] = ISSUERS,
        gamma: float = 0.999,
        pooling: float = DEFAULT_POOLING,
        seed: int = 0,
    ) -> None:
        super().__init__(n_gateways, seed)
        if pooling < 0:
            raise ValueError("pooling must be >= 0")
        self.issuers = tuple(issuers)
        self.pooling = float(pooling)
        self._issuer_index = {name.lower(): i for i, name in enumerate(self.issuers)}

        # Pooled, per-gateway: the fallback when a cell is thin or the context
        # is missing entirely.
        self.global_counts = DiscountedCounts(n_gateways, gamma=gamma)
        # One DiscountedCounts per issuer, each over all gateways. Storing it
        # this way keeps the per-issuer decay independent, which matters: an
        # issuer with little traffic should not have its evidence aged out by
        # transactions belonging to a busier one.
        self.cell_counts = [
            DiscountedCounts(n_gateways, gamma=gamma, prior=0.0)
            for _ in self.issuers
        ]
        self.updates_without_context = 0

    # -- helpers -----------------------------------------------------------

    def _issuer_slot(self, issuer: str | None) -> int | None:
        if issuer is None:
            return None
        return self._issuer_index.get(str(issuer).strip().lower())

    def _posterior(self, slot: int | None) -> tuple[np.ndarray, np.ndarray]:
        """Shrunk Beta parameters for every gateway under one issuer.

        The prior strength is capped by the pooled evidence that actually
        exists: ``min(pooling, global_total[g])``. Without that cap, a gateway
        nobody has tried yet gets a full-strength prior at its uninformative
        0.5 mean -- Beta(20, 20) at pooling=40, which is *confidently* mediocre
        and can never out-sample an arm with real evidence behind it. The
        result is a router that locks onto whichever gateway it happened to
        try first and never explores again. Capping by pooled evidence makes an
        unplayed gateway's prior Beta(1, 1), i.e. uniform, which explores.

        The principle: never be more certain about a cell than the data backing
        its shrinkage target supports.
        """
        pooled_mean = self.global_counts.mean  # per gateway, in (0, 1)
        strength = np.minimum(self.pooling, self.global_counts.total)
        alpha = strength * pooled_mean
        beta = strength * (1.0 - pooled_mean)
        if slot is not None:
            counts = self.cell_counts[slot]
            alpha = alpha + counts.successes
            beta = beta + counts.failures
        # Beta requires strictly positive parameters; with pooling=0 and an
        # empty cell both terms are zero, which would raise.
        return np.maximum(alpha, 1e-3), np.maximum(beta, 1e-3)

    # -- Router interface --------------------------------------------------

    def select(
        self,
        tick: int,
        context: RoutingContext | None = None,
        blocked: frozenset[int] = frozenset(),
    ) -> int:
        slot = self._issuer_slot(context.issuer if context else None)
        alpha, beta = self._posterior(slot)
        samples = self.rng.beta(alpha, beta)
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
        rng = rng if rng is not None else np.random.default_rng(0)
        alpha, beta = self._posterior(self._issuer_slot(context.issuer if context else None))
        draws = rng.beta(alpha, beta, size=(n_samples, self.n_gateways))
        if blocked:
            draws[:, list(blocked)] = -np.inf
        winners = np.argmax(draws, axis=1)
        return np.bincount(winners, minlength=self.n_gateways) / n_samples

    def decide(
        self,
        tick: int,
        context: RoutingContext | None = None,
        blocked: frozenset[int] = frozenset(),
        n_samples: int = 64,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, np.ndarray]:
        """One draw serves both decision and propensity; see ThompsonRouter."""
        rng = rng if rng is not None else np.random.default_rng(0)
        alpha, beta = self._posterior(self._issuer_slot(context.issuer if context else None))
        draws = rng.beta(alpha, beta, size=(n_samples + 1, self.n_gateways))
        if blocked:
            draws[:, list(blocked)] = -np.inf
        gateway = int(np.argmax(draws[0]))
        winners = np.argmax(draws[1:], axis=1)
        return gateway, np.bincount(winners, minlength=self.n_gateways) / n_samples

    def update(self, outcome: Outcome) -> None:
        # Every outcome updates the pooled posterior, so the shrinkage target
        # stays current even for issuers that are rarely seen.
        self.global_counts.update(outcome.gateway, outcome.success)
        slot = self._issuer_slot(outcome.issuer)
        if slot is None:
            # Counted rather than ignored: a run where this is large means the
            # simulator is not supplying issuer context and the router is
            # quietly behaving like a flat bandit.
            self.updates_without_context += 1
            return
        self.cell_counts[slot].update(outcome.gateway, outcome.success)

    def estimated_sr(self) -> np.ndarray:
        return self.global_counts.mean

    def estimated_sr_for(self, issuer: str | None) -> np.ndarray:
        """Per-gateway posterior mean for one issuer. Introspection only."""
        alpha, beta = self._posterior(self._issuer_slot(issuer))
        return alpha / (alpha + beta)

    def cell_observations(self, issuer: str | None = None) -> np.ndarray:
        """How much evidence each cell actually holds -- the cost of context."""
        slot = self._issuer_slot(issuer)
        if slot is None:
            return self.global_counts.total
        counts = self.cell_counts[slot]
        return counts.successes + counts.failures
