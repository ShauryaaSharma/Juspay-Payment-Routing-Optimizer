"""Turning a diagnosis into a routing decision.

This is the piece that closes the loop. The router detects a break, the agent
diagnoses it, and a confirmed finding becomes a constraint the router honours
on the next transaction: *stop sending HDFC cards to PG-Delta.*

Three properties matter more than the mechanism, because each one is a way the
loop can hurt more than it helps:

**Scoped, not blunt.** A flat bandit can only respond to "PG-Delta is bad" by
avoiding PG-Delta for everyone. But 90% of its traffic was converting fine --
only HDFC was failing. An issuer-scoped constraint removes the broken path and
keeps the rest, which is the entire reason per-transaction context was
threaded through the router.

**Never a total block.** Every constraint leaks a small `canary_rate` of
traffic to the gateway it blocks. Without it, a blocked gateway stops producing
observations, its posterior freezes, and nothing can ever justify lifting the
constraint -- the system would have made itself permanently blind, which is the
same survivorship trap the router already has to fight, now self-inflicted.

**Always expiring.** Constraints carry a TTL. Gateways recover, and a
constraint written during an incident is evidence about the past. When it
expires, traffic returns; if the problem is still there, the next
investigation re-applies it. An immortal constraint is a permanent capacity
cut based on a six-hour-old observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..gateways import RoutingContext
from .schemas import Diagnosis

DEFAULT_TTL_MINUTES = 6 * 60
DEFAULT_CANARY_RATE = 0.02
MIN_CONFIDENCE = 0.6

# Scopes that name a specific path worth constraining. `fleet_wide` names no
# culprit -- everything is degraded, so steering traffic between gateways
# cannot help -- and `no_incident` names nothing at all.
ACTIONABLE_SCOPES = frozenset({"single_gateway", "issuer_specific"})


@dataclass(frozen=True)
class RoutingConstraint:
    """One learned restriction: avoid ``gateway`` for ``issuer`` until it expires."""

    gateway: str
    issuer: str | None  # None == applies to all traffic on that gateway
    reason: str
    created_tick: int
    expires_tick: int
    confidence: float
    source_scope: str
    canary_rate: float = DEFAULT_CANARY_RATE

    def active(self, tick: int) -> bool:
        return self.created_tick <= tick < self.expires_tick

    def matches(self, issuer: str | None) -> bool:
        """Whether this constraint governs a transaction from ``issuer``."""
        if self.issuer is None:
            return True
        return issuer is not None and issuer.strip().lower() == self.issuer.strip().lower()

    def describe(self) -> str:
        target = f"{self.issuer} traffic" if self.issuer else "all traffic"
        return (
            f"avoid {self.gateway} for {target} "
            f"(ticks {self.created_tick}-{self.expires_tick}, "
            f"conf {self.confidence:.2f}, {self.canary_rate:.0%} canary)"
        )


def from_diagnosis(
    diagnosis: Diagnosis,
    tick: int,
    fleet_names: list[str],
    ttl_minutes: int = DEFAULT_TTL_MINUTES,
    min_confidence: float = MIN_CONFIDENCE,
    canary_rate: float = DEFAULT_CANARY_RATE,
) -> RoutingConstraint | None:
    """Translate a diagnosis into a constraint, or decline to.

    Declining is the common case and the important one. A diagnosis is not an
    instruction: an agent that is unsure, or that reports a fleet-wide event
    with no culprit, must not be allowed to move production traffic. Returning
    None here is the gate between "the model said something" and "the system
    did something".
    """
    if diagnosis.scope not in ACTIONABLE_SCOPES:
        return None
    if diagnosis.confidence < min_confidence:
        return None
    if not diagnosis.primary_gateway:
        return None
    if diagnosis.primary_gateway not in fleet_names:
        return None  # never act on a gateway that does not exist
    if diagnosis.scope == "issuer_specific" and not diagnosis.affected_issuer:
        return None  # issuer-scoped finding with no issuer named is incoherent

    return RoutingConstraint(
        gateway=diagnosis.primary_gateway,
        issuer=diagnosis.affected_issuer if diagnosis.scope == "issuer_specific" else None,
        reason=diagnosis.summary,
        created_tick=tick,
        expires_tick=tick + ttl_minutes,
        confidence=diagnosis.confidence,
        source_scope=diagnosis.scope,
        canary_rate=canary_rate,
    )


class ConstraintStore:
    """Holds active constraints and answers "what may this transaction use?"."""

    def __init__(self, fleet_names: list[str], seed: int = 0) -> None:
        self.fleet_names = fleet_names
        self._index = {n.lower(): i for i, n in enumerate(fleet_names)}
        self.constraints: list[RoutingConstraint] = []
        self.rng = np.random.default_rng(seed)
        self.canary_releases = 0
        self.blocked_decisions = 0

    def add(self, constraint: RoutingConstraint) -> None:
        self.constraints.append(constraint)

    def active(self, tick: int) -> list[RoutingConstraint]:
        return [c for c in self.constraints if c.active(tick)]

    def blocked(self, context: RoutingContext) -> frozenset[int]:
        """Gateway indices this transaction must avoid.

        Two safety valves, both of which can return fewer blocks than the
        constraints literally ask for:

        * the canary roll, which lets a small share of traffic through so a
          recovered gateway can prove it;
        * the all-blocked check -- if honouring every constraint would leave
          nowhere to route, none are applied. A transaction that reaches no
          gateway is a guaranteed failure, which is strictly worse than one
          routed to a suspect gateway.
        """
        blocked: set[int] = set()
        for constraint in self.active(context.tick):
            if not constraint.matches(context.issuer):
                continue
            index = self._index.get(constraint.gateway.lower())
            if index is None:
                continue
            if self.rng.random() < constraint.canary_rate:
                self.canary_releases += 1
                continue
            blocked.add(index)

        if len(blocked) >= len(self.fleet_names):
            return frozenset()
        if blocked:
            self.blocked_decisions += 1
        return frozenset(blocked)

    def stats(self) -> dict[str, Any]:
        return {
            "constraints": len(self.constraints),
            "blocked_decisions": self.blocked_decisions,
            "canary_releases": self.canary_releases,
        }


class ConstrainedRouter:
    """Wraps any router, applying learned constraints at selection time.

    Composition rather than a new strategy: the bandit keeps doing what it does
    well (tracking which gateway converts best) and the constraint layer
    supplies the one thing it structurally cannot know -- that a specific
    gateway-by-issuer path is broken, which a flat bandit averages away.
    """

    def __init__(self, inner: Any, store: ConstraintStore) -> None:
        self.inner = inner
        self.store = store

    @property
    def name(self) -> str:
        return f"constrained({getattr(self.inner, 'name', 'router')})"

    @property
    def n_gateways(self) -> int:
        return self.inner.n_gateways

    def select(
        self,
        tick: int,
        context: RoutingContext | None = None,
        blocked: frozenset[int] = frozenset(),
    ) -> int:
        context = context or RoutingContext(tick=tick)
        learned = self.store.blocked(context)
        return self.inner.select(tick, context, blocked | learned)

    def update(self, outcome: Any) -> None:
        # Outcomes from canary traffic flow into the posterior exactly like any
        # other. That is what lets the bandit notice recovery on its own.
        self.inner.update(outcome)

    def estimated_sr(self):
        return self.inner.estimated_sr()
