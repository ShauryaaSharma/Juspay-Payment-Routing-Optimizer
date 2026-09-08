"""Non-stationary payment gateway simulator.

The point of this module is to make the environment *hard* in the way real
payment infrastructure is hard: success rates drift on a diurnal cycle, and
gateways suffer sudden partial or total outages that recover later. A router
that assumes a stationary world will look great for the first few thousand
transactions and then quietly bleed conversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TICKS_PER_DAY = 24 * 60  # one tick == one simulated minute


ISSUERS: tuple[str, ...] = ("HDFC", "ICICI", "SBI", "AXIS", "KOTAK")
ISSUER_MIX: tuple[float, ...] = (0.31, 0.26, 0.20, 0.14, 0.09)


@dataclass(frozen=True)
class DegradationEvent:
    """A window during which a gateway is partially or fully degraded.

    ``issuer`` scopes the degradation to transactions from one issuing bank.
    This is the failure mode a fleet-level success rate cannot see: a gateway
    whose aggregate SR looks merely soft while it is declining *every* card
    from one bank. Diagnosing it requires segmenting, which is precisely what
    the investigation agent exists to do.
    """

    start: int
    end: int
    sr_multiplier: float
    latency_multiplier: float = 1.0
    label: str = ""
    issuer: str | None = None

    def active(self, t: int, issuer: str | None = None) -> bool:
        if not (self.start <= t < self.end):
            return False
        if self.issuer is None:
            return True
        # An issuer-scoped event applies only to that issuer's traffic. When
        # the caller asks without naming an issuer (the fleet-level view), the
        # event is treated as inactive -- it is invisible in the aggregate,
        # which is the whole point.
        return issuer == self.issuer


@dataclass
class GatewaySpec:
    """Static configuration for one simulated gateway."""

    name: str
    base_sr: float
    base_latency_ms: float
    latency_jitter_ms: float = 40.0
    diurnal_amplitude: float = 0.0
    diurnal_phase: float = 0.0
    cost_bps: float = 0.0
    events: list[DegradationEvent] = field(default_factory=list)

    def true_sr(self, t: int, issuer: str | None = None) -> float:
        """Ground-truth success probability at tick ``t``.

        Never visible to a router -- only used to generate outcomes and to
        compute regret against an omniscient oracle. Pass ``issuer`` to include
        issuer-scoped degradations; omit it for the fleet-level view.
        """
        phase = 2.0 * math.pi * t / TICKS_PER_DAY + self.diurnal_phase
        sr = self.base_sr * (1.0 + self.diurnal_amplitude * math.sin(phase))
        for event in self.events:
            if event.active(t, issuer):
                sr *= event.sr_multiplier
        return float(np.clip(sr, 0.001, 0.999))

    def true_latency_ms(self, t: int, issuer: str | None = None) -> float:
        latency = self.base_latency_ms
        for event in self.events:
            if event.active(t, issuer):
                latency *= event.latency_multiplier
        return latency


@dataclass(frozen=True)
class RoutingContext:
    """What the router knows about a transaction before it routes it.

    A flat bandit ignores this entirely. It exists so that a constraint learned
    from an investigation -- "PG-Delta is declining HDFC cards" -- has something
    to match against at selection time. Without per-transaction context, a
    finding about one issuer can only be acted on by blocking a gateway for
    *everyone*, which throws away the traffic that was converting perfectly.
    """

    tick: int
    issuer: str | None = None


@dataclass(frozen=True)
class Outcome:
    """Result of routing a single transaction.

    ``issuer`` is optional and defaults to None. A flat bandit ignores it; a
    contextual one needs it to credit the right gateway-by-issuer cell. Left
    optional so every existing construction site stays valid -- a contextual
    router receiving ``None`` falls back to its pooled per-gateway posterior
    rather than silently mis-crediting.
    """

    gateway: int
    success: bool
    latency_ms: float
    tick: int
    issuer: str | None = None


class GatewayEnvironment:
    """Holds the gateway fleet and turns routing decisions into outcomes."""

    def __init__(self, specs: list[GatewaySpec], seed: int = 0) -> None:
        if not specs:
            raise ValueError("need at least one gateway")
        self.specs = specs
        self.rng = np.random.default_rng(seed)

    @property
    def n_gateways(self) -> int:
        return len(self.specs)

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def attempt(self, gateway: int, tick: int) -> Outcome:
        spec = self.specs[gateway]
        sr = spec.true_sr(tick)
        success = bool(self.rng.random() < sr)
        latency = max(
            1.0,
            self.rng.normal(spec.true_latency_ms(tick), spec.latency_jitter_ms),
        )
        return Outcome(gateway=gateway, success=success, latency_ms=latency, tick=tick)

    def oracle_sr(self, tick: int) -> float:
        """Best success rate achievable at this tick by a clairvoyant router."""
        return max(spec.true_sr(tick) for spec in self.specs)

    def best_gateway(self, tick: int) -> int:
        return max(range(self.n_gateways), key=lambda i: self.specs[i].true_sr(tick))

    def sr_matrix(self, horizon: int) -> np.ndarray:
        """(n_gateways, horizon) array of ground-truth SRs, for plotting."""
        return np.array(
            [[spec.true_sr(t) for t in range(horizon)] for spec in self.specs]
        )


def default_fleet() -> list[GatewaySpec]:
    """A five-gateway fleet with the failure modes worth defending against.

    * ``PG-Alpha``  - the reliable default. Boring, slightly expensive.
    * ``PG-Bravo``  - marginally better than Alpha until it dies overnight.
    * ``PG-Charlie``- strong diurnal swing; great by day, poor at night.
    * ``PG-Delta``  - cheap, mediocre, and occasionally brownouts.
    * ``PG-Echo``   - a new gateway that is genuinely the best, but only
                      after a rocky warm-up period.
    """
    day = TICKS_PER_DAY
    return [
        GatewaySpec(
            name="PG-Alpha",
            base_sr=0.92,
            base_latency_ms=240.0,
            diurnal_amplitude=0.01,
            cost_bps=18.0,
        ),
        GatewaySpec(
            name="PG-Bravo",
            base_sr=0.94,
            base_latency_ms=210.0,
            diurnal_amplitude=0.01,
            cost_bps=22.0,
            events=[
                # Hard outage on night two: the classic 2AM pager event.
                DegradationEvent(
                    start=int(1.9 * day),
                    end=int(2.3 * day),
                    sr_multiplier=0.05,
                    latency_multiplier=4.0,
                    label="Bravo hard outage",
                ),
            ],
        ),
        GatewaySpec(
            name="PG-Charlie",
            base_sr=0.90,
            base_latency_ms=180.0,
            diurnal_amplitude=0.08,
            diurnal_phase=-math.pi / 2,
            cost_bps=15.0,
        ),
        GatewaySpec(
            name="PG-Delta",
            base_sr=0.86,
            base_latency_ms=150.0,
            diurnal_amplitude=0.02,
            cost_bps=9.0,
            events=[
                DegradationEvent(
                    start=int(3.4 * day),
                    end=int(3.7 * day),
                    sr_multiplier=0.55,
                    latency_multiplier=2.0,
                    label="Delta brownout",
                ),
            ],
        ),
        GatewaySpec(
            name="PG-Echo",
            base_sr=0.96,
            base_latency_ms=200.0,
            diurnal_amplitude=0.01,
            cost_bps=20.0,
            events=[
                # Warm-up: Echo is the best gateway, but punishes any router
                # that judges it on day-one evidence and never revisits.
                DegradationEvent(
                    start=0,
                    end=int(1.0 * day),
                    sr_multiplier=0.72,
                    label="Echo warm-up",
                ),
            ],
        ),
    ]
