"""Shared state: routing constraints and bandit posteriors that survive a restart.

The README lists two limitations under what to build next: constraints and
posteriors live in process, so a restart discards everything the router learned
and every constraint the agent installed. This closes both, and the *way* it
closes them is the interesting part, because the two have opposite requirements.

## Constraints belong in Redis

A constraint is low-volume, must be visible to every replica within seconds,
and has a natural TTL. That is a description of a Redis key. `SETEX` gives the
expiry for free, so a constraint that outlives its usefulness disappears
without a sweeper process, and `SCAN` over a prefix lists what is in force.

## Posteriors do not

A posterior is read on *every* transaction. At the volumes this is aimed at, a
network round-trip per routing decision is not a cache, it is the bottleneck --
a 0.2ms Redis GET against a decision budget measured in milliseconds, on every
payment, is the whole budget.

So posteriors stay in process and are *snapshotted* to Redis periodically. A
restart reloads the last snapshot instead of cold-starting, which is the
property that actually matters; losing the last few seconds of counts is
irrelevant to a Beta posterior with thousands of observations behind it.

That is a write-back cache, and it is the same shape as the in-house KvDB
Juspay describes for exactly this problem.

## Everything degrades

Without `redis` installed or reachable, both stores fall back to memory and the
service behaves exactly as it did before. A shared-state outage should cost you
persistence, not availability.
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.agent.constraints import ConstraintStore, RoutingConstraint

CONSTRAINT_PREFIX = "route:constraint:"
POSTERIOR_KEY = "route:posterior:snapshot"


def connect(url: str | None, timeout: float = 2.0) -> Any | None:
    """Return a Redis client, or None if unavailable.

    Never raises. A missing driver, a wrong URL, or a down server all produce
    None and a fallback to in-process state.
    """
    if not url:
        return None
    try:
        import redis
    except ImportError:
        return None
    try:
        client = redis.Redis.from_url(
            url, socket_connect_timeout=timeout, socket_timeout=timeout,
            decode_responses=True,
        )
        client.ping()
        return client
    except Exception:
        return None


class RedisConstraintStore(ConstraintStore):
    """ConstraintStore whose constraints are shared across replicas.

    Subclasses rather than replaces the in-process store, so the canary roll,
    the all-blocked safety check and the issuer scoping are inherited unchanged
    -- the behaviour that `tests/test_constraints.py` pins stays pinned.
    """

    def __init__(self, fleet_names: list[str], client: Any, seed: int = 0,
                 refresh_seconds: float = 1.0) -> None:
        super().__init__(fleet_names, seed=seed)
        self.client = client
        self.refresh_seconds = refresh_seconds
        self._fetched_at = 0.0
        self.degraded = False

    def add(self, constraint: RoutingConstraint) -> None:
        super().add(constraint)
        ttl = max(1, constraint.expires_tick - constraint.created_tick)
        payload = json.dumps({
            "gateway": constraint.gateway, "issuer": constraint.issuer,
            "reason": constraint.reason, "created_tick": constraint.created_tick,
            "expires_tick": constraint.expires_tick, "confidence": constraint.confidence,
            "source_scope": constraint.source_scope, "canary_rate": constraint.canary_rate,
        })
        key = f"{CONSTRAINT_PREFIX}{constraint.gateway}:{constraint.issuer or 'ALL'}"
        try:
            # TTL in seconds mirrors the constraint's own lifetime in simulated
            # minutes; in a real deployment the two clocks are the same clock.
            self.client.setex(key, ttl, payload)
            self.degraded = False
        except Exception:
            self.degraded = True  # keep serving from the in-process copy

    def _refresh(self, tick: int) -> None:
        """Pull peers' constraints in, at most once per refresh window."""
        now = time.monotonic()
        if now - self._fetched_at < self.refresh_seconds:
            return
        self._fetched_at = now
        try:
            keys = list(self.client.scan_iter(match=f"{CONSTRAINT_PREFIX}*", count=100))
            payloads = [self.client.get(k) for k in keys] if keys else []
            self.degraded = False
        except Exception:
            self.degraded = True
            return

        # Drop what has already expired before merging. `active()` filters
        # expired constraints out of its *result* but nothing removed them from
        # the list, so a long-running replica accumulated every constraint any
        # peer ever wrote. Pruning needs the current tick -- an earlier version
        # of this compared against max(expires_tick), which kept only the
        # longest-lived constraint and silently discarded every other live one.
        self.constraints = [c for c in self.constraints if c.expires_tick > tick]

        known = {(c.gateway, c.issuer) for c in self.constraints}
        for raw in payloads:
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if (data.get("gateway"), data.get("issuer")) in known:
                continue
            self.constraints.append(RoutingConstraint(**data))

    def active(self, tick: int) -> list[RoutingConstraint]:
        self._refresh(tick)
        return super().active(tick)


class PosteriorSnapshot:
    """Periodic write-back of a router's learned counts.

    Deliberately not write-through. The posterior is read on the hot path and
    written on every outcome; sending that to Redis synchronously would put a
    network round-trip inside the routing decision. Snapshotting trades a few
    seconds of counts -- immaterial against thousands of observations -- for
    keeping the decision path in-process.
    """

    def __init__(self, client: Any | None, key: str = POSTERIOR_KEY,
                 interval_seconds: float = 5.0) -> None:
        self.client = client
        self.key = key
        self.interval_seconds = interval_seconds
        self._last_write = 0.0
        self.writes = 0
        self.failures = 0

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def maybe_save(self, router: Any, force: bool = False) -> bool:
        if not self.enabled:
            return False
        now = time.monotonic()
        if not force and now - self._last_write < self.interval_seconds:
            return False
        counts = getattr(router, "counts", None) or getattr(router, "global_counts", None)
        if counts is None:
            return False
        self._last_write = now
        payload = json.dumps({
            "successes": list(map(float, counts.successes)),
            "failures": list(map(float, counts.failures)),
            "saved_at": time.time(),
        })
        try:
            self.client.set(self.key, payload)
            self.writes += 1
            return True
        except Exception:
            self.failures += 1
            return False

    def restore(self, router: Any) -> bool:
        """Warm a cold router from the last snapshot. False if there is none."""
        if not self.enabled:
            return False
        try:
            raw = self.client.get(self.key)
        except Exception:
            self.failures += 1
            return False
        if not raw:
            return False
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return False
        counts = getattr(router, "counts", None) or getattr(router, "global_counts", None)
        if counts is None:
            return False
        successes, failures = data.get("successes"), data.get("failures")
        if not successes or len(successes) != counts.successes.size:
            return False  # fleet changed shape since the snapshot; ignore it
        counts.successes[:] = successes
        counts.failures[:] = failures
        return True

    def stats(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "writes": self.writes, "failures": self.failures}
