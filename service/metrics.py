"""Prometheus metrics, written directly against the text exposition format.

`prometheus_client` is the obvious dependency here and it was not taken, for
the same reason matplotlib was not taken for the charts: the exposition format
is a documented, stable, sixty-line contract, and this project's claim that it
runs on numpy alone is worth more than the sixty lines saved.

What is exported is chosen to answer the questions an on-call engineer actually
has about a router:

* `routing_decision_duration_seconds` -- the hot-path histogram. The JD-relevant
  number: a routing decision has a latency budget measured in milliseconds, so
  it needs percentiles, not an average.
* `routing_decisions_total{gateway,outcome}` -- realised success rate, sliced by
  gateway, derivable in PromQL as a rate ratio.
* `routing_constraints_active` -- how many learned constraints are in force.
  A number that climbs and never falls means TTLs are not expiring.
* `agent_investigations_total{stop_reason}` -- how investigations end. Anything
  other than `completed` is a loop that gave up.

Histogram buckets are in seconds and deliberately dense below 10ms, because
that is the region a routing decision must live in and a default bucket set
would put every observation in the first bucket and tell you nothing.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Iterable

# Seconds. Dense at the bottom: sub-millisecond is the interesting range.
DECISION_BUCKETS: tuple[float, ...] = (
    0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.001,
    0.0025, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0,
)
INVESTIGATION_BUCKETS: tuple[float, ...] = (0.5, 1, 2, 5, 10, 30, 60, 120, 300)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(pairs: tuple[tuple[str, str], ...]) -> str:
    if not pairs:
        return ""
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in pairs)
    return "{" + inner + "}"


class Counter:
    """Monotonic count, optionally sliced by labels."""

    kind = "counter"

    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help_text = help_text
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] += amount

    def samples(self) -> Iterable[str]:
        with self._lock:
            items = list(self._values.items())
        for key, value in sorted(items):
            yield f"{self.name}{_labels(key)} {value:g}"


class Gauge:
    """A value that goes up and down."""

    kind = "gauge"

    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help_text = help_text
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = float(value)

    def samples(self) -> Iterable[str]:
        with self._lock:
            items = list(self._values.items())
        for key, value in sorted(items):
            yield f"{self.name}{_labels(key)} {value:g}"


class Histogram:
    """Cumulative bucket counts, as Prometheus expects them.

    Buckets are cumulative (`le` = less than or equal), which is the part
    people implementing this by hand usually get wrong: each bucket counts
    every observation at or below its bound, not only those inside it.
    """

    kind = "histogram"

    def __init__(self, name: str, help_text: str, buckets: tuple[float, ...]) -> None:
        self.name = name
        self.help_text = help_text
        self.buckets = tuple(sorted(buckets))
        self._counts: dict[tuple[tuple[str, str], ...], list[int]] = {}
        self._sum: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._total: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            counts = self._counts.setdefault(key, [0] * len(self.buckets))
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    counts[i] += 1
            self._sum[key] += value
            self._total[key] += 1

    def time(self, **labels: str) -> "_Timer":
        return _Timer(self, labels)

    def samples(self) -> Iterable[str]:
        with self._lock:
            keys = sorted(self._counts)
            snapshot = {k: (list(self._counts[k]), self._sum[k], self._total[k]) for k in keys}
        for key in keys:
            counts, total_sum, count = snapshot[key]
            for bound, cumulative in zip(self.buckets, counts):
                labels = key + (("le", repr(bound)),)
                yield f"{self.name}_bucket{_labels(labels)} {cumulative}"
            yield f"{self.name}_bucket{_labels(key + (('le', '+Inf'),))} {count}"
            yield f"{self.name}_sum{_labels(key)} {total_sum:g}"
            yield f"{self.name}_count{_labels(key)} {count}"


class _Timer:
    def __init__(self, histogram: Histogram, labels: dict[str, str]) -> None:
        self.histogram = histogram
        self.labels = labels

    def __enter__(self) -> "_Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.histogram.observe(time.perf_counter() - self.started, **self.labels)


class Registry:
    def __init__(self) -> None:
        self.metrics: list[Counter | Gauge | Histogram] = []

    def register(self, metric):
        self.metrics.append(metric)
        return metric

    def render(self) -> str:
        """Prometheus text exposition format, version 0.0.4."""
        lines: list[str] = []
        for metric in self.metrics:
            lines.append(f"# HELP {metric.name} {metric.help_text}")
            lines.append(f"# TYPE {metric.name} {metric.kind}")
            lines.extend(metric.samples())
        return "\n".join(lines) + "\n"


REGISTRY = Registry()

decision_duration = REGISTRY.register(Histogram(
    "routing_decision_duration_seconds",
    "Time to choose a gateway for one transaction.",
    DECISION_BUCKETS,
))
decisions_total = REGISTRY.register(Counter(
    "routing_decisions_total",
    "Routing decisions, by gateway and realised outcome.",
))
constraints_active = REGISTRY.register(Gauge(
    "routing_constraints_active",
    "Learned routing constraints currently in force.",
))
constraint_blocks_total = REGISTRY.register(Counter(
    "routing_constraint_blocks_total",
    "Decisions where a learned constraint removed at least one gateway.",
))
canary_releases_total = REGISTRY.register(Counter(
    "routing_canary_releases_total",
    "Decisions where a constraint deliberately let traffic through to keep the "
    "blocked gateway observable.",
))
investigations_total = REGISTRY.register(Counter(
    "agent_investigations_total",
    "Completed investigations, by stop reason.",
))
investigation_duration = REGISTRY.register(Histogram(
    "agent_investigation_duration_seconds",
    "Wall-clock time for one investigation.",
    INVESTIGATION_BUCKETS,
))
investigation_cost = REGISTRY.register(Counter(
    "agent_investigation_cost_usd_total",
    "Cumulative model spend on investigations.",
))
events_published_total = REGISTRY.register(Counter(
    "events_published_total",
    "Outcome events published downstream, by sink and result.",
))


def render() -> str:
    return REGISTRY.render()
