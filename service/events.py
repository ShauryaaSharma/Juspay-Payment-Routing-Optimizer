"""Where routing outcomes go after the decision: Kafka, then ClickHouse.

Every routed transaction produces an event carrying the context, the gateway
chosen, the outcome, and -- critically -- **the propensity the router assigned
to that choice**. That last field is the one nobody logs until it is too late:
without it, `src/ope.py` cannot run at all, and the logs are only good for
measuring the policy that produced them.

Two sinks, because they answer different questions:

* **Kafka** is the durable fan-out. Posterior replication to other replicas,
  the analytics load, and anything added later all read the same topic without
  the router knowing about them.
* **ClickHouse** is the OLAP store the off-policy evaluation queries. Bandit
  logs are append-only, enormous, and queried by time range and grouped by
  gateway -- a columnar store is the right shape, and doing this in Postgres
  alongside the trace store would put an analytics scan next to the write path.

The ClickHouse client is plain HTTP through urllib rather than
`clickhouse-connect`. Its HTTP interface takes `INSERT ... FORMAT JSONEachRow`
as a POST body, which is about fifteen lines and keeps the dependency count at
zero for the sink.

**Neither sink can fail a payment.** Both are wrapped, both count failures, and
both degrade to dropping events. A routing decision that succeeded but could
not be logged is a lost row; a routing decision blocked on a broker is a lost
payment.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

from .metrics import events_published_total

CLICKHOUSE_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table} (
    ts              DateTime64(3),
    transaction_id  String,
    tick            UInt32,
    issuer          LowCardinality(String),
    gateway         LowCardinality(String),
    success         UInt8,
    latency_ms      Float32,
    propensity      Float32,
    policy          LowCardinality(String),
    constrained     UInt8
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (gateway, issuer, ts)
"""


@dataclass
class RoutingEvent:
    """One routed transaction, in the shape off-policy evaluation needs."""

    transaction_id: str
    tick: int
    issuer: str
    gateway: str
    success: bool
    latency_ms: float
    # P(this gateway | this context) under the policy that made the choice.
    # Without it the logs cannot support importance weighting.
    propensity: float
    policy: str
    constrained: bool = False
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["success"] = int(self.success)
        data["constrained"] = int(self.constrained)
        # ClickHouse DateTime64(3) accepts 'YYYY-MM-DD HH:MM:SS.mmm'.
        data["ts"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.ts)) + \
            f".{int((self.ts % 1) * 1000):03d}"
        return data


class KafkaSink:
    """Fire-and-forget producer. Never blocks a routing decision."""

    name = "kafka"

    def __init__(self, brokers: str | None, topic: str = "routing.outcomes") -> None:
        self.topic = topic
        self.producer = None
        self.published = 0
        self.failures = 0
        self.last_error: str | None = None
        if not brokers:
            return
        try:
            from confluent_kafka import Producer
        except ImportError:
            self.last_error = "confluent-kafka not installed"
            return
        try:
            self.producer = Producer({
                "bootstrap.servers": brokers,
                # Bound how long a broker outage can hold memory, and never let
                # the send path block the caller.
                "queue.buffering.max.messages": 100_000,
                "linger.ms": 20,
                "socket.timeout.ms": 3000,
                "message.timeout.ms": 5000,
            })
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"

    @property
    def enabled(self) -> bool:
        return self.producer is not None

    def publish(self, event: RoutingEvent) -> bool:
        if not self.enabled:
            return False
        try:
            self.producer.produce(
                self.topic,
                key=event.issuer.encode(),  # same issuer -> same partition
                value=json.dumps(event.to_dict()).encode(),
            )
            self.producer.poll(0)  # serve delivery callbacks without blocking
            self.published += 1
            events_published_total.inc(sink="kafka", result="ok")
            return True
        except Exception as exc:
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            events_published_total.inc(sink="kafka", result="error")
            return False

    def flush(self, timeout: float = 5.0) -> None:
        if self.enabled:
            try:
                self.producer.flush(timeout)
            except Exception:
                pass

    def stats(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "topic": self.topic,
                "published": self.published, "failures": self.failures,
                "last_error": self.last_error}


class ClickHouseSink:
    """Batched inserts over the HTTP interface.

    Batching is not an optimisation here, it is the only workable mode:
    ClickHouse wants large infrequent inserts and degrades badly under a row at
    a time, since every insert creates a part that later has to be merged.
    """

    name = "clickhouse"

    def __init__(self, url: str | None, table: str = "routing_outcomes",
                 batch_size: int = 500, timeout: float = 3.0) -> None:
        self.url = url.rstrip("/") if url else None
        self.table = table
        self.batch_size = batch_size
        self.timeout = timeout
        self.buffer: list[dict[str, Any]] = []
        self.inserted = 0
        self.failures = 0
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return self.url is not None

    def _post(self, query: str, body: bytes = b"") -> bool:
        request = urllib.request.Request(
            f"{self.url}/?query={urllib.parse.quote(query)}",
            data=body, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def ensure_schema(self) -> bool:
        if not self.enabled:
            return False
        return self._post(CLICKHOUSE_SCHEMA.format(table=self.table))

    def publish(self, event: RoutingEvent) -> bool:
        if not self.enabled:
            return False
        self.buffer.append(event.to_dict())
        if len(self.buffer) >= self.batch_size:
            return self.flush()
        return True

    def flush(self) -> bool:
        if not self.enabled or not self.buffer:
            return True
        rows = self.buffer
        self.buffer = []
        body = "\n".join(json.dumps(r) for r in rows).encode()
        ok = self._post(f"INSERT INTO {self.table} FORMAT JSONEachRow", body)
        if ok:
            self.inserted += len(rows)
            events_published_total.inc(len(rows), sink="clickhouse", result="ok")
        else:
            # Dropped on purpose rather than retried in the request path: a
            # lost analytics row is cheaper than a stalled payment.
            self.failures += len(rows)
            events_published_total.inc(len(rows), sink="clickhouse", result="error")
        return ok

    def stats(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "table": self.table,
                "inserted": self.inserted, "failures": self.failures,
                "buffered": len(self.buffer), "last_error": self.last_error}


class EventPipeline:
    """Both sinks behind one call, so the router only knows it emitted an event."""

    def __init__(self, kafka: KafkaSink | None = None,
                 clickhouse: ClickHouseSink | None = None) -> None:
        self.kafka = kafka or KafkaSink(None)
        self.clickhouse = clickhouse or ClickHouseSink(None)

    def publish(self, event: RoutingEvent) -> None:
        self.kafka.publish(event)
        self.clickhouse.publish(event)

    def flush(self) -> None:
        self.kafka.flush()
        self.clickhouse.flush()

    def stats(self) -> dict[str, Any]:
        return {"kafka": self.kafka.stats(), "clickhouse": self.clickhouse.stats()}
