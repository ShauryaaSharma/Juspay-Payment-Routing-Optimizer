"""Memory across investigations.

Two tiers, because they answer different questions:

* **Episodic** -- what happened in one past incident. "On day 2, PG-Bravo
  collapsed to 4% and the router pulled away within 9 minutes."
* **Semantic** -- what repetition has taught us. "PG-Bravo has now degraded
  three times, always overnight." No single episode contains that; it only
  exists once several are compared, and `consolidate()` is what derives it.

Retrieval is lexical and dependency-free by design. An embedding model would
retrieve better, but it would also make memory quality depend on a second
network service, and the point of this layer is to measure whether *memory*
helps -- not whether one embedding model beats another. Swapping in vectors
later changes `_score` and nothing else.

Memories are surfaced to the model as explicitly untrusted hypotheses (see
`prompts.render_incident_brief`). A memory is a claim written by a previous
run of a fallible agent; treating it as established fact is how one early
wrong diagnosis becomes permanent.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

_WORD = re.compile(r"[a-z0-9\-]+")
_STOPWORDS = frozenset(
    "the a an and or of to in on at is was for with by from this that it as be are "
    "has have had its than then when while but not no".split()
)


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(str(text).lower()) if w not in _STOPWORDS and len(w) > 1]


@dataclass
class Memory:
    """One remembered item. ``kind`` is 'episodic' or 'semantic'."""

    kind: str
    text: str
    gateway: str | None = None
    issuer: str | None = None
    scope: str | None = None
    day: int | None = None
    seq: int = 0  # insertion order, used for recency
    support: int = 1  # how many episodes back a semantic memory
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryStore:
    """Append-only episodic log plus a derived semantic layer."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.memories: list[Memory] = []
        self._seq = 0
        if path and os.path.exists(path):
            self.load()

    # -- writing -----------------------------------------------------------

    def remember_investigation(
        self,
        diagnosis: Any,
        window: tuple[int, int],
        verified: bool | None = None,
    ) -> Memory:
        """Record one completed investigation as an episodic memory.

        ``verified`` carries whether the diagnosis was later confirmed. In this
        project the eval harness knows ground truth and can set it; in
        production it would come from the incident's resolution. Unverified
        memories are still stored -- and still marked -- because an agent that
        only remembers its confirmed wins learns nothing from its mistakes.
        """
        day = window[0] // (24 * 60) + 1
        mark = {True: "confirmed", False: "later found incorrect", None: "unverified"}[verified]
        target = diagnosis.primary_gateway or "the fleet"
        detail = f" affecting {diagnosis.affected_issuer} traffic" if diagnosis.affected_issuer else ""
        text = (
            f"On day {day}, diagnosed {diagnosis.scope} on {target}{detail} "
            f"(confidence {diagnosis.confidence:.2f}, {mark}). {diagnosis.summary}"
        )
        self._seq += 1
        memory = Memory(
            kind="episodic",
            text=text,
            gateway=diagnosis.primary_gateway,
            issuer=diagnosis.affected_issuer,
            scope=diagnosis.scope,
            day=day,
            seq=self._seq,
            tags=_tokenize(text),
        )
        self.memories.append(memory)
        return memory

    def consolidate(self, min_support: int = 2) -> list[Memory]:
        """Promote repeated episodic patterns into semantic facts.

        Rebuilt from scratch each call rather than incrementally updated, so a
        semantic memory can never outlive the episodes that justified it.
        """
        self.memories = [m for m in self.memories if m.kind != "semantic"]
        groups: dict[tuple[str | None, str | None, str | None], list[Memory]] = defaultdict(list)
        for memory in self.memories:
            if memory.kind == "episodic" and memory.scope not in (None, "no_incident"):
                groups[(memory.scope, memory.gateway, memory.issuer)].append(memory)

        created = []
        for (scope, gateway, issuer), episodes in sorted(
            groups.items(), key=lambda kv: str(kv[0])
        ):
            if len(episodes) < min_support:
                continue
            days = sorted({e.day for e in episodes if e.day is not None})
            target = gateway or "the fleet"
            detail = f", affecting {issuer} traffic" if issuer else ""
            self._seq += 1
            memory = Memory(
                kind="semantic",
                text=(
                    f"RECURRING: {target} has shown {scope} degradation{detail} "
                    f"{len(episodes)} times (days {', '.join(map(str, days))}). "
                    f"Check it early, and confirm against current telemetry before relying on this."
                ),
                gateway=gateway,
                issuer=issuer,
                scope=scope,
                day=days[-1] if days else None,
                seq=self._seq,
                support=len(episodes),
            )
            memory.tags = _tokenize(memory.text)
            self.memories.append(memory)
            created.append(memory)
        return created

    # -- reading -----------------------------------------------------------

    def _idf(self) -> dict[str, float]:
        """Inverse document frequency, so shared boilerplate scores near zero."""
        n = len(self.memories) or 1
        df: Counter[str] = Counter()
        for memory in self.memories:
            df.update(set(memory.tags))
        return {term: math.log(1.0 + n / (1.0 + count)) for term, count in df.items()}

    def _score(self, memory: Memory, query_terms: set[str], idf: dict[str, float]) -> float:
        overlap = query_terms & set(memory.tags)
        score = sum(idf.get(term, 0.0) for term in overlap)
        # A consolidated pattern outranks any single episode behind it.
        if memory.kind == "semantic":
            score *= 1.0 + 0.35 * memory.support
        # Mild recency preference: gateways change behaviour, and a six-day-old
        # observation is weaker evidence than yesterday's.
        newest = max((m.seq for m in self.memories), default=1) or 1
        score *= 0.75 + 0.25 * (memory.seq / newest)
        return score

    def recall(self, query: str, k: int = 3) -> list[Memory]:
        """Top-k memories for a query. Zero-overlap memories are never returned."""
        if not self.memories:
            return []
        query_terms = set(_tokenize(query))
        if not query_terms:
            return []
        idf = self._idf()
        scored = [(self._score(m, query_terms, idf), m) for m in self.memories]
        hits = [(s, m) for s, m in scored if s > 0]
        hits.sort(key=lambda pair: (-pair[0], -pair[1].seq))
        return [m for _, m in hits[:k]]

    def recall_text(self, query: str, k: int = 3) -> list[str]:
        return [m.text for m in self.recall(query, k)]

    # -- persistence -------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        target = path or self.path
        if not target:
            raise ValueError("no path configured for this MemoryStore")
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump([m.to_dict() for m in self.memories], fh, indent=2)
        return target

    def load(self, path: str | None = None) -> None:
        target = path or self.path
        if not target or not os.path.exists(target):
            return
        with open(target, encoding="utf-8") as fh:
            self.memories = [Memory(**row) for row in json.load(fh)]
        self._seq = max((m.seq for m in self.memories), default=0)

    def stats(self) -> dict[str, int]:
        kinds = Counter(m.kind for m in self.memories)
        return {"total": len(self.memories), **kinds}
