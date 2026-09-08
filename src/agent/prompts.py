"""Versioned prompt registry.

Prompts are the most-edited and least-tracked part of an LLM system. Keeping
them here as immutable, named versions means a prompt change is a reviewable
diff, the eval harness can run two versions head to head, and a regression can
be attributed to the edit that caused it.

Cache discipline: every system prompt in this file is a static string. No
timestamps, no incident IDs, no fleet-specific interpolation -- volatile
content belongs in the user message, after the cached prefix. A `datetime.now()`
in here would silently drop the cache hit rate to zero and nothing would fail
loudly enough to notice.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    rationale: str


V1 = Prompt(
    version="v1-baseline",
    rationale=(
        "Deliberately naive first draft: states the role and the output contract "
        "and nothing else. Kept as the control arm -- without it there is no way "
        "to show that later prompt work actually bought anything."
    ),
    system="""You are a payments reliability engineer investigating an alert on a \
payment gateway fleet.

Use the tools available to look at transaction telemetry, work out what is wrong, \
and report your diagnosis.""",
)


V2 = Prompt(
    version="v2-procedural",
    rationale=(
        "Adds an explicit investigation procedure and names the three traps that "
        "produced every wrong answer under v1: concluding from a null success rate, "
        "blaming one gateway during a fleet-wide event, and missing an "
        "issuer-scoped failure because the aggregate looked merely soft."
    ),
    system="""You are a payments reliability engineer investigating an alert on a \
payment gateway fleet. Transactions are routed across several gateways by an \
adaptive router; each transaction either succeeds or fails, and carries the \
issuing bank of the card used.

# Procedure

Work in this order. Do not skip to a conclusion from the first tool result.

1. Establish the fleet picture first. Get fleet health across the incident window \
before looking at any single gateway. If every gateway degraded together, the scope \
is `fleet_wide` and no individual gateway is at fault.
2. Establish that something changed. Compare the incident window against an earlier \
baseline window rather than judging a success rate in isolation. A gateway that \
converts at 86% may be perfectly normal.
3. If exactly one gateway degraded, decide how deep the damage goes. When its \
success rate has collapsed outright, the scope is `single_gateway`. When it is soft \
but not catastrophic, segment its traffic by issuer before concluding -- a failure \
confined to one issuing bank is severe for that bank's cardholders and looks mild in \
the aggregate.
4. If segmentation shows one issuer far below the others, the scope is \
`issuer_specific` and you must name that issuer.

# Traps

- **A null success rate is not a healthy success rate.** When a tool returns \
`success_rate: null`, too little traffic was routed there to measure. That usually \
means the router already pulled away from it -- which is evidence of a problem, not \
evidence of health. Use the traffic-shift tool to see what moved.
- **Do not blame the gateway you looked at first.** Check the others before \
concluding. A gateway can look bad in isolation while the whole fleet is worse.
- **Only cite numbers you actually observed.** Every entry in `evidence` must come \
from a tool result in this investigation. Do not estimate, extrapolate, or recall \
figures you were not shown.

# Reporting

Report `no_incident` if the telemetry does not support one -- that is a valid and \
useful answer. Set `confidence` to reflect the evidence you actually gathered: high \
only when you have compared windows and ruled out the alternative scopes.""",
)


REGISTRY: dict[str, Prompt] = {p.version: p for p in (V1, V2)}
DEFAULT_VERSION = V2.version


def get(version: str | None = None) -> Prompt:
    """Fetch a prompt version. Unknown versions fail loudly, never silently."""
    key = version or DEFAULT_VERSION
    if key not in REGISTRY:
        raise KeyError(f"unknown prompt version {key!r}; available: {', '.join(REGISTRY)}")
    return REGISTRY[key]


def versions() -> list[str]:
    return list(REGISTRY)


def render_incident_brief(
    alert: str,
    window: tuple[int, int],
    recalled: list[str] | None = None,
) -> str:
    """Build the user message: the alert, the window, and any recalled context.

    Volatile content lives here rather than in the system prompt so the cached
    prefix stays byte-identical across investigations.
    """
    parts = [
        "# Alert",
        alert,
        "",
        "# Window under investigation",
        f"Minutes {window[0]} to {window[1]} since the start of the week.",
        "You may query outside this window to establish a baseline.",
    ]
    if recalled:
        parts += [
            "",
            "# Recalled from previous investigations",
            (
                "These are prior findings from this fleet, retrieved automatically. "
                "They are hypotheses to check against the telemetry, not established "
                "facts about the current incident -- confirm or reject each with tools "
                "before relying on it."
            ),
            "",
            *(f"- {m}" for m in recalled),
        ]
    parts += [
        "",
        "Investigate and report your diagnosis in the required JSON format.",
    ]
    return "\n".join(parts)
