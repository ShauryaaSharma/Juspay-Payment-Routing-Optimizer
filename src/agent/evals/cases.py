"""Eval cases with real ground truth.

Most agent evals are graded by vibes or by another model, because nobody knows
the right answer. Here the simulator *plants* each incident, so the correct
diagnosis is known exactly: which gateway, which issuer, which scope, which
minutes. That turns "did the agent get it right" from a judgement call into a
string comparison, and it is the single biggest reason this eval is worth
trusting.

Six scenarios, chosen to cover all four scopes and -- more importantly -- the
pair that is genuinely easy to confuse:

* ``partial_degradation`` is one gateway soft across *every* issuer.
* ``issuer_outage`` is one gateway soft *because* one issuer is failing.

Their fleet-level telemetry looks nearly identical. Only segmentation
separates them, so an agent that skips that step scores 50% on the pair no
matter how fluent its write-up.

Telemetry is generated with a high-exploration epsilon-greedy router rather
than the benchmark winner, deliberately. Every case must be *answerable*: the
hard cases turn on comparing one issuer's success rate against the others on
the same gateway, which needs enough traffic in every gateway-by-issuer cell
to compute a rate at all. Under the benchmark's winning router the affected
gateway drew ~120 transactions in the window, so four of five issuer segments
came back as null and the case could not be solved by any agent, however good.
Raising volume to 300k and exploration to 25% puts every cell above the
sampling floor. The router still adapts, so `get_traffic_shift` still shows it
pulling away from a dying gateway -- the survivorship signal survives, it just
no longer censors the evidence.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ...gateways import TICKS_PER_DAY, DegradationEvent, GatewaySpec
from ...metrics import RunResult
from ...routers import EpsilonGreedyRouter
from ...simulator import SimConfig, run_once
from ..telemetry import TelemetryStore, format_clock

EVAL_TRANSACTIONS = 300_000
EVAL_DAYS = 7
EVAL_SEED = 11

INCIDENT_START = 3 * TICKS_PER_DAY + 8 * 60  # day 4, 08:00
INCIDENT_END = INCIDENT_START + 360  # six hours

# Midnight on day 4: the bottom of PG-Charlie's nightly cycle in _diurnal_fleet.
TROUGH_START = 3 * TICKS_PER_DAY
TROUGH_END = TROUGH_START + 360


def _calm_fleet() -> list[GatewaySpec]:
    """A stable five-gateway fleet with no incidents.

    Every scenario starts here and plants exactly one problem, so nothing in
    the background can be mistaken for the thing being diagnosed.
    """
    return [
        GatewaySpec("PG-Alpha", base_sr=0.93, base_latency_ms=240, diurnal_amplitude=0.01, cost_bps=18),
        GatewaySpec("PG-Bravo", base_sr=0.94, base_latency_ms=210, diurnal_amplitude=0.01, cost_bps=22),
        GatewaySpec("PG-Charlie", base_sr=0.92, base_latency_ms=180, diurnal_amplitude=0.02,
                    diurnal_phase=-math.pi / 2, cost_bps=15),
        GatewaySpec("PG-Delta", base_sr=0.91, base_latency_ms=150, diurnal_amplitude=0.01, cost_bps=9),
        GatewaySpec("PG-Echo", base_sr=0.94, base_latency_ms=200, diurnal_amplitude=0.01, cost_bps=20),
    ]


@dataclass(frozen=True)
class EvalCase:
    """One incident plus the answer key."""

    case_id: str
    alert: str
    window: tuple[int, int]
    expected_scope: str
    expected_gateway: str | None
    expected_issuer: str | None
    difficulty: str
    notes: str
    specs: list[GatewaySpec] = field(repr=False, default_factory=list)

    @property
    def window_readable(self) -> str:
        return f"{format_clock(self.window[0])} to {format_clock(self.window[1])}"


def _plant(index: int, **event_kwargs) -> list[GatewaySpec]:
    fleet = _calm_fleet()
    spec = fleet[index]
    fleet[index] = GatewaySpec(
        name=spec.name, base_sr=spec.base_sr, base_latency_ms=spec.base_latency_ms,
        latency_jitter_ms=spec.latency_jitter_ms, diurnal_amplitude=spec.diurnal_amplitude,
        diurnal_phase=spec.diurnal_phase, cost_bps=spec.cost_bps,
        events=[DegradationEvent(start=INCIDENT_START, end=INCIDENT_END, **event_kwargs)],
    )
    return fleet


def _diurnal_fleet() -> list[GatewaySpec]:
    """Calm fleet, but PG-Charlie has a pronounced overnight swing.

    Its success rate bottoms out near 79% at midnight every night with nothing
    wrong. That is the point: a number low enough to trip any static alert
    threshold, which is nonetheless entirely normal.
    """
    fleet = _calm_fleet()
    charlie = fleet[2]
    fleet[2] = GatewaySpec(
        name=charlie.name, base_sr=charlie.base_sr, base_latency_ms=charlie.base_latency_ms,
        latency_jitter_ms=charlie.latency_jitter_ms, diurnal_amplitude=0.14,
        diurnal_phase=-math.pi / 2, cost_bps=charlie.cost_bps,
    )
    return fleet


def _plant_most(spare: int = 4, **event_kwargs) -> list[GatewaySpec]:
    """Degrade every gateway except one -- a fleet event with a survivor."""
    return [
        GatewaySpec(
            name=s.name, base_sr=s.base_sr, base_latency_ms=s.base_latency_ms,
            latency_jitter_ms=s.latency_jitter_ms, diurnal_amplitude=s.diurnal_amplitude,
            diurnal_phase=s.diurnal_phase, cost_bps=s.cost_bps,
            events=[] if i == spare else
            [DegradationEvent(start=INCIDENT_START, end=INCIDENT_END, **event_kwargs)],
        )
        for i, s in enumerate(_calm_fleet())
    ]


def _plant_all(**event_kwargs) -> list[GatewaySpec]:
    return [
        GatewaySpec(
            name=s.name, base_sr=s.base_sr, base_latency_ms=s.base_latency_ms,
            latency_jitter_ms=s.latency_jitter_ms, diurnal_amplitude=s.diurnal_amplitude,
            diurnal_phase=s.diurnal_phase, cost_bps=s.cost_bps,
            events=[DegradationEvent(start=INCIDENT_START, end=INCIDENT_END, **event_kwargs)],
        )
        for s in _calm_fleet()
    ]


GENERIC_ALERT = (
    "Conversion is down. The routing controller flagged a calibration break at "
    "{when} -- realised success rate diverged from what the router forecast. "
    "Work out what is wrong."
)


def build_cases() -> list[EvalCase]:
    """The eval set.

    Alerts are deliberately uninformative: they say "something is wrong", never
    which gateway. An alert that names the culprit would test reading
    comprehension instead of diagnosis.
    """
    when = format_clock(INCIDENT_START)
    alert = GENERIC_ALERT.format(when=when)
    window = (INCIDENT_START, INCIDENT_END)

    return [
        EvalCase(
            case_id="hard_outage",
            alert=alert, window=window,
            expected_scope="single_gateway", expected_gateway="PG-Bravo", expected_issuer=None,
            difficulty="easy",
            notes="PG-Bravo collapses to ~5%. Obvious in fleet health; the only "
                  "trap is that the router pulls away, thinning the sample.",
            specs=_plant(1, sr_multiplier=0.05, latency_multiplier=3.0, label="Bravo hard outage"),
        ),
        EvalCase(
            case_id="partial_degradation",
            alert=alert, window=window,
            expected_scope="single_gateway", expected_gateway="PG-Charlie", expected_issuer=None,
            difficulty="medium",
            notes="PG-Charlie drops to ~65% across ALL issuers. Pairs with "
                  "issuer_outage: segmenting shows a flat profile, so the correct "
                  "answer is single_gateway, not issuer_specific.",
            specs=_plant(2, sr_multiplier=0.70, label="Charlie partial degradation"),
        ),
        EvalCase(
            case_id="issuer_outage",
            alert=alert, window=window,
            expected_scope="issuer_specific", expected_gateway="PG-Delta", expected_issuer="HDFC",
            difficulty="hard",
            notes="PG-Delta declines HDFC almost entirely; other issuers are fine. "
                  "Aggregate SR only sags, so this is invisible without segmenting.",
            specs=_plant(3, sr_multiplier=0.10, issuer="HDFC", label="Delta/HDFC decline"),
        ),
        EvalCase(
            case_id="issuer_outage_secondary",
            alert=alert, window=window,
            expected_scope="issuer_specific", expected_gateway="PG-Alpha", expected_issuer="ICICI",
            difficulty="hard",
            notes="Same shape as issuer_outage on a different gateway/issuer pair, "
                  "so a lucky guess of PG-Delta/HDFC does not score twice.",
            specs=_plant(0, sr_multiplier=0.15, issuer="ICICI", label="Alpha/ICICI decline"),
        ),
        EvalCase(
            case_id="fleet_wide",
            alert=alert, window=window,
            expected_scope="fleet_wide", expected_gateway=None, expected_issuer=None,
            difficulty="medium",
            notes="Every gateway degrades together. Punishes an agent that blames "
                  "whichever gateway it happened to inspect first.",
            specs=_plant_all(sr_multiplier=0.72, label="fleet-wide degradation"),
        ),
        EvalCase(
            case_id="no_incident",
            alert=alert, window=window,
            expected_scope="no_incident", expected_gateway=None, expected_issuer=None,
            difficulty="medium",
            notes="Nothing is planted; the alert is a false positive. Measures "
                  "whether the agent will invent an incident when asked to find one.",
            specs=_calm_fleet(),
        ),
        # -- discriminating cases -------------------------------------------
        # The six above are all solvable by "find the gateway with the lowest
        # success rate and describe it", which is why the fixed-threshold
        # baseline policy scored 100% on them. These two are not: each one
        # inverts a threshold rule, so passing them requires comparing against
        # context rather than reading a number off a dashboard.
        EvalCase(
            case_id="diurnal_trough",
            alert=GENERIC_ALERT.format(when=format_clock(TROUGH_START)),
            window=(TROUGH_START, TROUGH_END),
            expected_scope="no_incident", expected_gateway=None, expected_issuer=None,
            difficulty="hard",
            notes="PG-Charlie sits near 79% -- comfortably 'degraded' by any fixed "
                  "threshold -- but this is its normal overnight trough and the same "
                  "dip appears at the same hour on every previous day. Only a "
                  "baseline comparison distinguishes routine variation from an "
                  "incident. A threshold policy reports a false incident here.",
            specs=_diurnal_fleet(),
        ),
        EvalCase(
            case_id="fleet_wide_partial",
            alert=alert, window=window,
            expected_scope="fleet_wide", expected_gateway=None, expected_issuer=None,
            difficulty="hard",
            notes="Four of five gateways degrade together; PG-Echo stays healthy. "
                  "The scope is still fleet_wide -- the damage is not attributable "
                  "to any one gateway. A policy that requires *every* gateway to be "
                  "down before saying fleet_wide will instead blame whichever of the "
                  "four looks worst.",
            specs=_plant_most(sr_multiplier=0.70, label="fleet-wide, Echo spared"),
        ),
    ]


CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "eval_cache"
)
# Bump when anything that shapes the telemetry changes, so a stale cache can
# never be silently graded against new ground truth.
CACHE_VERSION = "v2"


def _cache_path(case_id: str) -> str:
    return os.path.join(
        os.path.abspath(CACHE_DIR),
        f"{case_id}_{CACHE_VERSION}_{EVAL_TRANSACTIONS}_{EVAL_DAYS}_{EVAL_SEED}.npz",
    )


@lru_cache(maxsize=None)
def _telemetry_for(case_id: str) -> TelemetryStore:
    """Simulate (or reload) one case's telemetry.

    Generating six weeks of transactions takes ~2.5 minutes, which is long
    enough to discourage running the eval -- and an eval you avoid running is
    an eval that stops catching regressions. Results are deterministic given
    the seed, so they cache to disk and every later run starts instantly.
    """
    case = next(c for c in build_cases() if c.case_id == case_id)
    path = _cache_path(case_id)

    if os.path.exists(path):
        with np.load(path) as data:
            result = RunResult(
                router=str(data["router"]), chosen=data["chosen"], success=data["success"],
                latency_ms=data["latency_ms"], instant_regret=data["instant_regret"],
                cost_bps=data["cost_bps"], n_gateways=int(data["n_gateways"]),
                tick=data["tick"], issuer=data["issuer"],
            )
        return TelemetryStore(result, case.specs)

    config = SimConfig(days=EVAL_DAYS, transactions=EVAL_TRANSACTIONS, seeds=(EVAL_SEED,))
    result = run_once(
        lambda n, s: EpsilonGreedyRouter(n, epsilon=0.25, gamma=0.999, seed=s),
        case.specs, config, EVAL_SEED,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path, router=result.router, chosen=result.chosen, success=result.success,
        latency_ms=result.latency_ms, instant_regret=result.instant_regret,
        cost_bps=result.cost_bps, n_gateways=result.n_gateways,
        tick=result.tick, issuer=result.issuer,
    )
    return TelemetryStore(result, case.specs)


def telemetry_for(case: EvalCase) -> TelemetryStore:
    """Simulated telemetry for one case, memoised across repeated eval runs."""
    return _telemetry_for(case.case_id)
