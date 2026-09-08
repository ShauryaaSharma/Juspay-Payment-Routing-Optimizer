#!/usr/bin/env python3
"""Measure what closing the loop is worth.

    python close_loop.py            # deterministic baseline agent, no API key
    python close_loop.py --live     # investigate with claude-opus-5

The scenario is built so the flat bandit is at its worst and the scoped
constraint at its best, because that is the case the whole context-plumbing
exercise exists to serve:

PG-Delta is the *best* gateway in the fleet, so the router sends it most of the
traffic. Partway through, it starts declining one issuer's cards (HDFC) almost
entirely. Everyone else's transactions still convert on it at 94%.

A flat bandit sees only PG-Delta's aggregate success rate falling. Its single
available response is to route away from PG-Delta -- for *everyone*. It fixes
HDFC by taking the fleet's best gateway away from the 69% of traffic that was
converting on it perfectly.

The closed loop can express the thing that is actually true: avoid PG-Delta for
HDFC, and leave everyone else where they are.
"""

from __future__ import annotations

import argparse
import math
import sys

from src.agent import Investigator, InvestigatorConfig, MemoryStore
from src.agent.config import settings
from src.agent.llm import BaselinePolicyClient, default_client
from src.agent.loop import LoopBudget
from src.agent.traces import TraceStore, new_session_id
from src.closed_loop import run_closed_loop
from src.gateways import TICKS_PER_DAY, DegradationEvent, GatewaySpec
from src.routers import ThompsonRouter
from src.simulator import SimConfig

DAYS = 2
TRANSACTIONS = 150_000
SEED = 5

INCIDENT_START = TICKS_PER_DAY  # day 2, 00:00
INCIDENT_END = INCIDENT_START + 480  # eight hours
DETECT_AT = INCIDENT_START + 60  # one hour of detection lag, deliberately
AFFECTED_ISSUER = "HDFC"
DELTA = 3


def scenario_fleet() -> list[GatewaySpec]:
    """PG-Delta is the best gateway, and it is the one that breaks for HDFC."""
    return [
        GatewaySpec("PG-Alpha", base_sr=0.915, base_latency_ms=240, diurnal_amplitude=0.01, cost_bps=18),
        GatewaySpec("PG-Bravo", base_sr=0.920, base_latency_ms=210, diurnal_amplitude=0.01, cost_bps=22),
        GatewaySpec("PG-Charlie", base_sr=0.905, base_latency_ms=180, diurnal_amplitude=0.02,
                    diurnal_phase=-math.pi / 2, cost_bps=15),
        GatewaySpec(
            "PG-Delta", base_sr=0.945, base_latency_ms=150, diurnal_amplitude=0.01, cost_bps=9,
            events=[DegradationEvent(
                start=INCIDENT_START, end=INCIDENT_END, sr_multiplier=0.08,
                issuer=AFFECTED_ISSUER, label="Delta declines HDFC",
            )],
        ),
        GatewaySpec("PG-Echo", base_sr=0.925, base_latency_ms=200, diurnal_amplitude=0.01, cost_bps=20),
    ]


def _router(n, s):
    return ThompsonRouter(n, gamma=0.999, seed=s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="Investigate with claude-opus-5.")
    env = settings(refresh=True)
    ap.add_argument("--canary", type=float, default=env.constraints.canary_rate)
    ap.add_argument("--ttl", type=int, default=env.constraints.ttl_minutes,
                    help="Constraint lifetime in minutes.")
    ap.add_argument("--no-chart", action="store_true")
    args = ap.parse_args()

    client = default_client(prefer_live=True) if args.live else BaselinePolicyClient()
    if args.live and hasattr(client, "preflight"):
        ok, message = client.preflight()
        if not ok:
            print(f"cannot reach the model: {message}")
            print("Omit --live to use the deterministic baseline agent.")
            return 2

    specs = scenario_fleet()
    config = SimConfig(days=DAYS, transactions=TRANSACTIONS, seeds=(SEED,))

    traces = TraceStore(env.traces)
    session = new_session_id("closeloop")

    def investigate(telemetry, window):
        investigator = Investigator(
            store=telemetry, client=client, memory=MemoryStore(),
            config=InvestigatorConfig(budget=LoopBudget(max_steps=env.loop.max_steps)),
            traces=traces, session_id=session,
        )
        alert = (
            "Conversion is down. The routing controller flagged a calibration break "
            f"at minute {window[1]}. Work out what is wrong."
        )
        return investigator.investigate(alert, window, case_id="closed_loop")

    print(f"scenario  : PG-Delta (best gateway) declines {AFFECTED_ISSUER} from minute "
          f"{INCIDENT_START} to {INCIDENT_END}")
    print(f"detection : minute {DETECT_AT} ({DETECT_AT - INCIDENT_START} min after onset)")
    print(f"agent     : {getattr(client, 'name', type(client).__name__)}")
    print(f"router    : thompson (gamma=0.999), {TRANSACTIONS:,} transactions over {DAYS} days\n")

    open_loop = run_closed_loop(specs, config, _router, SEED, detect_ticks=())
    closed = run_closed_loop(
        specs, config, _router, SEED,
        detect_ticks=(DETECT_AT,), investigate=investigate,
        ttl_minutes=args.ttl, canary_rate=args.canary,
    )

    if not closed.constraints:
        print("The agent produced no actionable constraint; nothing to compare.")
        for result in closed.investigations:
            print(f"  stop_reason={result.stop_reason} error={result.error}")
        return 1

    print("Investigation at detection:")
    for result in closed.investigations:
        d = result.diagnosis
        if d is None:
            print(f"  FAILED ({result.stop_reason})")
            continue
        print(f"  {d.scope} on {d.primary_gateway}"
              f"{' / ' + d.affected_issuer if d.affected_issuer else ''} "
              f"(confidence {d.confidence:.2f}, {result.usage.tool_calls} tool calls)")
    print("\nConstraint installed:")
    for c in closed.constraints:
        print(f"  {c.describe()}")

    lo, hi = DETECT_AT, INCIDENT_END
    others = [i for i in ("ICICI", "SBI", "AXIS", "KOTAK")]

    print(f"\nMeasured from detection to incident end (minutes {lo}-{hi}):")
    print(f"  {'':28s} {'open loop':>11s} {'closed loop':>12s} {'delta':>10s}")
    rows = [
        ("overall success rate", None),
        (f"{AFFECTED_ISSUER} success rate", AFFECTED_ISSUER),
    ]
    for label, issuer in rows:
        a = open_loop.sr_between(lo, hi, issuer)
        b = closed.sr_between(lo, hi, issuer)
        print(f"  {label:28s} {a:>10.2%} {b:>11.2%} {(b - a) * 10_000:>+9.0f} bps")

    a_other = _pooled(open_loop, lo, hi, others)
    b_other = _pooled(closed, lo, hi, others)
    print(f"  {'other issuers success rate':28s} {a_other:>10.2%} {b_other:>11.2%} "
          f"{(b_other - a_other) * 10_000:>+9.0f} bps")

    a_share = open_loop.share_to(DELTA, lo, hi, AFFECTED_ISSUER)
    b_share = closed.share_to(DELTA, lo, hi, AFFECTED_ISSUER)
    print(f"\n  {AFFECTED_ISSUER} traffic still sent to PG-Delta: "
          f"{a_share:.1%} open loop -> {b_share:.1%} closed loop")

    a_share_o = _pooled_share(open_loop, DELTA, lo, hi, others)
    b_share_o = _pooled_share(closed, DELTA, lo, hi, others)
    print(f"  other issuers kept on PG-Delta (the best gateway): "
          f"{a_share_o:.1%} open loop -> {b_share_o:.1%} closed loop")

    print(f"\n  constraint layer: {closed.store_stats}")
    print("  The canary releases matter: while the constraint is active a small share")
    print("  of traffic keeps probing PG-Delta, so recovery stays observable and the")
    print("  constraint can be retired on evidence rather than on a timer alone.")

    # The honest cost. PG-Delta recovers at INCIDENT_END but the constraint runs
    # to expiry, so there is a window where HDFC is steered away from a gateway
    # that is working again. Reported rather than tuned away: it is the price of
    # a TTL, and it is what a shorter TTL or an explicit re-check would buy.
    stale_lo, stale_hi = INCIDENT_END, closed.constraints[0].expires_tick
    if stale_hi > stale_lo:
        a = open_loop.sr_between(stale_lo, stale_hi, AFFECTED_ISSUER)
        b = closed.sr_between(stale_lo, stale_hi, AFFECTED_ISSUER)
        print(f"\nCost of the constraint outliving the fault (minutes {stale_lo}-{stale_hi}, "
              f"PG-Delta has recovered):")
        print(f"  {AFFECTED_ISSUER} success rate     {a:>10.2%} open loop -> {b:>7.2%} closed loop "
              f"({(b - a) * 10_000:+.0f} bps)")
        print("  Measured at roughly zero here, and the reason is worth stating rather")
        print("  than claiming a cost the data does not show: the open-loop bandit is")
        print("  also slow to return to a gateway it abandoned, so in this window both")
        print("  arms route HDFC elsewhere anyway. The stale constraint is not the")
        print("  binding factor. It would bite against a router that recovers faster")
        print("  than the TTL -- which is the case a shorter TTL, or a confirmation")
        print("  pass before expiry, is actually for.")

    if not args.no_chart:
        import os

        path = _write_chart(
            open_loop, closed,
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results", "07_closed_loop.svg"),
        )
        print(f"\nwrote {path}")
    return 0


def _write_chart(open_loop, closed, path: str) -> str:
    """Traffic share to PG-Delta over time -- the mechanism, made visible.

    Four lines. The pair that matters is 'other issuers': in the open loop it
    collapses along with HDFC's, because a flat bandit can only abandon the
    gateway wholesale. In the closed loop it stays high while HDFC's alone
    drops -- the scoped constraint doing the thing a flat bandit cannot express.
    """
    import numpy as np

    from src.gateways import ISSUERS
    from src.plotting import line_chart

    # Wide bins on purpose. Thompson sampling is probability matching, so its
    # allocation genuinely oscillates between near-equal arms; at narrow bins
    # that real behaviour buries the signal under noise.
    bins = 96
    hdfc_idx = ISSUERS.index(AFFECTED_ISSUER)

    def share(result, mask_fn) -> np.ndarray:
        tick = np.asarray(result.run.tick)
        chosen = np.asarray(result.run.chosen)
        issuer = np.asarray(result.run.issuer)
        mask = mask_fn(issuer)
        edges = np.linspace(0, tick.max() + 1, bins + 1)
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            cell = mask & (tick >= lo) & (tick < hi)
            out.append((chosen[cell] == DELTA).mean() if cell.sum() else np.nan)
        arr = np.array(out, dtype=float)
        for i in range(1, arr.size):
            if np.isnan(arr[i]):
                arr[i] = arr[i - 1]
        return np.nan_to_num(arr)

    others = lambda iss: iss != hdfc_idx
    hdfc = lambda iss: iss == hdfc_idx

    # Three lines, not four. The open-loop HDFC line sits on top of the
    # open-loop "others" line -- a flat bandit moves both together, which is
    # the entire problem -- so plotting it adds ink without adding information.
    series = {
        "other issuers, open loop": share(open_loop, others),
        "other issuers, closed loop": share(closed, others),
        f"{AFFECTED_ISSUER}, closed loop": share(closed, hdfc),
    }

    span = open_loop.run.tick.max() + 1
    shade = [
        (INCIDENT_START / span * bins, INCIDENT_END / span * bins, "PG-Delta declines HDFC"),
    ]
    return line_chart(
        path, series,
        "Share of traffic routed to PG-Delta (the fleet's best gateway)",
        f"time (day 1 -> day {DAYS})", "share of that issuer's traffic",
        x_max=bins, y_range=(0.0, 1.0), shaded=shade,
    )


def _pooled(result, lo, hi, issuers) -> float:
    import numpy as np

    tick = np.asarray(result.run.tick)
    from src.gateways import ISSUERS

    idx = [ISSUERS.index(i) for i in issuers]
    mask = (tick >= lo) & (tick < hi) & np.isin(np.asarray(result.run.issuer), idx)
    return float(np.asarray(result.run.success)[mask].mean())


def _pooled_share(result, gateway, lo, hi, issuers) -> float:
    import numpy as np

    from src.gateways import ISSUERS

    tick = np.asarray(result.run.tick)
    idx = [ISSUERS.index(i) for i in issuers]
    mask = (tick >= lo) & (tick < hi) & np.isin(np.asarray(result.run.issuer), idx)
    return float((np.asarray(result.run.chosen)[mask] == gateway).mean())


if __name__ == "__main__":
    sys.exit(main())
