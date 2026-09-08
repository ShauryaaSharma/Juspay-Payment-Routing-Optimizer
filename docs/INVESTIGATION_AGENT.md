# Investigation agent

The router detects that something broke. This layer works out **what**.

When the PID controller's calibration residual spikes, it knows the fleet is
underdelivering against its own forecast — and nothing more. It cannot say
which gateway, whether the damage is fleet-wide, or that the real problem is
one issuing bank being declined on one gateway. That gap is what an on-call
engineer closes manually at 2am, and it is what this agent automates.

```
router detects break  ->  agent investigates  ->  structured diagnosis
   (calibration            (tools over              (scope, gateway,
    residual)               telemetry)               issuer, evidence)
```

## What it does

Given an alert and a time window, the agent queries transaction telemetry
through six read-only tools and returns a structured diagnosis: the **scope**
of the incident (`single_gateway`, `issuer_specific`, `fleet_wide`,
`no_incident`), the gateway and issuer responsible, a confidence, and the
specific numbers it based that on.

```bash
python run_investigation.py --case issuer_outage
```

```
ISSUER_SPECIFIC on PG-Delta / HDFC  (confidence 0.70)
  PG-Delta is declining HDFC traffic at 9.49% while other issuers convert near 93.33%.
  evidence:
    - HDFC on PG-Delta: 9.49%
    - other issuers on PG-Delta average 93.33%
  4 steps, 3 tool calls (0 errors), 0.01s
    step 0: get_fleet_health(...)
    step 1: compare_windows(gateway=PG-Delta, ...)
    step 2: segment_failures(by=issuer, gateway=PG-Delta, ...)
```

That incident is invisible at the fleet level — PG-Delta's aggregate success
rate only sags from 91% to 70%, which looks like a soft gateway rather than a
bank being declined outright. Only segmentation separates the two.

## The eval set is the point

Most agent evals are graded by vibes, or by another model, because nobody knows
the right answer. Here the simulator **plants** each incident, so ground truth
is exact: which gateway, which issuer, which scope, which minutes. Grading is
`==`, not a judgement call.

Eight cases across all four scopes. Two pairs do the real work:

| pair | why it discriminates |
|---|---|
| `partial_degradation` vs `issuer_outage` | Nearly identical fleet-level telemetry. One is a gateway soft across every issuer; the other is soft *because* one issuer is failing. Only segmentation separates them. |
| `fleet_wide` vs `fleet_wide_partial` | In the second, four of five gateways degrade and one stays healthy. The scope is still fleet-wide, but any rule of the form "fleet-wide only if *every* gateway is down" will instead blame whichever of the four looks worst. |

And `diurnal_trough` is a false positive by construction: PG-Charlie sits at
79%, comfortably "degraded" by any static threshold, because that is its normal
overnight trough — the identical dip appears at the same hour on every previous
day. Only a baseline comparison tells routine variation from an incident.

### The eval caught its own uselessness

The first version of the eval had six cases. The deterministic baseline policy
— a fixed heuristic with no model at all — scored **100%**.

That is not a good result, it is a broken eval. All six cases were solvable by
"find the gateway with the lowest success rate and describe it", which is
precisely what the heuristic's thresholds encode. The eval was measuring what
its designer already knew, and with every arm saturated at 1.0 it could not
have detected a prompt regression, a memory bug, or a model downgrade.

`diurnal_trough` and `fleet_wide_partial` were added specifically to invert a
threshold rule. The baseline now scores **75% (6/8)**, failing exactly those
two. There is a test that fails if the baseline ever returns to 100%, because
that would mean the eval has quietly lost its discriminating power again.

## Why there is a no-model baseline

`BaselinePolicyClient` runs a fixed investigation heuristic — fleet sweep, pick
the worst gateway, compare against a baseline window, segment if the damage
looks partial — with no LLM involved.

It exists because an eval without a floor cannot tell you anything. If the
model scores 0.82, the only useful question is "compared to what?" A
hand-written heuristic is the honest comparison, and it makes the entire suite
runnable in CI with no API key and no network.

## Loop engineering

The loop is written by hand rather than delegated to the SDK's tool runner. The
runner is the right default for most agents, but everything that pages someone
lives in the parts it abstracts away. Five guarantees:

1. **Bounded** — steps, tool calls, output tokens, and wall-clock all capped.
2. **Terminating** — every exit path sets a `stop_reason`; no branch can spin.
3. **Recoverable** — a tool error is data, not an exception. Only *repeated*
   errors break the loop.
4. **Non-repeating** — an identical repeated tool call gets a nudge instead of
   the same bytes again, which is how these loops usually stall.
5. **Answer-forcing** — the final step is reserved to demand a conclusion, so
   budget exhaustion degrades to a low-confidence answer rather than nothing.

One interaction worth knowing, found by testing rather than by design: when the
model repeats the *same* failing call, duplicate suppression fires first and
returns a nudge, so the consecutive-error counter resets and the error circuit
breaker never trips. The loop still terminates — via the step budget instead.
Both paths are bounded, but only one is the one you would guess, so there is a
test that documents it.

## Prompt engineering, versioned

Prompts live in `src/agent/prompts.py` as immutable named versions, so a prompt
change is a reviewable diff and the harness can run two versions head to head.

- **`v1-baseline`** states the role and the output contract, nothing else. Kept
  as the control arm — without it there is no way to show later work bought
  anything.
- **`v2-procedural`** adds an explicit investigation procedure and names the
  three traps: concluding from a null success rate, blaming one gateway during
  a fleet-wide event, and missing an issuer-scoped failure because the
  aggregate looked merely soft.

Every system prompt is a static string. A `datetime.now()` in there would drop
the cache hit rate to zero and nothing would fail loudly enough to notice.

## Memory, and how it is measured

Two tiers, because they answer different questions:

- **Episodic** — what happened in one past incident.
- **Semantic** — what repetition taught us. "PG-Bravo has now degraded three
  times, always overnight." No single episode contains that; `consolidate()`
  derives it once several episodes can be compared.

Memory is **injected** into the brief rather than exposed as a tool. Both work,
but injection makes memory a clean on/off variable. A `search_memory` tool
would confound "memory helped" with "the agent chose to use memory", and the
A/B would stop answering the question it was built to ask.

It is measured on **recurrence**, not on a single pass: each arm runs the case
set twice with memory persisting between passes. A cold pass has nothing to
remember, so measuring there measures nothing.

Memories reach the model as explicitly untrusted hypotheses. A memory is a
claim written by a previous run of a fallible agent — treating it as
established fact is how one early wrong diagnosis becomes permanent.

## Results

```
arm              pass   exact  partial   scope   brier   seg%  tools
v1-baseline         0    75%     0.80    75%   0.210   62%    2.4
v2-procedural       0    75%     0.80    75%   0.210   62%    2.4
v2+memory           1    75%     0.80    75%   0.210   62%    2.4
```

All three arms tie because the baseline policy ignores prompts and memory
entirely — it is a fixed heuristic. **That is the expected result, and it is
also the honest statement of what has been measured so far: the prompt and
memory arms have not been run against a live model.** The harness, the arms,
and the gate are built and tested; the comparison itself needs an API key.

Run `python run_evals.py --live` to fill that table in.

## Metrics, and why these

| metric | what it catches |
|---|---|
| `exact_match` | Scope AND gateway AND issuer all correct. The headline — a diagnosis naming the wrong gateway is wrong. |
| `partial_credit` | 0.5/0.3/0.2 split, so progress is visible while exact_match is still 0. |
| `brier` | Calibration. Right at 0.9 confidence scores 0.01; wrong at 0.9 scores 0.81. Confidently wrong is the expensive failure on call. |
| `segmentation_rate` | Process, not outcome. An agent that never segments cannot be solving the hard cases, even when it guesses them right. |
| `hallucination_rate` | Named a gateway or issuer that does not exist. |
| `tool_calls` / `tokens` | Cost. A correct answer that took 15 tool calls is not free. |

The LLM judge grades only what has no ground truth — whether the write-up would
help the engineer who got paged — and reports in separate columns so a fluent
wrong answer can never inflate the headline. Its limits are documented in
`graders.py`: it sees ground truth, it is not calibrated against human raters,
and it scores prose rather than correctness.

## Regression gate

```bash
python run_evals.py --save-baseline   # record current scores
python run_evals.py --gate            # exit 1 if exact_match regressed
```

This is what makes the eval a gate rather than a dashboard. Wire it into CI and
a prompt edit that quietly breaks the hard cases fails the build instead of
shipping. Tolerance is near-zero because the offline arm is fully
deterministic; a live-model arm needs a wider band and several seeds before its
variance is known.

## Closing the loop

The agent no longer just diagnoses and stops. A confirmed finding becomes a
routing constraint the router honours on the next transaction.

```bash
python close_loop.py
```

### Why this needs per-transaction context

A flat bandit tracks one number per gateway. When PG-Delta starts declining
HDFC cards, all it sees is PG-Delta's *aggregate* success rate falling, and its
only available response is to route away from PG-Delta — **for everyone**. It
fixes HDFC by taking the fleet's best gateway away from the 69% of traffic that
was converting on it perfectly.

The closed loop can express what is actually true: *avoid PG-Delta for HDFC,
and leave everyone else where they are.* That is why `RoutingContext` was
threaded through every router — a finding about one issuer is unactionable
without it.

### Measured

PG-Delta is the fleet's best gateway (94.5%) and starts declining HDFC at
minute 1440. Detection fires 60 minutes later; both arms share a seed.

| minutes 1500–1920 | open loop | closed loop | delta |
|---|---|---|---|
| overall success rate | 92.31% | 93.07% | **+76 bps** |
| HDFC success rate | 92.48% | 92.46% | −1 bps |
| other issuers success rate | 92.24% | 93.34% | **+110 bps** |

| traffic share to PG-Delta | open loop | closed loop |
|---|---|---|
| HDFC | 0.8% | 0.8% |
| other issuers | 0.8% | **46.3%** |

**Read the HDFC row carefully: it is unchanged.** The flat bandit had already
rescued HDFC by abandoning PG-Delta wholesale. The entire gain is in the third
row — the closed loop fixes HDFC *without* taking the best gateway away from
everyone else. The headline is not "we fixed the broken issuer", it is "we
stopped punishing the healthy ones to do it".

### Three properties that keep the loop safe

**It refuses more often than it acts.** `from_diagnosis` returns `None` for low
confidence, for `fleet_wide` and `no_incident` (neither names a culprit, so
steering between gateways cannot help), for an issuer-scoped finding with no
issuer named, and for any gateway not in the fleet. That last one is the final
barrier between a hallucinated gateway name and production traffic. Most of
`test_constraints.py` tests what the layer must *refuse* to do.

**It never blocks completely.** Every constraint leaks a small `canary_rate`
(2%) to the gateway it blocks. Without it a blocked gateway stops producing
observations, its posterior freezes, and nothing can ever justify lifting the
constraint — the system makes itself permanently blind. That is the same
survivorship trap the router already fights, except self-inflicted. Canary
outcomes feed the bandit's posterior like any other, so recovery is noticed on
its own.

**It always expires.** Constraints carry a TTL. A constraint written during an
incident is evidence about the past, and gateways recover. On expiry traffic
returns; if the fault persists, the next investigation re-applies it.

And a fourth, really a safety interlock: if honouring every constraint would
leave a transaction with nowhere to go, none are applied. A transaction that
reaches no gateway is a guaranteed failure, strictly worse than one routed to a
suspect gateway.

### The loop's precision depends on traffic volume

Found by testing, not by design. Segmenting by issuer needs enough traffic in
each gateway-by-issuer cell to compute a rate. In this scenario:

| transactions / 2 days | diagnosis | constraint |
|---|---|---|
| 40k | `single_gateway` | blunt — all traffic off PG-Delta |
| 150k | `issuer_specific` | scoped — only HDFC |

Below the sampling floor the agent cannot establish that the failure is
issuer-scoped and falls back to a blunt constraint. That is worse than the
scoped one but **safe**: it matches what a flat bandit would have done anyway,
and the agent withholds the claim it cannot support rather than guessing an
issuer. The crossover is not a sharp cliff — between roughly 80k and 120k the
outcome flips with sampling noise — so only the two ends are pinned in tests.

The practical consequence: this technique needs volume. On a low-traffic
merchant the loop degrades to what you already had.

## What I would build next

1. **Run the live arms.** Everything is in place; the numbers are not.
2. **Multi-seed live runs.** One run of a sampled model is an anecdote. The
   gate tolerance cannot be set honestly until the variance is measured.
3. **Confirmation before expiry.** Today a constraint expires on a timer. It
   should wake the agent to re-check first, so a still-broken path is renewed
   and a recovered one is retired on evidence.
4. **Learn the constraint instead of asserting it.** A contextual bandit
   (issuer × gateway) would eventually discover the same thing without an
   investigation. The interesting comparison is which is faster, and the honest
   guess is that the agent wins early and the contextual bandit wins later.
5. **Human-in-the-loop verification.** `remember_investigation` already takes a
   `verified` flag that nothing currently sets outside evals. Wiring it to an
   engineer's thumbs-up turns the memory store into a growing labelled dataset.
6. **Adversarial cases.** Two gateways failing at once, an incident spanning
   the window boundary, telemetry with a gap. Each breaks a different
   assumption the current agent makes.
