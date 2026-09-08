# Payment Routing Optimizer

[![CI](https://github.com/ShauryaaSharma/Juspay-Payment-Routing-Optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/ShauryaaSharma/Juspay-Payment-Routing-Optimizer/actions/workflows/ci.yml)

**An adaptive payment router that lifts success rate by +323 bps over a static
baseline, an LLM agent that diagnoses *why* a gateway broke, and the loop
between them closed — a confirmed diagnosis becomes a routing constraint the
router honours on the next transaction.**

| | |
|---|---|
| **+323 bps** | success rate over a static weighted router, 3 seeds |
| **+110 bps** | for the issuers a flat bandit would punish while fixing a broken one |
| **75%** | ground-truth eval score for a no-model baseline — headroom kept on purpose |
| **158 tests** | numpy-only · no API key required · CI runs the eval gate on every push |

![Realised success rate over one simulated week](results/03_rolling_sr.svg)

*The plunge is a static router still feeding a gateway that collapsed to 5%
success. The adaptive routers dip and recover.*

Started from Juspay's published research on control-theoretic payment routing —
see [Inspiration](#inspiration).

**If you have two minutes**, read [Part 1 — the γ result](#the-interesting-result)
(the discount factor that makes the best router win is not a constant), then
[Part 4](#part-4--was-the-agent-even-necessary) (a contextual bandit beats the
LLM agent, reported rather than tuned away).

---

## Inspiration

> **"A Control-Theoretic Approach to Dynamic Payment Routing for Success Rate
> Optimization"** — Juspay Technologies
>
> Juspay's Intelligence Systems team describes using *"control-theoretic
> feedback loops and multi-arm RL bandits to dynamically steer transactions
> toward the most reliable gateways."*

That sentence is the entire starting point for this project. It names a problem
worth solving — routing payments across gateways whose reliability changes
underneath you — and two techniques for solving it, and I wanted to find out
what actually happens when you build it.

**This is not a reproduction of that paper.** I worked from the problem
statement, not the algorithm: a non-stationary bandit over a simulated gateway
fleet, then a control loop on top of it, then an honest measurement of whether
the control loop helps.

The headline answer is a **negative result**. The PID controller does *not* beat
a correctly tuned discount factor. What it does buy — and why that turns out to
be the more useful finding — is in [Part 1](#the-interesting-result): the
correct discount factor is not a constant, it moves with transaction volume, so
a fleet-wide setting is wrong for most merchants on it. The controller is not a
better bandit; it is a way to avoid a per-merchant tuning problem that scales
with merchant count.

Anyone who has read the paper will spot where this diverges from it. That is
expected, and the divergences are documented rather than smoothed over.

How the three layers line up with the work Juspay describes:

| Their track | What is here |
|---|---|
| **Intelligence Systems** — control-theoretic feedback, RL bandits, routing | [Part 1](#part-1--the-router): five bandit strategies, a PID controller on the posterior's calibration residual, and a γ-sensitivity study |
| **JusTrust** — real-time trust infrastructure, agentic investigation, explainable ML | [Part 2](#part-2--the-investigation-agent): a tool-using agent that diagnoses gateway failures with cited evidence, graded against planted ground truth |
| **Backend & Platform** — reliability, observability at scale | Trace persistence, env-driven config, an eval regression gate wired into CI, and off-policy evaluation so a policy change can be assessed before it touches live traffic |

---

## Architecture

```mermaid
flowchart TB
    subgraph ENV["Simulated environment — src/gateways.py, src/simulator.py"]
        FLEET["Gateway fleet: 5 gateways x 5 issuers<br/>diurnal drift, outages, issuer-scoped failures<br/>ground truth known, never exposed to the agent"]
    end

    subgraph ROUTING["Routing — src/routers/"]
        BANDIT["Bandit strategies<br/>Thompson · UCB1 · eps-greedy · static<br/>PID-Thompson · contextual (per gateway x issuer)"]
        CONSTRAINT["Constraint layer — src/agent/constraints.py<br/>issuer-scoped · 2% canary · TTL · refuses low confidence"]
    end

    subgraph AGENT["Investigation agent — src/agent/"]
        LOOP["Agent loop — loop.py<br/>bounded · terminating · recoverable · answer-forcing"]
        TOOLS["6 read-only tools — tools.py<br/>strict JSON schemas over telemetry"]
        MEM["Memory — memory.py<br/>episodic + semantic, consolidated"]
        PROMPT["Prompt registry — prompts.py<br/>immutable versions, v1 control vs v2"]
    end

    subgraph MEASURE["Measurement"]
        EVALS["Ground-truth evals — evals/<br/>8 planted incidents · exact grading · CI gate"]
        OPE["Off-policy evaluation — ope.py<br/>IPS · SNIPS · DM · DR + ESS diagnostics"]
    end

    subgraph OPS["Operations"]
        TRACE["Trace store — traces.py<br/>JSONL or Postgres, fails soft"]
        CONF["Config — config.py<br/>env-driven, secrets redacted"]
    end

    FLEET -->|transactions| BANDIT
    BANDIT -->|"calibration residual spikes"| LOOP
    LOOP --> TOOLS
    TOOLS -->|"observable telemetry only"| FLEET
    MEM -.->|untrusted hypotheses| LOOP
    PROMPT -.-> LOOP
    LOOP -->|structured diagnosis| CONSTRAINT
    CONSTRAINT -->|blocks gateway x issuer| BANDIT
    LOOP -->|every step, tool call, token| TRACE
    LOOP --> EVALS
    BANDIT -->|logged propensities| OPE
    CONF -.-> AGENT
    CONF -.-> TRACE
```

### Stack, and what each piece replaces

The project has **one runtime dependency: numpy.** That is a deliberate
constraint — the whole system runs offline and reproducibly, and the interesting
parts stay visible instead of being delegated to a framework. Every slot below
is one a typical LLM stack fills with a library:

| Slot | What this project uses | Common alternative |
|---|---|---|
| Agent orchestration | Hand-written loop, `src/agent/loop.py` | LangGraph, CrewAI, the Anthropic SDK tool runner |
| Model access | Anthropic SDK, `claude-opus-5`, adaptive thinking + structured outputs | — |
| Tool layer | 6 tools, `strict: true` JSON schemas | LangChain tools, MCP servers |
| Memory + retrieval | Episodic/semantic store with IDF-weighted lexical recall | Pinecone, Weaviate, Chroma + embeddings |
| Prompt management | Versioned immutable registry, A/B'd by the harness | LangSmith, PromptLayer |
| Tracing / observability | Own store: JSONL or Postgres, cost & latency per diagnosis | Langfuse, LangSmith, W&B Weave |
| Evaluation | Ground-truth harness, Brier calibration, CI regression gate | Braintrust, Promptfoo, DeepEval |
| Off-policy evaluation | IPS / SNIPS / DM / DR, `src/ope.py` | Open Bandit Pipeline |
| Charts | ~200-line SVG writer, `src/plotting.py` | matplotlib, plotly |
| CI | GitHub Actions: 3 Python versions + eval gate | — |

The orchestration row is the one worth defending. The SDK's tool runner is the
right default for most agents; it was skipped here because everything that
actually pages someone lives in the parts it abstracts away — budget exhaustion
mid-investigation, repeated tool errors, a duplicated call, an unparseable final
answer. Those are the branches under test in `tests/test_agent_loop.py`.

---

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q                 # 158 tests, ~30s
```

CI runs the suite on Python 3.11–3.13 and then runs the eval regression gate on
every push — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml). No API
key is needed there: the agent falls back to the deterministic baseline policy,
which is exactly why that baseline exists.

**The router**

```bash
python benchmark.py                 # 6 strategies, 7 days, 100k transactions, 3 seeds
python sweep_gamma.py               # discount-factor sensitivity study
```

**The agent** (no API key needed)

```bash
python run_investigation.py --list          # show the eval cases
python run_investigation.py --case issuer_outage
python run_evals.py                         # full eval suite, all arms
python run_evals.py --gate                  # exit 1 on regression
python run_evals.py --live                  # against claude-opus-5
```

**The closed loop**

```bash
python close_loop.py                        # measures what closing the loop is worth
python contextual_vs_agent.py               # was the agent even necessary?
python run_ope.py                           # off-policy evaluation vs ground truth
```

Charts are written to `results/` as dependency-free SVG.

### Configuration and traces

Everything tunable resolves from environment variables, documented in
[`.env.example`](.env.example):

```bash
cp .env.example .env
```

**Every default reproduces the behaviour these numbers were measured with**, so
an unset variable can never silently change a result. Credentials are read from
the environment only, never written to a trace, and never printed — a DSN is
reduced to `<redacted>@host:port/db` anywhere it is logged.

Every investigation is persisted as a trace: each step, each tool call with its
arguments and result, tokens, timings, cost, stop reason, and the diagnosis.
Local runs append JSONL; set `TRACE_DSN` to store rows in a database instead.

```bash
python manage_traces.py config    # resolved settings, secrets redacted
python manage_traces.py tail      # recent local traces
```

```bash
# hosted Postgres (Neon, Supabase, Railway, RDS, ...)
export TRACE_DSN='postgresql://user:pass@host:5432/db?sslmode=require'
python manage_traces.py init      # create table + indexes
python manage_traces.py check     # verify a write succeeds
```

[**docs/TRACE_QUERIES.md**](docs/TRACE_QUERIES.md) is a query cookbook: cost per
prompt version, cost per *successful* diagnosis, whether prompt caching is
actually working, tool latency and error rates, calibration, and the
find-candidate-eval-cases query that turns a suspicious trace into a regression
test. It covers the JSONL backend too, so none of it needs a database.

Traces earn their place for three reasons: a wrong diagnosis is almost
impossible to explain from the answer alone; the best eval cases start life as
real traces someone flagged; and cost-per-diagnosis tracked per prompt version
is how you notice that a prompt edit tripled spend.

**Tracing never fails a run.** Writes are wrapped, failures counted and warned
once. An unreachable database degrades to "no traces" within
`TRACE_CONNECT_TIMEOUT` seconds (default 5) rather than stalling an
investigation — observability failing is an annoyance; observability taking
down incident response is an outage.

---

## Part 1 — The router

Several payment gateways, each with a success rate that drifts on a daily cycle
and occasionally collapses without warning. The router never observes a
gateway's true success rate — only the outcomes of transactions it chose to
send there. That feedback loop is what makes this a bandit problem rather than
a forecasting one: route away from a gateway and you stop learning about it,
including the fact that it recovered.

### Results

Seven simulated days, 100k transactions, 5 gateways, averaged over 3 seeds:

| router | success rate | total regret | outage response | cost (bps) |
|---|---|---|---|---|
| static-weighted | 91.73% | 6474 | never | 16.79 |
| ucb1 (γ=0.999) | 92.93% | 3004 | 22 min | 18.12 |
| thompson (γ=1.0) | 94.13% | 1859 | 216 min | 20.19 |
| epsilon-greedy (γ=0.999) | 94.88% | 1085 | 0 min | 19.51 |
| pid-thompson (γ=1.0) | 94.93% | 1029 | 240 min | 20.32 |
| **thompson (γ=0.999)** | **94.96%** | **1025** | **9 min** | 19.06 |

**+323 bps of success rate** over the static baseline. At India-scale payment
volumes that ratio is the difference between a lot of completed payments and a
lot of abandoned carts.

*Outage response* is how long a router kept feeding a gateway after it died. A
static router never notices. `pid-thompson` scores badly here despite a strong
success rate — it deliberately keeps probing the dead gateway at just above the
5% measurement threshold. A real tradeoff, left visible rather than tuned away.

In [the chart at the top](#payment-routing-optimizer), the plunge inside the
shaded band is the static router still feeding a gateway that has collapsed to
5%. The second, smaller dip later in the week is a separate brownout.

### The interesting result

Discounted Thompson sampling wins. But the discount factor γ that makes it win
is not a constant:

| γ | SR @ 20k tx/week | SR @ 100k tx/week |
|---|---|---|
| 1.0 (no forgetting) | 93.53% | 94.13% |
| 0.9999 | 94.12% | **95.47%** |
| 0.999 | **94.66%** | 94.96% |
| 0.99 | 93.61% | 93.79% |
| *pid-thompson, γ=1.0* | *94.27%* | *94.93%* |

The optimum **moves with transaction volume**, because γ forgets per
*observation* while gateways degrade per *unit time*. A fleet-wide constant is
therefore wrong for most merchants on it, and getting it wrong costs 50–170 bps.

The PID controller is a response to that. It replaces the fixed discount with a
feedback loop that measures its own forecast error and injects exploration only
when the model has gone stale. Against an untuned posterior it is worth **+74
to +80 bps**, it beats a *badly* set γ at both volumes, and it recovers roughly
60% of the gap to a *perfectly* tuned γ with no tuning at all.

**It does not beat a correctly tuned γ.** Stacking both slightly over-explores.
That is the honest finding; `sweep_gamma.py` reproduces it. The controller is
not a better bandit — it is a way to avoid a per-merchant tuning problem that
scales with merchant count.

### How the controller works

Inner loop is Thompson sampling over Beta posteriors. Outer loop is a PID
controller whose error signal is a **calibration residual**:

```
error = EWMA(predicted SR of the arm we chose) − EWMA(what we actually got)
```

When the posterior is calibrated this is zero, the controller idles, and it
costs nothing. When a gateway dies, predictions stay high while outcomes
collapse — the residual spikes, and the controller widens the posterior by
dividing both Beta parameters by an inflation factor, preserving the mean while
increasing variance. Deliberate, bounded, self-cancelling doubt.

Two setpoints that look reasonable and are not, both tried and discarded:

- **A constant target SR** needs hand-tuning per merchant and cannot separate
  "our model is stale" from "this merchant's traffic is simply harder".
- **`max(posterior means)`** is the maximum of several noisy estimates, so it
  sits *above* the true best arm. The bias never washes out, the integrator
  winds up against it, and the router over-explores permanently. This was the
  first implementation — it is why the controller idled at inflation 4.15
  instead of 1.0.

Because the residual is unbiased, the loop also stays quiet during network-wide
degradation: when every gateway is down, predictions fall with outcomes, and
spreading traffic around would only widen the damage.

---

## Part 2 — The investigation agent

The controller knows the fleet is underdelivering against its own forecast, and
nothing more. It cannot say which gateway, whether the damage is fleet-wide, or
that the real problem is one issuing bank being declined on one gateway. That
gap is what an on-call engineer closes manually at 2am.

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

### The eval set is the point

Most agent evals are graded by vibes, or by another model, because nobody knows
the right answer. Here the simulator **plants** each incident, so ground truth
is exact: which gateway, which issuer, which scope, which minutes. Grading is
`==`, not a judgement call.

Eight cases across all four scopes (`single_gateway`, `issuer_specific`,
`fleet_wide`, `no_incident`). Two pairs do the real work:

| pair | why it discriminates |
|---|---|
| `partial_degradation` vs `issuer_outage` | Nearly identical fleet-level telemetry. One is a gateway soft across every issuer; the other is soft *because* one issuer is failing. Only segmentation separates them. |
| `fleet_wide` vs `fleet_wide_partial` | In the second, four of five gateways degrade and one stays healthy. Scope is still fleet-wide, but any rule of the form "fleet-wide only if *every* gateway is down" blames whichever of the four looks worst. |

And `diurnal_trough` is a false positive by construction: PG-Charlie sits at
79%, comfortably "degraded" by any static threshold, because that is its normal
overnight trough — the identical dip appears at the same hour every day.

### The eval caught its own uselessness

The first version had six cases. The deterministic baseline policy — a fixed
heuristic with **no model at all** — scored **100%**.

That is not a good result, it is a broken eval. All six were solvable by "find
the gateway with the lowest success rate and describe it", which is precisely
what the heuristic's thresholds encode. It was measuring what its designer
already knew, and with every arm saturated at 1.0 it could not have detected a
prompt regression, a memory bug, or a model downgrade.

`diurnal_trough` and `fleet_wide_partial` were added specifically to invert a
threshold rule. The baseline now scores **75% (6/8)**, failing exactly those
two. There is a test that fails if it ever returns to 100%.

### Why there is a no-model baseline

`BaselinePolicyClient` runs a fixed investigation heuristic — fleet sweep, pick
the worst gateway, compare against a baseline window, segment if the damage
looks partial — with no LLM involved.

An eval without a floor cannot tell you anything. If the model scores 0.82, the
only useful question is "compared to what?" A hand-written heuristic is the
honest comparison, and it keeps the whole suite runnable with no key.

### Loop engineering

The loop is hand-written rather than delegated to the SDK's tool runner. The
runner is the right default for most agents, but everything that pages someone
lives in the parts it abstracts away. Five guarantees:

1. **Bounded** — steps, tool calls, output tokens, wall-clock all capped.
2. **Terminating** — every exit path sets a `stop_reason`; no branch can spin.
3. **Recoverable** — a tool error is data, not an exception. Only *repeated*
   errors break the loop.
4. **Non-repeating** — an identical repeated tool call gets a nudge instead of
   the same bytes again, which is how these loops usually stall.
5. **Answer-forcing** — the final step is reserved to demand a conclusion, so
   budget exhaustion degrades to a low-confidence answer rather than nothing.

One interaction found by testing, not design: when the model repeats the *same*
failing call, duplicate suppression fires first and returns a nudge, so the
consecutive-error counter resets and the error circuit breaker never trips. The
loop still terminates — via the step budget instead. Both paths are bounded,
but only one is the one you would guess, so there is a test documenting it.

### Prompts, versioned

Prompts live in `src/agent/prompts.py` as immutable named versions, so a prompt
change is a reviewable diff and the harness can run two versions head to head.

- **`v1-baseline`** states the role and output contract, nothing else. The
  control arm — without it there is no way to show later work bought anything.
- **`v2-procedural`** adds an explicit investigation procedure and names the
  three traps: concluding from a null success rate, blaming one gateway during
  a fleet-wide event, and missing an issuer-scoped failure because the
  aggregate looked merely soft.

Every system prompt is a static string. A `datetime.now()` in there would drop
the cache hit rate to zero and nothing would fail loudly enough to notice.

### Memory

Two tiers, because they answer different questions:

- **Episodic** — what happened in one past incident.
- **Semantic** — what repetition taught us. *"PG-Bravo has now degraded three
  times, always overnight."* No single episode contains that; `consolidate()`
  derives it once several episodes can be compared.

Memory is **injected** into the brief rather than exposed as a tool. Both work,
but injection makes memory a clean on/off variable — a `search_memory` tool
would confound "memory helped" with "the agent chose to use memory".

It is measured on **recurrence**: each arm runs the case set twice with memory
persisting between passes. A cold pass has nothing to remember.

Memories reach the model as explicitly **untrusted hypotheses**. A memory is a
claim written by a previous run of a fallible agent; treating it as established
fact is how one early wrong diagnosis becomes permanent.

### Metrics

| metric | what it catches |
|---|---|
| `exact_match` | Scope AND gateway AND issuer all correct. The headline. |
| `partial_credit` | 0.5/0.3/0.2 split, so progress shows while exact_match is 0. |
| `brier` | Calibration. Right at 0.9 confidence scores 0.01; wrong at 0.9 scores 0.81. Confidently wrong is the expensive failure on call. |
| `segmentation_rate` | Process, not outcome. An agent that never segments cannot be solving the hard cases, even when it guesses them right. |
| `hallucination_rate` | Named a gateway or issuer that does not exist. |
| `tool_calls` / `tokens` | Cost. A correct answer that took 15 tool calls is not free. |

An optional LLM judge grades only what has no ground truth — whether the
write-up would help the engineer who got paged — in separate columns, so a
fluent wrong answer can never inflate the headline.

---

## Part 3 — Closing the loop

A confirmed finding becomes a routing constraint the router honours on the next
transaction.

### Why this needs per-transaction context

A flat bandit tracks one number per gateway. When PG-Delta starts declining
HDFC cards, all it sees is PG-Delta's *aggregate* success rate falling, and its
only available response is to route away from PG-Delta — **for everyone**. It
fixes HDFC by taking the fleet's best gateway away from the 69% of traffic that
was converting on it perfectly.

The closed loop can express what is actually true: *avoid PG-Delta for HDFC,
and leave everyone else where they are.*

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

![Traffic share routed to PG-Delta, open loop vs closed loop](results/07_closed_loop.svg)

The mechanism in one picture. Inside the shaded incident, two lines collapse to
zero: HDFC under the closed loop (correctly steered away) and *other issuers
under the open loop* (collateral damage). The line that stays up is other
issuers under the closed loop — healthy traffic keeping the fleet's best
gateway, which is the entire difference between the two arms.

### Three properties that keep it safe

**It refuses more often than it acts.** `from_diagnosis` returns `None` for low
confidence, for `fleet_wide` and `no_incident` (neither names a culprit), for
an issuer-scoped finding with no issuer named, and for any gateway not in the
fleet — the last barrier between a hallucinated gateway name and production
traffic. Most of `test_constraints.py` tests what the layer must *refuse* to do.

**It never blocks completely.** Every constraint leaks 2% canary traffic to the
gateway it blocks. Without it a blocked gateway stops producing observations,
its posterior freezes, and nothing can ever justify lifting the constraint —
the same survivorship trap the router already fights, self-inflicted. Canary
outcomes feed the posterior like any other, so recovery is noticed on its own.

**It always expires.** Constraints carry a TTL, because a constraint written
during an incident is evidence about the past and gateways recover.

Plus a safety interlock: if honouring every constraint would leave a
transaction with nowhere to go, none are applied. A transaction that reaches no
gateway is a guaranteed failure, strictly worse than one routed to a suspect.

### Precision depends on traffic volume

Found by testing, not design. Segmenting by issuer needs enough traffic in each
gateway-by-issuer cell to compute a rate:

| transactions / 2 days | diagnosis | constraint |
|---|---|---|
| 40k | `single_gateway` | blunt — all traffic off PG-Delta |
| 150k | `issuer_specific` | scoped — only HDFC |

Below the sampling floor the agent falls back to a blunt constraint. Worse than
scoped, but **safe**: it matches what a flat bandit would have done anyway, and
the agent withholds the claim it cannot support rather than guessing an issuer.
The crossover is not a sharp cliff — between ~80k and ~120k the outcome flips
with sampling noise — so only the two ends are pinned in tests.

**Practical consequence: this technique needs volume.** On a low-traffic
merchant the loop degrades to what you already had.

---

## Part 4 — Was the agent even necessary?

The obvious objection to everything above: *why build an LLM investigation loop
to discover that PG-Delta is declining HDFC cards, when a bandit conditioned on
the issuer would learn it on its own?*

```bash
python contextual_vs_agent.py
```

`ContextualThompsonRouter` keeps a posterior per (gateway, issuer) cell,
shrunk toward its gateway's pooled posterior. One knob, `k`, controls how hard:
`k=0` is pure per-cell, larger `k` means issuers look more alike. That matters
because conditioning on 5 issuers turns 5 posteriors into 25, so every cell
holds roughly a fifth of the evidence — context is not free.

### The prediction was wrong

I expected a crossover: the agent winning at low volume (one investigation,
fixed cost) and the contextual bandit winning at high volume. **There is no
crossover. The contextual bandit wins at every volume tested.**

| overall SR during incident | 25k | 50k | 100k | 200k |
|---|---|---|---|---|
| flat | 91.82% | 92.20% | 91.74% | 92.18% |
| **contextual** | **92.92%** | **93.22%** | **93.00%** | **93.33%** |
| contextual-nopool | 92.56% | 93.21% | 92.70% | 93.14% |
| closed-loop (agent) | 92.22% | 92.51% | 92.35% | 92.80% |

![Overall success rate by traffic volume](results/08_contextual_vs_agent.svg)

The mechanism is how much healthy traffic keeps using the best gateway:

| share of other issuers still on PG-Delta | |
|---|---|
| flat | 1–4% — abandons it entirely |
| contextual | 50–66% — never stops trusting it for them |
| closed-loop | 32–38% — recovers some, but not all |

Two reasons the agent trails, and the second is the interesting one:

1. **Detection lag.** The constraint lands 60 minutes in. The flat bandit has
   already spent that hour souring on PG-Delta for everyone.
2. **The constraint stops the bleeding but does not repair the damage.** It
   prevents further HDFC failures from being attributed to PG-Delta, but the
   posterior underneath is already contaminated by the failures that happened
   before detection — and the canary keeps adding a trickle more. A contextual
   bandit never contaminates it, because HDFC's failures only ever touched
   HDFC's cell.

### What that means

For a failure mode you can enumerate in advance — issuer × gateway — a
contextual bandit is simply the better tool, and this experiment says so rather
than arguing otherwise. The agent's value has to be defended somewhere else:

- it handles failures nobody enumerated as a dimension (a fleet-wide event, a
  diurnal trough mistaken for an incident) — a bandit has no cell for those;
- it produces an explanation a human can act on, which a posterior does not;
- it does not pay the dimensionality cost, which matters when the context is
  card network × issuer × amount band × merchant rather than five issuers.

The strongest version of this system is both: a contextual bandit for the axes
you know about, an agent for the ones you do not.

---

## Part 5 — Would this survive contact with production?

Everything above compares policies by *running* them. In production you cannot
do that: routing real payments through an untested policy to find out it is
worse costs declined transactions. Off-policy evaluation estimates a candidate's
value from logs the current policy already produced.

```bash
python run_ope.py
```

Five estimators — IPS, clipped IPS, SNIPS, Direct Method, Doubly Robust — plus
the diagnostics that say whether to believe any of them. Because this is a
simulator, the **true** value of each candidate is computable in closed form, so
every estimate is checked against the answer. That validation is the reason to
build OPE here rather than take it on faith.

Logging policy: Thompson sampling, 40k transactions, propensities recorded at
decision time.

| candidate | true | IPS | SNIPS | DR | ESS |
|---|---|---|---|---|---|
| flat-greedy | 0.9300 | 0.9415 | **0.9309** | **0.9306** | 11,790 |
| uniform | 0.8981 | **1.3408** | 0.8675 | **0.8956** | 675 |
| frozen-contextual | 0.9415 | **1.9476** | 0.9108 | 0.9342 | 119 |

Three things that fall out of this, none of which are obvious until you measure:

**IPS returns success rates above 100%.** Rewards are 0/1, so 1.9476 is not
merely wrong, it is incoherent — and nothing in the estimator prevents it. It
is unbiased and simultaneously useless, because a handful of transactions with
tiny logging propensities dominate the average.

**ESS is the number that decides whether an estimate means anything.** With
11,790 effective samples every estimator lands within 15 bps. With 119, IPS is
off by 100 percentage points. Same 40,000 rows.

**The Direct Method's confidence interval lies.** It reports a standard error of
0.0001 and misses the true value on all three candidates. Its variance reflects
only the spread of contexts, not how wrong the reward model is — so it looks
precise and is merely confident.

The practical rule the numbers support: ship SNIPS or DR, always report ESS
alongside, and treat a candidate far from the logging policy as unevaluable
rather than as evaluated-badly. Off-policy evaluation is interpolation, not
extrapolation.

There is also a constraint worth knowing before you deploy anything: **you
cannot do OPE from a deterministic logging policy.** UCB1 assigns probability
zero to every action it did not take, so importance weights are undefined and no
volume of logs will help. If you want to evaluate policies offline later, what
you deploy today has to be stochastic. There is a test asserting exactly that.

---

## What is measured, and what is not

Stated plainly, because it is the first thing worth asking about.

**Measured and reproducible:**
- Every router benchmark number, over 3 seeds.
- The γ sensitivity study.
- The agent eval suite at 75% exact-match against the deterministic baseline.
- The closed-loop comparison, both arms sharing a seed.

**Not measured:**
- **Anything requiring an API key.** The live prompt-vs-prompt arm, the
  memory-vs-no-memory arm, and `close_loop.py --live` are built, tested, and
  wired up — but the numbers reported here all come from the baseline policy.
  Set `ANTHROPIC_API_KEY` and run `python run_evals.py --live` to fill them in.
- **Model variance.** One run of a sampled model is an anecdote. The gate
  tolerance cannot be set honestly until variance is measured across seeds.

Everything here is a **simulation**. Gateway behaviour comes from
hand-specified curves. It demonstrates that the control loop and the agent
behave correctly under known ground truth; it is not evidence about any real
payment network.

---

## Project layout

```
src/gateways.py            non-stationary gateway simulator + fleet definitions
src/routers/               base.py holds DiscountedCounts, one file per strategy
src/routers/contextual.py  per-(gateway, issuer) Thompson with partial pooling
src/simulator.py           runs a router over a fleet, per-transaction records
src/metrics.py             regret, rolling SR, allocation, outage response
src/plotting.py            SVG charts, no dependencies
src/ope.py                 off-policy estimators: IPS, SNIPS, DM, DR + diagnostics
src/closed_loop.py         simulation that investigates and constrains itself

src/agent/telemetry.py     the observable view the agent queries
src/agent/tools.py         six read-only tools, strict schemas
src/agent/prompts.py       versioned prompt registry
src/agent/llm.py           Claude client + no-model baseline policy
src/agent/loop.py          the agent loop: budgets, termination, recovery
src/agent/memory.py        episodic + semantic memory with consolidation
src/agent/constraints.py   diagnosis -> routing constraint (canary + TTL)
src/agent/config.py        env-driven settings, .env loader, DSN redaction
src/agent/traces.py        trace persistence: JSONL and Postgres backends
src/agent/evals/           ground-truth cases, graders, arm harness, gate

benchmark.py               all router strategies, one table, five charts
sweep_gamma.py             discount-factor sensitivity study
run_investigation.py       one investigation, full trace
run_evals.py               the eval suite
close_loop.py              measures what closing the loop is worth
contextual_vs_agent.py     contextual bandit vs. agent, swept over volume
run_ope.py                 estimates policy value from logs, checked against truth
manage_traces.py           trace store: config / init / check / tail
.env.example               every environment variable, documented

docs/TRACE_QUERIES.md      SQL cookbook for the trace store
docs/DESIGN.md             router: what was tried, what failed, what is missing
docs/INVESTIGATION_AGENT.md  the agent: eval design, loop guarantees, memory
AGENTS.md                  conventions for AI coding agents working on this repo
```

## Testing

```bash
python -m pytest -q        # 158 tests
```

CI (`.github/workflows/ci.yml`) runs two jobs on every push:

1. **tests** — the full suite across Python 3.11, 3.12, and 3.13, plus an
   explicit check that the router's benchmark numbers are still reproducible
   (it asserts `realised_sr=0.94916, total_regret=532.5` and fails loudly with a
   pointer to invariant 1 if the RNG path changed).
2. **eval regression gate** — `run_evals.py --gate`, which exits non-zero if
   `exact_match` drops against `results/eval_baseline.json`. Simulated telemetry
   is cached on a key derived from every file that determines it, so a change to
   the fleet or the seed forces a rebuild rather than grading stale data. The
   eval table is published to the run summary and the report, metrics, and
   traces are uploaded as artifacts.

That second job is what makes the eval a gate rather than a dashboard: a prompt
edit that quietly breaks the hard cases fails the build instead of shipping.

| file | covers |
|---|---|
| `test_gateways.py` | Simulator correctness — it is the measuring instrument. |
| `test_routers.py` | Each strategy does what it claims; discounting follows regime change. |
| `test_metrics.py` | Regret and outage-response, the numbers that get quoted. |
| `test_agent_loop.py` | Every termination and recovery path, driven by scripted clients. |
| `test_agent_evals.py` | Tools, memory, graders, and the eval set's own integrity. |
| `test_constraints.py` | Mostly what the constraint layer must **refuse** to do. |
| `test_contextual.py` | Whether conditioning on issuer buys resolution a flat bandit cannot have. |
| `test_ope.py` | Estimator unbiasedness against closed-form truth, and the failure modes. |
| `test_traces.py` | Config parsing, secret redaction, and that tracing fails soft. |

Two tests exist specifically to catch the project fooling itself:
`test_baseline_policy_scores_below_ceiling_and_above_floor` fails if the eval
loses its discriminating power, and
`test_unconstrained_selection_is_unchanged_by_the_new_parameters` fails if the
context refactor silently changed the benchmark numbers.

## Limitations

- **Simulated, not real.** Hand-specified gateway curves, not production data.
- **Limited context.** The routers are flat bandits; issuer context reaches
  them only as a constraint handed down by the agent, never as something they
  learn. A production router would condition on card network, issuer bank,
  amount band, and merchant category directly — a contextual bandit.
- **Success rate only.** Cost per transaction is tracked and reported but not
  optimised. The real objective is closer to `max(SR) − λ·cost − μ·latency`.
  Note the adaptive routers all drift toward more expensive gateways (~19–20
  bps vs 16.79 static) — they buy conversion with fees.
- **No retry policy.** A failed transaction is not terminal; the second choice
  matters as much as the first, and the two decisions are not independent.
- **Single region, no router failure.** Nothing addresses what happens when the
  routing service is partitioned from its own state.

## References

1. **"A Control-Theoretic Approach to Dynamic Payment Routing for Success Rate
   Optimization"** — Juspay Technologies. The starting point for this project;
   see [Inspiration](#inspiration) for what was taken from it and what was not.
2. Thompson, W. R. (1933). *On the likelihood that one unknown probability
   exceeds another in view of the evidence of two samples.* Biometrika. The
   posterior-sampling rule the router is built on.
3. Garivier, A. & Moulines, E. (2011). *On upper-confidence bound policies for
   switching bandit problems.* The discounted / sliding-window treatment of
   non-stationarity that `DiscountedCounts` implements.
4. Dudík, M., Langford, J. & Li, L. (2011). *Doubly robust policy evaluation and
   learning.* The DR estimator in `src/ope.py`.
5. Swaminathan, A. & Joachims, T. (2015). *The self-normalized estimator for
   counterfactual learning.* Why SNIPS cannot return a value outside the
   observed reward range, unlike IPS.

---

## What I would build next

1. **Run the live arms.** Everything is in place; the numbers are not.
2. **Contextual routing.** Condition on issuer bank and card network directly.
   Issuer-level degradation is the failure a flat router cannot see at all.
3. **Confirmation before expiry.** A constraint should wake the agent to
   re-check rather than lapsing on a timer.
4. **Learn the constraint instead of asserting it.** A contextual bandit would
   eventually discover the same thing. The interesting comparison is which is
   faster — the honest guess is the agent wins early, the bandit wins later.
5. **Off-policy evaluation.** Inverse propensity scoring, so a new policy can
   be assessed on logged traffic before it touches live payments. This is what
   makes any of it deployable.
6. **State durability.** Posteriors are in-process. Production needs them to
   survive a restart and stay coherent across replicas — the point where this
   stops being an algorithms problem and becomes a distributed-systems one.
