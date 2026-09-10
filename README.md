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
| **11µs** | p50 routing decision latency, ~2,400x inside a 100ms budget |
| **232 tests** | numpy-only library · optional service layer · CI gate on every push |

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
    subgraph CALLER["CALLER — checkout/, separate process on :8001"]
        SHOP["Merchant checkout<br/>browser -> merchant -> router, server-to-server"]
    end

    subgraph HOT["HOT PATH — per transaction, microseconds"]
        API["POST /route — service/api.py<br/>FastAPI · touches no network"]
        BANDIT["Bandit strategies — src/routers/<br/>Thompson · UCB1 · eps-greedy · static<br/>PID-Thompson · contextual (gateway x issuer)"]
        CONSTRAINT["Constraint layer — src/agent/constraints.py<br/>issuer-scoped · 2% canary · TTL"]
    end

    subgraph COLD["COLD PATH — per incident, seconds"]
        LOOP["Agent loop — src/agent/loop.py<br/>bounded · terminating · recoverable"]
        TOOLS["6 read-only tools — tools.py<br/>strict JSON schemas over telemetry"]
        MEM["Memory — memory.py<br/>episodic + semantic, consolidated"]
        PROMPT["Prompt registry — prompts.py<br/>immutable versions, A/B'd"]
    end

    subgraph ENV["Environment — src/gateways.py, src/simulator.py"]
        FLEET["Gateway fleet: 5 gateways x 5 issuers<br/>diurnal drift, outages, issuer-scoped failures<br/>ground truth never exposed to the agent"]
    end

    subgraph STATE["Shared state — service/state.py"]
        REDIS["Redis<br/>constraints as TTL keys<br/>posterior snapshots (write-back, not write-through)"]
    end

    subgraph STREAM["Event path — service/events.py"]
        KAFKA["Kafka / Redpanda<br/>outcome events + propensities"]
        CH["ClickHouse<br/>OLAP store for off-policy evaluation"]
    end

    subgraph MEASURE["Measurement"]
        EVALS["Ground-truth evals — evals/<br/>8 planted incidents · CI gate"]
        OPE["Off-policy evaluation — ope.py<br/>IPS · SNIPS · DM · DR + ESS"]
        PROM["Prometheus — service/metrics.py<br/>decision latency, SR, constraints"]
    end

    subgraph GOV["Governance"]
        DSL["Constraint DSL — dsl/ (Haskell)<br/>parse · type-check · emit JSON"]
        TRACE["Trace store — traces.py<br/>JSONL or Postgres, fails soft"]
        CONF["Config — config.py<br/>env-driven, secrets redacted"]
    end

    SHOP -->|"POST /route, POST /outcome"| API
    API --> CONSTRAINT
    CONSTRAINT --> BANDIT
    BANDIT -->|transactions| FLEET
    FLEET -->|outcomes| KAFKA
    KAFKA --> CH
    CH -->|logged propensities| OPE
    CONSTRAINT <-->|TTL keys| REDIS
    BANDIT -.->|periodic snapshot| REDIS
    API --> PROM

    BANDIT -->|"calibration residual spikes"| LOOP
    LOOP --> TOOLS
    TOOLS -->|"observable telemetry only"| FLEET
    MEM -.->|untrusted hypotheses| LOOP
    PROMPT -.-> LOOP
    LOOP -->|structured diagnosis| CONSTRAINT
    LOOP -->|every step, tool call, token| TRACE
    LOOP --> EVALS
    DSL -->|validated constraints| CONSTRAINT
    CONF -.-> COLD
    CONF -.-> TRACE
```

The split down the middle is the load-bearing idea. `/route` runs on every
payment and must answer in microseconds, so it touches no network — the
posterior is in process and constraints come from a locally refreshed cache.
`/investigate` runs per incident, calls a language model, and takes seconds.
They share a codebase here for demonstration; in production they are separate
services with separate SLOs, because an agent that is slow is fine and a router
that is slow is an outage.

The merchant at the top is genuinely separate: its own process, its own port,
no shared memory, reaching the router only over the same HTTP API an external
integrator would use.

### Stack

**The library has one runtime dependency: numpy.** Every experiment, every
benchmark and the whole eval suite run offline with nothing else installed.
The service layer adds infrastructure on top, and **all of it is optional** —
with no Redis, Kafka, ClickHouse or API key the service still starts and still
routes, keeping state in memory and dropping events.

| Slot | What this project uses | Common alternative |
|---|---|---|
| Numerics | NumPy | — |
| HTTP service | FastAPI + Uvicorn, `service/api.py` | Flask, Litestar |
| Shared state | Redis — TTL keys for constraints, write-back snapshots for posteriors | Aerospike, DynamoDB, an in-house KvDB |
| Event stream | Kafka (Redpanda locally), `service/events.py` | Pulsar, Kinesis |
| Analytics store | ClickHouse over its HTTP interface | Druid, BigQuery |
| Metrics | Prometheus exposition format, written directly | `prometheus_client` |
| Constraint language | **Haskell** DSL — parse, type-check, emit JSON (`dsl/`) | YAML plus runtime validation |
| Packaging | Docker, multi-stage, non-root; Compose for the full stack | — |
| Agent orchestration | Hand-written loop, `src/agent/loop.py` | LangGraph, CrewAI, the Anthropic SDK tool runner |
| Model access | Anthropic SDK, `claude-opus-5`, adaptive thinking + structured outputs | — |
| Tool layer | 6 tools, `strict: true` JSON schemas | LangChain tools, MCP servers |
| Memory + retrieval | Episodic/semantic store, IDF-weighted lexical recall | Pinecone, Weaviate, Chroma + embeddings |
| Prompt management | **LangSmith** when configured, falling back to a versioned registry in git | PromptLayer |
| Tracing | **Langfuse** (trace trees), or JSONL / Postgres | LangSmith, W&B Weave |
| Evaluation | Ground-truth harness, Brier calibration, CI regression gate | Braintrust, Promptfoo, DeepEval |
| Off-policy evaluation | IPS / SNIPS / DM / DR, `src/ope.py` | Open Bandit Pipeline |
| Charts | ~200-line SVG writer, `src/plotting.py` | matplotlib, plotly |
| CI | GitHub Actions: 3 Python versions + eval gate | — |

Three rows are worth defending, because in each case the obvious library was
considered and declined.

**Agent orchestration.** The SDK's tool runner is the right default for most
agents. It was skipped because everything that actually pages someone lives in
the parts it abstracts away — budget exhaustion mid-investigation, repeated
tool errors, a duplicated call, an unparseable final answer. Those are the
branches under test in `tests/test_agent_loop.py`, and behind a framework they
are someone else's branches.

**Memory retrieval.** A vector database would retrieve better. It would also
make "does memory help?" inseparable from "is this embedding model good?", and
the memory A/B exists to answer the first question. Lexical scoring keeps the
experiment clean and adds no service dependency.

**Metrics.** `prometheus_client` is one line of `requirements.txt`. The text
exposition format is a documented, stable, sixty-line contract, and keeping the
library at numpy-only is worth more than the sixty lines.

---

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q                 # 232 tests, ~2 min
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

**The service** (Docker, or bare with `pip install -r requirements-service.txt`)

```bash
docker compose up -d                        # router + Redis + Kafka + ClickHouse + Prometheus
curl -s localhost:8000/health
python bench_latency.py                     # decision latency vs the 100ms budget
```

**The two front ends** — no Docker, no API key, no backends

```bash
uvicorn service.api:app --port 8000     # router + walkthrough UI  -> localhost:8000
uvicorn checkout.app:app --port 8001    # merchant checkout        -> localhost:8001
```

`docker compose up -d` starts both, plus every backend.

Charts are written to `results/` as dependency-free SVG.

### The walkthrough UI

**The loop, in about thirty seconds.**

`service/static/index.html` is a single file: no build step, no framework, no
dependency, served by the same FastAPI app as everything else. If you delete it
the service is unaffected — `/` says so and points at `/docs`.

It is not a dashboard bolted onto a simulation. Every button drives the real
system through `/simulate/*`, which plays the caller the service normally has:
it routes through the actual bandit, resolves the outcome against the actual
simulated fleet, and feeds it back through the actual update path. The only
thing standing in for production is the gateway itself. The numbers on screen
are the router's numbers.

The sequence it is built around:

| Step | What you do | What you see |
| --- | --- | --- |
| 1 | Send 600 transactions | The bandit explores, then concentrates. Traffic share and posterior success rate per gateway. |
| 2 | Break the busiest gateway, for HDFC only | The gateway's *aggregate* rate merely sags. Nothing looks broken. |
| 3 | Send 600 more | Fleet conversion falls a few points. **HDFC falls off a cliff.** |
| 4 | Investigate | The agent segments, names `PG-Bravo / HDFC`, installs a constraint scoped to that pair. |
| 5 | Send 600 more | HDFC recovers. Everyone else keeps the gateway. |

A recorded run, no API key, backends off:

```
after the fault      fleet 83.2%   HDFC 67.2%   ICICI 90.3%   SBI 90.4%   AXIS 87.8%
diagnosis            ISSUER_SPECIFIC on PG-Bravo / HDFC   confidence 0.70   3 tool calls
                       HDFC on PG-Bravo:            10.71%
                       other issuers on PG-Bravo:   97.67%
constraint           avoid PG-Bravo for HDFC, ticks 180-660, 2% canary
after the constraint fleet 92.2%   HDFC 89.7%   180 blocked decisions   5 canary releases
```

Two details the UI deliberately exposes rather than hides.

**The window is an operator control, not a constant.** The investigation reads a
time range, and the range decides the answer. Too wide and healthy traffic from
before the fault dilutes the segment below the 30-transaction minimum that makes
a success rate meaningful; too narrow and it opens *after* the bandit has already
fled the gateway, so there is nothing left to segment. Across 12 trials per
setting, a window matching the traffic sent since the fault produced the scoped
diagnosis 9 times out of 12, while the same run with the window at 90 or 120
minutes produced it 0 times out of 12 — it fell back to blaming the whole
gateway. Breaking a gateway sets the field to "everything since", which is what
an alert timestamp gives a real responder; it stays editable, and when the
diagnosis comes back gateway-wide the UI says why and tells you to widen it.

**A gateway-wide diagnosis is a real answer, not a failure.** It means no single
issuer cleared the sample minimum, so there was not enough evidence to blame one.
The honest response is a wider constraint, and that is what gets installed.

### The merchant checkout

**A second service, on port 8001, that consumes the first.**

The walkthrough UI above has one honest weakness: `/simulate/traffic` plays both
the merchant *and* the gateway. Nothing in the repo exercised the public contract
the way an integrator would, so nothing proved the router is a service rather
than a simulation with a web page attached.

`checkout/` is that proof. It is a separate process on a separate port sharing no
memory with the router, and it speaks the same three calls a real merchant
backend speaks:

| | call | what it is |
| --- | --- | --- |
| 1 | `POST /route` | "where should this payment go?" — production contract |
| 2 | `POST /simulate/attempt` | the acquiring network, simulated — **the only fiction** |
| 3 | `POST /outcome` | "here is what happened" — production contract |

Pick a bank, press Pay, and the page shows what the merchant got back: the
gateway chosen, the propensity that makes the decision replayable for off-policy
evaluation, which gateways the constraint layer withheld *before* the bandit
chose, and the decision latency in microseconds. Break a gateway for HDFC in the
router's UI on 8000, run the investigation, then pay as HDFC on 8001 — the
constraint the agent installed shows up in the merchant's trace:

```
HDFC   -> PG-Alpha    declined  blocked=['PG-Bravo']
HDFC   -> PG-Alpha    approved  blocked=['PG-Bravo']
ICICI  -> PG-Alpha    declined  blocked=['PG-Bravo']
```

Four decisions in it are deliberate.

**The browser never talks to the router.** A routing API decides where money
goes; that is a backend concern. The page posts to the merchant, and the merchant
calls the router server-to-server — which is how a real integration works, and
why there is no CORS configuration anywhere in this repo.

**No HTTP client dependency.** urllib is enough for three JSON POSTs.
`service/events.py` posts to ClickHouse the same way.

**A router outage is a 502, not a 500,** with the command to fix it in the
message, and `/health` reports the merchant and its dependency separately so the
page explains the outage instead of showing a dead button. That path is tested
against a real closed port rather than a mock, because "the dependency is
unreachable" is the one failure a merchant integration has to handle well.

**No card fields.** Routing decisions are made per issuing bank, so net banking
is the method wired up; UPI, cards and wallets are shown inert. A test asserts
the page renders nothing that could collect a card number. Kirana Cart is a
fictional storefront.

One measurement worth keeping. The first version reported a 6.1-second round trip
for three localhost calls, which made the router look slow when it was answering
in microseconds. The cause was `localhost` resolving to `::1` first on Windows,
with urllib waiting for that connection to fail before retrying IPv4. Pinning
`127.0.0.1` took it to 32 ms. The router's own `decision_micros` was never the
problem, and would have been blamed.

### An invariant the demo layer carries

The demo endpoints carry one invariant worth naming, because getting it wrong
fails silently. The service's tick is wall-clock minutes; the demo replays hours
of traffic in milliseconds. Without an explicit simulated clock every transaction
lands on tick 0, every windowed telemetry query comes back empty, and the agent
answers "no gateway carried enough traffic to assess" no matter what is actually
broken — which reads as a broken agent rather than a broken clock.
`RouterService.clock_offset` is zero in production and advanced by the demo at
10 transactions per simulated minute, matching the offline simulator's implicit
rate. `tests/test_demo.py` pins it.

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

### Langfuse and LangSmith

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` and traces go to Langfuse
instead, as a tree rather than a flat row:

```
agent      one investigation
├─ span    step 0
│  └─ tool   get_fleet_health    (input: window, output: the JSON returned)
├─ span    step 1
│  └─ tool   segment_failures
└─ generation  model call, with token usage so Langfuse costs it
```

That nesting is what a flat row cannot give you: opening a wrong diagnosis and
seeing *which tool result the reasoning turned on*. Langfuse over LangSmith for
this specifically because it self-hosts — payment telemetry under RBI
localisation rules cannot go to a SaaS-only endpoint, and `LANGFUSE_HOST` is a
one-line change.

Prompts can come from **LangSmith** (`AGENT_PROMPT_SOURCE=langsmith`) instead of
the registry in `src/agent/prompts.py`, which allows editing a prompt without a
deploy.

**That trade has a cost worth naming.** A prompt that changes without a code
change means the eval gate no longer gates the prompt that actually ran: CI
green on a commit stops implying "this prompt scores 75%" and starts implying
"some prompt did, once". Three things keep it honest — `AGENT_PROMPT_COMMIT`
pins an exact commit, the resolved source and commit hash are written onto every
trace, and an unreachable LangSmith **falls back to the registry** rather than
failing. Offline reproducibility is not negotiable, so resolution fails open.

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

## Part 6 — Running it in production shape

Everything above is a simulation you invoke from a script. This part makes it a
service you can deploy, and measures the one thing the rest of the project never
did: whether a routing decision is fast enough to be allowed to happen at all.

```bash
docker compose up -d
curl -s localhost:8000/health | python -m json.tool
```

That brings up the router, Redis, Redpanda (the Kafka API without a JVM),
ClickHouse and Prometheus. Stop any of them and the router keeps running.

### Decision latency

The JusTrust brief quotes a **100ms** budget. Until now nothing here measured
the decision itself — every other number is about whether the router chooses
*well*, not whether it chooses *fast*.

```bash
python bench_latency.py
```

| router | p50 | p95 | p99 | vs 100ms budget |
|---|---|---|---|---|
| static-weighted | 7.2µs | 11.5µs | 27.2µs | 3,676× |
| epsilon-greedy | 9.9µs | 36.1µs | 67.3µs | 1,486× |
| ucb1 | 12.6µs | 17.6µs | 40.0µs | 2,500× |
| **thompson** | **10.7µs** | **14.8µs** | **41.7µs** | **2,398×** |
| contextual-thompson | 13.9µs | 28.1µs | 57.1µs | 1,751× |

Percentiles rather than a mean, because a mean hides the tail and the tail is
what times out. Two caveats printed with the results rather than buried: this is
the *decision only* — network, TLS and the downstream gateway call dominate any
real end-to-end budget — and it is CPython on one core. The claim is not that
Python is fast enough at 350M/day; it is that **the algorithm costs
microseconds**, which makes a Go or Rust port a transport decision rather than
an algorithmic one.

### What a review of this layer turned up

Six defects, found by reading the code rather than by a failing test. Each now
has a regression test in `tests/test_service_hardening.py`.

| Defect | Why it mattered |
|---|---|
| Propensity computed in a **second pass**, costing 2.1x the decision itself — and measured *outside* the timed region | The latency reported to the caller understated a request by ~3x. `decide()` now serves both from one batched draw: **35% faster**, and honest |
| That draw counted the decision sample in its own propensity | Biased the chosen arm's propensity up by ~1/n. Importance weights divide by it, so every off-policy estimate built on those logs would skew — invisibly |
| Mutable state unguarded under FastAPI's **threadpool** | `_append_history` writes five lists; interleaved, they end at different lengths and a tick gets paired with someone else's outcome |
| `urllib.parse` used but never imported | Worked only because `urllib.request` imports it transitively |
| Expired constraints never left memory | `active()` filtered them from its *result*; the list grew with every constraint any peer ever wrote |
| `/investigate` ignored `CONSTRAINT_*` settings, and leaked a `TraceStore` per call | Configuration silently did nothing; a Langfuse background thread leaked per investigation |

The propensity-bias one is the one worth dwelling on. It would never have
surfaced as a crash or a failing assertion — it would have quietly shifted
every IPS and DR estimate in [Part 5](#part-5--would-this-survive-contact-with-production),
in a direction that flattered whichever actions the router already preferred.

One of the fixes was itself wrong on the first attempt: pruning constraints
against `max(expires_tick)` rather than the current tick kept only the
longest-lived one and discarded every other live constraint — strictly worse
than the leak it replaced. It was caught by a test whose fake Redis honoured
TTL; the first fake did not, and made a working fix look broken.

### The hot path holds no network calls

`POST /route` reads an in-process posterior and a locally refreshed constraint
cache, then hands the outcome event to a fire-and-forget pipeline. Nothing in it
can block on Redis, Kafka or ClickHouse.

That constraint drove the state design. Constraints go into Redis as `SETEX`
keys — low volume, natural TTL, visible to every replica, and a constraint that
outlives its usefulness expires without a sweeper. **Posteriors do not.** They
are read on every transaction, so a round trip per decision would be the entire
budget; they are snapshotted periodically instead. That is a write-back cache,
and it is the same shape as the in-house KvDB Juspay describes for this problem.

It also closes a limitation this README used to list: constraints and posteriors
now survive a restart.

### Events carry propensities

Every outcome event records **the probability the router assigned to its own
choice**. That is the field nobody logs until it is too late — without it,
[Part 5](#part-5--would-this-survive-contact-with-production)'s off-policy
evaluation cannot run at all, and the logs only measure the policy that produced
them. Kafka fans out; ClickHouse is the columnar store the estimators query.

### A typed constraint language

Constraints were ad-hoc Python dataclasses validated at runtime. `dsl/` moves
most of that into a type system:

```
avoid PG-Delta when issuer HDFC ttl 8h canary 2% confidence 0.70
```

```bash
routing-dsl check   constraints.route   # non-zero exit on the first bad line
routing-dsl compile constraints.route   # emits JSON the router consumes
```

`Scope` carries its own payload, so an issuer-scoped constraint *without* an
issuer cannot be constructed. `CanaryRate` is abstract with a smart constructor,
so a rate outside `[0, 1)` is not representable rather than merely rejected.
`check` exits non-zero, so it sits in CI as a gate the same way the eval gate
does — a constraint file that does not parse should fail the build, not reach
the router.

Haskell because the job is a parser with invariants, which is what the language
is for; `base` only, no megaparsec, so it builds with a bare GHC.

**Verification status:** the library and CLI compile cleanly under `-Wall` on
GHC 9.6 and were exercised end to end — the example file validates, the emitted
JSON round-trips into the Python `RoutingConstraint` type, and an unknown
gateway is rejected with a useful message. **The QuickCheck property suite in
`dsl/test/Spec.hs` has not been run.** Run `cabal test` before relying on it.

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

service/api.py             FastAPI: /route (hot path), /investigate (cold path)
service/state.py           Redis-backed constraints + posterior snapshots
service/events.py          Kafka and ClickHouse sinks, both fail-soft
service/metrics.py         Prometheus exposition format, written directly
service/demo.py            /simulate/*: drives real traffic through the live router
service/static/index.html  Walkthrough UI, one file, no build step
checkout/app.py            Merchant service on :8001, consumes the routing API
checkout/static/index.html Checkout page, one file, no build step
src/agent/langfuse_sink.py Langfuse backend: one trace tree per investigation
src/agent/prompt_source.py LangSmith prompts, falling back to the git registry
dsl/                       Haskell constraint DSL: parser, validator, CLI
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
bench_latency.py           routing decision latency, percentiles
Dockerfile                 multi-stage, non-root
docker-compose.yml         the whole stack locally
deploy/                    Prometheus scrape config and alert rules
.env.example               every environment variable, documented

docs/TRACE_QUERIES.md      SQL cookbook for the trace store
docs/DESIGN.md             router: what was tried, what failed, what is missing
docs/INVESTIGATION_AGENT.md  the agent: eval design, loop guarantees, memory
AGENTS.md                  conventions for AI coding agents working on this repo
```

## Testing

```bash
python -m pytest -q        # 232 tests
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
| `test_service.py` | The HTTP surface, metrics format, and that every backend is optional. |
| `test_observability.py` | The Langfuse trace tree, and that LangSmith fails open to the registry. |
| `test_service_hardening.py` | One test per defect found reviewing the service layer. |
| `test_demo.py` | The walkthrough endpoints, and that the UI only calls routes that exist. |
| `test_checkout.py` | The merchant against the real router, plus the router-is-down path. |
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
