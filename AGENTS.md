# AGENTS.md

Conventions for AI coding agents working in this repository.

> **Not to be confused with `docs/INVESTIGATION_AGENT.md`**, which documents the
> LLM investigation agent that this project *builds*. This file is instructions
> *to* an agent; that one is documentation *of* one.

---

## What this project is

An adaptive payment router (a non-stationary multi-armed bandit), an LLM agent
that diagnoses gateway failures from telemetry, and the loop between them
closed — a confirmed diagnosis becomes a routing constraint.

Three layers, in dependency order:

| layer | code | what it does |
|---|---|---|
| Router | `src/gateways.py`, `src/routers/`, `src/simulator.py` | Simulates a gateway fleet; bandit strategies route transactions. |
| Agent | `src/agent/` | Investigates telemetry with tools, returns a structured diagnosis. |
| Loop | `src/agent/constraints.py`, `src/closed_loop.py` | Diagnosis → routing constraint → next transaction. |

The README quotes specific measured numbers. **Those numbers are claims.** Most
of the invariants below exist to stop a refactor from silently invalidating
them.

---

## Commands

```bash
python -m pytest -q          # 158 tests, ~30s. Run this before claiming anything.
python benchmark.py          # router comparison. SLOW (~2-3 min).
python sweep_gamma.py        # discount-factor study. SLOW (~2 min).
python run_evals.py          # agent eval suite, all arms. Fast after first run.
python run_evals.py --gate   # exit 1 on regression vs results/eval_baseline.json
python close_loop.py         # closed-loop measurement
python contextual_vs_agent.py  # contextual bandit vs agent. SLOW (~3 min).
python run_ope.py            # off-policy evaluation. SLOW (~2 min).
python run_investigation.py --case issuer_outage   # single investigation, full trace
python manage_traces.py config                     # resolved settings, secrets redacted
python manage_traces.py tail                       # recent traces
```

Nothing needs an API key. `--live` flags switch to `claude-opus-5`; without
credentials they fail fast with a clear message rather than producing zeros.

**First `run_evals.py` on a clean checkout takes ~100s** generating six weeks of
telemetry, then caches to `results/eval_cache/` and takes <1s thereafter. Do not
"optimise" this by shrinking the case volumes — see invariant 3.

---

## Invariants

Breaking any of these makes the project's published results wrong. Each has a
test; if you find yourself editing the test to make your change pass, stop.

### 1. Unconstrained router selection must stay RNG-identical

`Router.select(tick, context=None, blocked=frozenset())` gained two parameters
when the constraint layer landed. Every implementation **early-returns on the
original code path when `blocked` is empty**, so random-number consumption is
byte-identical to before those parameters existed.

The benchmark numbers in the README were measured before constraints existed. If
you change RNG consumption on the unblocked path, every number becomes a lie.

Guarded by `test_unconstrained_selection_is_unchanged_by_the_new_parameters`.
Spot-check with:

```bash
python -c "import sys;sys.path.insert(0,'.');from src.simulator import run_once,SimConfig;from src.gateways import default_fleet;from src.routers import ThompsonRouter;r=run_once(lambda n,s:ThompsonRouter(n,gamma=0.999,seed=s),default_fleet(),SimConfig(transactions=50000),0);print(r.realised_sr,r.total_regret)"
# must print 0.9492 532.5...
```

### 2. Bump `CACHE_VERSION` when telemetry generation changes

`src/agent/evals/cases.py` caches simulated telemetry to `results/eval_cache/`.
If you change fleet definitions, incident parameters, transaction volume, the
router used to generate telemetry, or the seed — **bump `CACHE_VERSION`**.
Otherwise stale telemetry is graded against new ground truth and the eval
silently reports nonsense.

### 3. The eval must keep headroom

The deterministic baseline policy currently scores **75% (6/8)**. It must stay
strictly between 50% and 100%.

This is not arbitrary. The eval originally had six cases and the baseline scored
**100%** — a saturated eval that could not distinguish any two arms, because
every case was solvable by "find the lowest success rate", which is exactly what
the heuristic encodes. Two cases (`diurnal_trough`, `fleet_wide_partial`) were
added to invert a threshold rule.

If `test_baseline_policy_scores_below_ceiling_and_above_floor` fails at 1.0, the
eval has lost its discriminating power. **Add harder cases; do not relax the
test.**

### 4. Prompts are append-only versions

`src/agent/prompts.py` holds immutable named versions. Never edit `V1` or `V2`
in place — add `V3` and change `DEFAULT_VERSION`. `v1-baseline` is deliberately
naive and is the control arm; "improving" it destroys the ability to show that
prompt work bought anything.

### 5. System prompts stay static strings

No `datetime.now()`, no UUIDs, no per-incident interpolation in any `system`
text or tool definition. Tools render before system, which renders before
messages, so any churn invalidates the prompt cache prefix — and nothing fails
loudly enough to notice. Volatile content goes in the user message via
`render_incident_brief`.

### 6. The agent must never see ground truth

`TelemetryStore` exposes only *observable* aggregates — outcomes of transactions
that were actually routed. It must never expose `GatewaySpec.true_sr`, which the
simulator knows and a production system could not. If the agent can read true
success rates, the eval measures nothing.

### 7. No lookahead in the closed loop

`run_closed_loop` hands the agent a `_partial_result` truncated at the detection
index. The agent must never observe a transaction that had not happened yet.
Guarded by `test_agent_sees_no_data_from_after_the_detection_point`.

### 8. Config defaults must reproduce the measured behaviour

Every default in `src/agent/config.py` equals the hardcoded value it replaced.
If an unset environment variable changed a default, the README's numbers would
depend on the shell they were run in. When you add a setting, its default must
be the current behaviour, and `test_defaults_reproduce_the_measured_behaviour`
must cover it.

### 9. Secrets never leave the environment

Credentials are read from env (or `.env`, which is gitignored) and must never
be written to a trace, printed, or committed. Anywhere a DSN is surfaced it
goes through `redact_dsn`. `Settings.describe()` exists precisely so a run can
log its own config safely. Three tests assert a known password never appears in
`describe()`, `TraceStore.target`, or `stats()`.

### 10. Tracing fails soft, always

`TraceStore.write` catches everything, counts the failure, and warns once. An
investigation must complete normally with a broken trace backend — observability
failing is an annoyance, observability taking down incident response is an
outage. `TRACE_CONNECT_TIMEOUT` (default 5s) bounds a hung database. Never let
a trace write raise, and never make a run depend on one succeeding.

### 11. `from_diagnosis` refusals are safety-critical

This function is the gate between "a language model said something" and "the
system moved production traffic". It returns `None` for: low confidence,
`fleet_wide`/`no_incident` scopes, issuer-scoped findings with no issuer named,
and any gateway not in the fleet. That last one is the final barrier against a
hallucinated gateway name.

Constraints must also always (a) leak `canary_rate` traffic so a blocked gateway
stays observable, (b) carry a TTL, and (c) never block every gateway at once.
Most of `test_constraints.py` tests what this layer must **refuse** to do.

---

## Dependencies

**numpy is the only required dependency.** `anthropic` and `psycopg` are both
optional and imported lazily at their point of use — without `anthropic` the
agent runs the baseline policy, without `psycopg` traces fall back to JSONL.

Do not add: `matplotlib` (there is a dependency-free SVG writer in
`src/plotting.py`), `pydantic` (plain dataclasses in `src/agent/schemas.py`,
with the JSON Schema written out explicitly), `pandas`, or an embedding library
for memory retrieval (lexical BM25-ish scoring in `src/agent/memory.py` is
deliberate — it keeps "does memory help" from being confounded with "is this
embedding model good").

The offline path must run on a machine with no API key and no SDK. That is what
makes the eval suite CI-able.

---

## Code conventions

Match what is already there:

- **Comments explain *why*, and name rejected alternatives.** The valuable
  comments in this codebase record what was tried and failed — see the
  `max(posterior means)` bias explanation in `src/routers/pid_thompson.py`. When
  you discard an approach, write down why.
- **Module docstrings carry design rationale**, not a restatement of the code.
- **Test names are behavioural sentences**:
  `test_undiscounted_router_gets_stuck_after_a_regime_change`, not `test_ucb1_2`.
- **Tests document known interactions**, including surprising ones. See
  `test_identical_failing_call_is_deduplicated_rather_than_counted`.
- Type hints throughout; `from __future__ import annotations` at the top.
- Prefer composition over new strategy classes — `ConstrainedRouter` wraps any
  router rather than duplicating five of them.

---

## Claude API conventions

If you touch `src/agent/llm.py`:

- Model is **`claude-opus-5`**. Do not downgrade for cost without being asked.
- `thinking={"type": "adaptive"}`. **Never** `budget_tokens` — it is removed on
  this model family and returns a 400.
- `output_config={"effort": ..., "format": {"type": "json_schema", ...}}` pins
  the final answer to `DIAGNOSIS_SCHEMA`, so the loop never salvages JSON from
  prose.
- `cache_control` on the system block; tools and system are byte-stable across
  an investigation.
- Tools are `strict: true` with `additionalProperties: false`.
- Check `stop_reason == "refusal"` **before** reading `content` — a refusal is
  HTTP 200 with no usable content.
- Parse tool inputs with `json.loads`, never string matching.

The agent loop is **hand-written on purpose** (`src/agent/loop.py`) rather than
using the SDK's `tool_runner`. Owning budgets, termination, error recovery, and
duplicate suppression is the point of that module. Do not replace it with the
runner.

---

## A finding that must not be quietly reversed

`contextual_vs_agent.py` measures the contextual bandit **beating** the
closed-loop agent at every volume tested. That is a negative result for the
agent layer and it is reported as one, in the module docstring and in README
Part 4. Do not soften it, and do not tune the scenario until the agent wins.

If you change the comparison and the ordering flips, establish *why* before
publishing it -- the mechanism is documented (detection lag, plus a contaminated
flat posterior that a constraint cannot repair), so a reversal should come with
an explanation of which of those changed.

### 12. Propensity computation must never touch the router's own RNG

`Router.action_probabilities` takes a caller-supplied `rng`. Sampling from
`self.rng` to compute a propensity would change the sequence of routing
decisions -- the act of measuring would alter what is measured, and every
benchmark number with it. Guarded by
`test_propensity_computation_does_not_disturb_routing`.

Related: off-policy evaluation requires a **stochastic** logging policy. UCB1's
propensities are one-hot once its counts diverge, which makes importance
weights undefined. Do not make a deterministic router the default logger.

## Known gotchas

- **Duplicate suppression pre-empts the error circuit breaker.** If the model
  repeats the *same* failing tool call, it gets a nudge instead of a re-run, so
  the consecutive-error counter resets and `tool_error_limit` never fires. The
  loop still terminates via the step budget. Both paths are bounded; this is
  documented, not a bug.
- **The closed loop's precision depends on traffic volume.** Below ~80k
  transactions/2 days the issuer segments fall under the sampling floor and the
  agent returns a blunt `single_gateway` constraint instead of a scoped one. The
  crossover is noisy between ~80k and ~120k, so tests pin only 40k and 150k.
  Do not add a test that asserts a specific diagnosis in that band.
- **`MIN_SAMPLES_FOR_SR = 30`** in `src/agent/telemetry.py` makes small-sample
  rates return `None` rather than noise. Tools returning `null` is correct
  behaviour, not a bug to paper over.
- **Windows/Git Bash**: heredocs with nested quotes break frequently in this
  environment. Prefer the Write/Edit tools over `cat > file <<'EOF'` for any
  file containing apostrophes or mixed quoting.
- **`str.replace` fails silently.** When patching files with a script, assert
  the search string is present first. A no-match otherwise looks like success
  and the edit is quietly lost.
- **`psycopg` is installed in this environment.** A test using a real DSN will
  attempt an actual TCP connection and can hang the suite. Stub `_connect`
  instead of relying on the driver being absent.
- **Tests that mutate env** must use the `clean_env` fixture in
  `tests/test_traces.py` and `reset_settings()` (not `settings(refresh=True)`)
  in teardown — refresh re-parses and raises if the env is deliberately invalid.

---

## CI

`.github/workflows/ci.yml` runs the suite on Python 3.11-3.13 and then the eval
gate. Two things to know before you change it:

- **`results/eval_baseline.json` must stay committed.** `.gitignore` excludes
  `results/*.json` and then re-includes this one specifically. Without it in the
  repo the gate has nothing to compare against and silently passes.
- **The telemetry cache key** hashes every file that determines simulated
  telemetry. If you add such a file, add it to the `hashFiles(...)` list, or CI
  will grade new ground truth against cached old telemetry.

The workflow targets Python 3.11, so avoid 3.12-only syntax: nested same-quote
f-strings, backslashes inside f-string expressions, `type` alias statements,
`itertools.batched`, and `datetime.UTC` (use `timezone.utc`).

## Before you claim done

1. `python -m pytest -q` — all 99 pass.
2. If you touched `src/routers/`, `src/gateways.py`, or `src/simulator.py`, run
   the RNG spot-check in invariant 1.
3. If you touched anything under `src/agent/`, run `python run_evals.py --gate`.
   If you added a setting, add it to `.env.example` with its default and a
   one-line reason.
6. If you added a trace column or an eval case, `docs/TRACE_QUERIES.md` has
   tests asserting it stays in sync — they will tell you.
4. If you changed telemetry generation, bump `CACHE_VERSION` first.
5. If you changed a number the README quotes, re-run the command that produced
   it and update the README. Do not leave stale numbers.

---

## Reporting discipline

This project's credibility rests on its numbers being real, and several of its
most interesting findings are negative results:

- The PID controller **does not** beat a correctly tuned discount factor.
- The first eval design was **useless** and said so.
- The stale-constraint cost measured **~0 bps**, and the docs say why rather
  than asserting a cost the data does not show.

Keep this standard:

- **Never report an unmeasured result.** The README has a "What is measured, and
  what is not" section. Anything requiring an API key is currently unmeasured —
  say so rather than implying otherwise.
- **Do not tune a benchmark until it flatters the method.** If a result is
  unflattering, report it and explain the mechanism.
- If a measurement contradicts a claim you already wrote, change the claim.
