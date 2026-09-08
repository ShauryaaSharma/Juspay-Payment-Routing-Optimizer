# Design notes

Reasoning, rejected alternatives, and known gaps. The README says what the
system does; this says why it is shaped that way.

## 1. Why a bandit and not a forecasting model

The obvious alternative is to forecast each gateway's success rate from
historical data and route to the argmax. It fails for a structural reason: the
router's own decisions determine what data it collects. Send no traffic to a
gateway and you learn nothing about it — including the fact that it recovered.
A supervised model trained on routed traffic is trained on a sample its own
predecessor selected, and it will confidently recommend the gateway it already
prefers.

Bandits address exactly this feedback loop. The exploration term is not a
nuisance; it is what keeps the dataset informative.

## 2. Why the environment is built the way it is

A benchmark is only as honest as the environment it runs in, and there are two
easy ways to build one that flatters your method:

- **A stationary environment.** Every algorithm converges, differences shrink
  to noise, and the result says nothing about production.
- **A single dominant arm.** If one gateway is always best, "route everything
  to gateway 3" wins and no adaptation is needed.

So the fleet is built to make leadership change hands at least three times
across the simulated week — asserted by a test, not left to chance. Three
distinct non-stationarities are present, because they stress different things:

| mechanism | what it breaks |
|---|---|
| diurnal drift (`PG-Charlie`) | slow adaptation; punishes long memory |
| hard outage (`PG-Bravo`) | detection latency; punishes confident posteriors |
| warm-up (`PG-Echo`) | premature convergence; punishes greedy strategies |

`PG-Echo` is the sharpest of the three. It is the best gateway overall but
performs poorly for the first simulated day. Any router that stops exploring
early will write it off permanently and lose several hundred bps for the rest
of the week.

## 3. Why discounting lives in the shared base class

`DiscountedCounts` sits in `routers/base.py` rather than inside any one
strategy, because the choice of *how to remember* turned out to dominate the
choice of *which strategy*. Discounted UCB1 beats undiscounted Thompson;
discounted Thompson beats both. The exploration rule matters less than whether
the router can forget.

Decay is applied to *all* arms on every update, not just the played one. This
is deliberate: unplayed arms drift back toward the prior, which re-opens them
for exploration after a long absence. Decaying only the played arm would let a
gateway that was written off during an outage stay written off forever, which
is precisely the `PG-Echo` failure.

## 4. The control loop, and the bug that shaped it

The first implementation set the controller's setpoint to
`max(posterior means)` — "if my best arm claims 96% and I am realising 71%, my
model is stale." It is intuitive and it is wrong.

`max` over several noisy estimates is a biased estimator of the true maximum:
it sits above it, always. So the error term carried a permanent positive bias,
the integrator wound up against it, and the controller sat at inflation ≈ 4
forever — a router that over-explores permanently and pays for it in
conversion. Measured median inflation was 4.15 where it should have been 1.0.

The fix reframes the error as a **calibration residual**. At selection time the
router records what it predicted for the arm it chose; on update it compares
that against the realised outcome. Averaged over transactions the difference is
zero whenever the posterior is calibrated, regardless of which arms were
chosen or how many there are. That makes the controller idle at 1.0 and cost
nothing until the model is genuinely wrong.

Two properties follow from choosing an unbiased residual, and both are worth
more than the performance number:

- The loop is **silent during network-wide degradation**. If every gateway
  drops together, predictions fall with outcomes, the residual stays near zero,
  and no exploration is triggered. A target-SR controller would do the opposite
  — spraying traffic across a failing network at exactly the wrong moment.
- The loop is **self-tuning across merchants**. There is no per-merchant
  constant to configure, because the setpoint is derived from the router's own
  forecasts.

Other implementation details that matter:

- Inflation divides both Beta parameters, preserving the posterior mean while
  widening variance. It adds doubt without biasing the estimate.
- Output is clamped to `[1, 25]` and the integrator to `±2`. The controller can
  only ever *add* doubt — a negative output would sharpen the posterior beyond
  what evidence supports, which is how a router talks itself into a stale arm.
- The loop is held open for the first 50 observations so cold-start noise does
  not slam the integrator.

## 5. What the benchmark actually showed

The controller did **not** beat a correctly tuned discount factor. Reporting
that plainly is the point of the exercise.

What it did show is that the correct γ is not a constant: it moves with
transaction volume, because γ forgets per observation while gateways degrade
per unit time. A merchant doing 20k transactions/week and one doing 100k want
different values (0.999 and 0.9999 respectively), and a fleet-wide constant is
therefore wrong for nearly everyone on it. Getting it wrong costs 50–170 bps.

That reframes what the controller is for. It is not a better bandit — it is a
way to avoid a per-merchant tuning problem that scales with merchant count. It
recovers ~60% of the tuning gap while requiring no tuning, and it beats a
*badly* set γ at both volumes tested.

The honest summary: **if you can tune γ per merchant and keep it tuned as
volume changes, do that. If you cannot, the controller is the better default.**

## 6. Rejected alternatives

- **Sliding window instead of geometric discounting.** Equivalent in effect,
  but needs O(window) memory per gateway. Geometric decay is O(1) per arm and
  matters at 350M transactions/day.
- **Change-point detection (CUSUM / Bayesian online).** Sharper at detecting
  the `PG-Bravo` outage, but requires a detection threshold — which is the same
  per-merchant tuning problem the controller exists to avoid.
- **Contextual bandit (LinUCB / neural).** The correct production answer, and
  the natural next step. Skipped here because a flat bandit isolates the
  control-loop question; adding features would confound "the controller helps"
  with "the features help".
- **Optimising a cost-adjusted objective.** Cost per transaction is tracked and
  reported but not optimised, so the SR comparison stays single-objective and
  legible. Note the adaptive routers all drift toward more expensive gateways
  (~19–20 bps vs 16.79 for static) — they are buying conversion with fees, and
  a production system must price that tradeoff explicitly.

## 7. What I would build next, in order

1. **Contextual routing.** Condition on issuer bank, card network, amount band.
   Issuer-level degradation is the failure mode a flat router cannot see at
   all: a gateway can be healthy overall while failing every transaction from
   one bank.
2. **Cost-aware objective.** `maximise(SR) - λ·cost - μ·latency`, with λ
   negotiated per merchant.
3. **Retry policy as part of the decision.** A failed transaction is not
   terminal — the second choice matters as much as the first, and the two
   decisions are not independent.
4. **Shadow evaluation.** Off-policy estimation (inverse propensity scoring) so
   a new policy can be assessed on logged traffic before it touches live
   payments. This is what makes any of it deployable.
5. **State durability.** Posteriors are in-process today. A production router
   needs them to survive a restart without a cold start, and to stay coherent
   across replicas — the point where this stops being an algorithms problem and
   becomes a distributed-systems one.
