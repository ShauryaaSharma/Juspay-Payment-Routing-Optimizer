"""Off-policy evaluation: what would a different router have earned?

The question that decides whether any of this is deployable. You have logs from
the policy currently in production. You have a candidate policy. You would like
to know whether the candidate is better *before* routing real payments through
it, because the alternative is an A/B test that costs declined transactions to
run.

Four estimators, in increasing order of how much they assume:

* **IPS** -- reweight logged rewards by ``pi_target / pi_logging``. Unbiased,
  and often uselessly noisy: one transaction with a small logging propensity
  can dominate the whole estimate.
* **SNIPS** -- self-normalised IPS. Divides by the summed weights instead of n.
  Slightly biased, dramatically lower variance, and it cannot return a value
  outside the range of observed rewards, which IPS can.
* **DM** -- fit a reward model from the logs, then average its predictions
  under the target policy. Low variance, biased by exactly however wrong the
  model is, and it fails silently where the logs are thin.
* **DR** -- doubly robust. DM plus an IPS correction on the model's residuals.
  Consistent if *either* the reward model or the propensities are right, which
  is why it is the default in practice.

## Why this module can prove it works

Off-policy evaluation is normally unfalsifiable in the moment: you produce an
estimate and find out years later whether it was any good. Here the environment
is a simulator, so the true value of a policy is computable in closed form --
no deployment, no Monte Carlo:

    V(pi) = (1/n) * sum_i sum_a pi(a | x_i) * true_sr[a, t_i, x_i]

That makes every estimate checkable against the answer, which is the only
honest way to claim an estimator works.

## The assumption that actually bites

Both the logging and target policies here are **fixed** during evaluation. That
is not a simplification, it is a requirement. Importance sampling asks "what if
this same decision had been made differently"; it cannot answer "what if a
*learning* policy had seen different data and therefore been in a different
state at every subsequent step". Evaluating an adaptive learner offline needs
counterfactual histories, which logs do not contain. Freeze the candidate,
evaluate the frozen thing, then deploy it and let it learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class LoggedData:
    """Bandit feedback from a deployed policy.

    ``propensity`` is the part nobody logs until it is too late. Without the
    probability the logging policy assigned to the action it took, none of the
    estimators below are computable, and the logs are only good for measuring
    the policy that produced them.
    """

    context: np.ndarray  # (n,) issuer index -- the covariate
    action: np.ndarray  # (n,) gateway index -- what was actually routed
    reward: np.ndarray  # (n,) 1.0 on success
    propensity: np.ndarray  # (n,) pi_logging(action | context) at decision time
    tick: np.ndarray  # (n,) simulated minute, for the analytic ground truth
    n_actions: int
    n_contexts: int

    def __post_init__(self) -> None:
        sizes = {len(self.context), len(self.action), len(self.reward),
                 len(self.propensity), len(self.tick)}
        if len(sizes) != 1:
            raise ValueError(f"logged arrays have mismatched lengths: {sizes}")
        if np.any(self.propensity <= 0):
            raise ValueError(
                "a logged action has propensity <= 0, which makes importance "
                "weights infinite. The logging policy must assign positive "
                "probability to every action it actually takes."
            )

    @property
    def n(self) -> int:
        return len(self.action)

    @property
    def observed_value(self) -> float:
        """Mean reward actually earned by the logging policy."""
        return float(self.reward.mean())


@dataclass
class Estimate:
    """One estimator's output, with the diagnostics needed to trust it."""

    estimator: str
    value: float
    std_err: float
    ess: float = float("nan")  # effective sample size
    max_weight: float = float("nan")
    clipped_fraction: float = 0.0
    off_support_fraction: float = 0.0
    notes: str = ""

    def error_vs(self, truth: float) -> float:
        return self.value - truth

    def covers(self, truth: float, z: float = 1.96) -> bool:
        """Does the 95% interval contain the true value?"""
        if not np.isfinite(self.std_err):
            return False
        return abs(self.value - truth) <= z * self.std_err


def _weights(logged: LoggedData, target: np.ndarray) -> np.ndarray:
    chosen = target[np.arange(logged.n), logged.action]
    return chosen / logged.propensity


def _diagnostics(weights: np.ndarray, logged: LoggedData, target: np.ndarray) -> dict[str, float]:
    """Effective sample size and support overlap.

    ESS is the number that tells you whether an estimate means anything. With
    n=40,000 logged transactions but an ESS of 200, the estimate rests on 200
    effective observations and its confidence interval is a fiction.
    """
    total = weights.sum()
    ess = float(total**2 / np.sum(weights**2)) if total > 0 else 0.0
    # Rows where the target wants an action the logger never took in that
    # context. Those rows are pure extrapolation -- no amount of data can
    # inform them, and every estimator is guessing there.
    support = _empirical_support(logged, target.shape[1])
    off_support = (target > 0.01) & (support == 0.0)
    return {
        "ess": ess,
        "max_weight": float(weights.max()) if weights.size else float("nan"),
        "off_support_fraction": float(off_support.any(axis=1).mean()),
    }


def _empirical_support(logged: LoggedData, n_actions: int) -> np.ndarray:
    """Per-context-row indicator of which actions the logger ever took."""
    seen = np.zeros((logged.n_contexts, n_actions))
    np.add.at(seen, (logged.context, logged.action), 1.0)
    return (seen > 0).astype(float)[logged.context]


def ips(logged: LoggedData, target: np.ndarray, clip: float | None = None) -> Estimate:
    """Inverse propensity scoring. Unbiased, high variance."""
    weights = _weights(logged, target)
    clipped = 0.0
    if clip is not None:
        clipped = float((weights > clip).mean())
        weights = np.minimum(weights, clip)
    terms = weights * logged.reward
    diag = _diagnostics(weights, logged, target)
    return Estimate(
        estimator="IPS" + ("" if clip is None else f" (clip={clip:g})"),
        value=float(terms.mean()),
        std_err=float(terms.std(ddof=1) / np.sqrt(logged.n)),
        clipped_fraction=clipped,
        **diag,
    )


def snips(logged: LoggedData, target: np.ndarray, clip: float | None = None) -> Estimate:
    """Self-normalised IPS. Biased, far lower variance, range-respecting."""
    weights = _weights(logged, target)
    clipped = 0.0
    if clip is not None:
        clipped = float((weights > clip).mean())
        weights = np.minimum(weights, clip)
    total = weights.sum()
    if total <= 0:
        return Estimate("SNIPS", float("nan"), float("nan"), notes="no weight mass")
    value = float((weights * logged.reward).sum() / total)
    # Delta-method standard error for a ratio estimator.
    residual = weights * (logged.reward - value)
    std_err = float(np.sqrt((residual**2).sum()) / total)
    diag = _diagnostics(weights, logged, target)
    return Estimate(
        estimator="SNIPS" + ("" if clip is None else f" (clip={clip:g})"),
        value=value, std_err=std_err, clipped_fraction=clipped, **diag,
    )


def fit_reward_model(logged: LoggedData, rows: np.ndarray | None = None) -> np.ndarray:
    """Empirical mean reward per (context, action) cell.

    Non-parametric on purpose: the context here is a five-valued categorical, so
    a cell mean *is* the correct model and there is nothing to be gained by
    fitting something fancier that could also be wrong in interesting ways.
    Cells with no data fall back to the per-action mean, then the global mean.
    """
    idx = np.arange(logged.n) if rows is None else rows
    totals = np.zeros((logged.n_contexts, logged.n_actions))
    counts = np.zeros((logged.n_contexts, logged.n_actions))
    np.add.at(totals, (logged.context[idx], logged.action[idx]), logged.reward[idx])
    np.add.at(counts, (logged.context[idx], logged.action[idx]), 1.0)

    per_action_total = totals.sum(axis=0)
    per_action_count = counts.sum(axis=0)
    global_mean = float(logged.reward[idx].mean()) if idx.size else 0.0
    per_action = np.where(
        per_action_count > 0, per_action_total / np.maximum(per_action_count, 1), global_mean
    )
    model = np.where(counts > 0, totals / np.maximum(counts, 1), per_action[None, :])
    return model


def _cross_fit(logged: LoggedData, folds: int = 2, seed: int = 0) -> np.ndarray:
    """Per-row reward-model predictions, fitted out-of-fold.

    Fitting the reward model on the same rows it is then evaluated on biases DM
    and DR toward the logging policy's own choices -- the model has memorised
    the actions it is being asked to score. Cross-fitting costs three lines and
    removes it.
    """
    rng = np.random.default_rng(seed)
    assignment = rng.integers(folds, size=logged.n)
    predictions = np.zeros((logged.n, logged.n_actions))
    for fold in range(folds):
        train = np.flatnonzero(assignment != fold)
        test = np.flatnonzero(assignment == fold)
        if train.size == 0 or test.size == 0:
            continue
        model = fit_reward_model(logged, train)
        predictions[test] = model[logged.context[test]]
    return predictions


def direct_method(logged: LoggedData, target: np.ndarray, folds: int = 2) -> Estimate:
    """Average the fitted reward model's predictions under the target policy."""
    predictions = _cross_fit(logged, folds)
    terms = np.sum(target * predictions, axis=1)
    return Estimate(
        estimator="DM",
        value=float(terms.mean()),
        std_err=float(terms.std(ddof=1) / np.sqrt(logged.n)),
        notes="variance reflects context spread only, not model error",
    )


def doubly_robust(
    logged: LoggedData, target: np.ndarray, clip: float | None = None, folds: int = 2
) -> Estimate:
    """DM plus an importance-weighted correction on the model's residuals."""
    predictions = _cross_fit(logged, folds)
    baseline = np.sum(target * predictions, axis=1)
    weights = _weights(logged, target)
    clipped = 0.0
    if clip is not None:
        clipped = float((weights > clip).mean())
        weights = np.minimum(weights, clip)
    residual = logged.reward - predictions[np.arange(logged.n), logged.action]
    terms = baseline + weights * residual
    diag = _diagnostics(weights, logged, target)
    return Estimate(
        estimator="DR" + ("" if clip is None else f" (clip={clip:g})"),
        value=float(terms.mean()),
        std_err=float(terms.std(ddof=1) / np.sqrt(logged.n)),
        clipped_fraction=clipped,
        **diag,
    )


def true_value(logged: LoggedData, target: np.ndarray, sr: np.ndarray) -> float:
    """Exact policy value, using ground truth only the simulator has.

    ``sr`` is (n_actions, n_ticks, n_contexts). This is what the estimators are
    trying to recover, and it is why this module can be validated rather than
    merely believed.
    """
    truth = sr[:, logged.tick, logged.context].T  # (n, n_actions)
    return float(np.sum(target * truth, axis=1).mean())


def evaluate_all(
    logged: LoggedData,
    target: np.ndarray,
    sr: np.ndarray | None = None,
    clip: float | None = 20.0,
) -> tuple[list[Estimate], float | None]:
    """Run every estimator, plus the analytic truth when it is available."""
    estimates = [
        ips(logged, target),
        ips(logged, target, clip=clip),
        snips(logged, target),
        direct_method(logged, target),
        doubly_robust(logged, target, clip=clip),
    ]
    truth = None if sr is None else true_value(logged, target, sr)
    return estimates, truth
