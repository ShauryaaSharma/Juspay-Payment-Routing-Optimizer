"""Thompson sampling with a PID outer loop on realised success rate.

This is the strategy the whole benchmark exists to test. The inner loop is
ordinary discounted Thompson sampling. The outer loop is a controller that
watches for a specific failure signature and responds by widening the
router's own uncertainty.

The signature: the router's best arm *claims* it converts at 96%, but
realised traffic is converting at 71%. That gap means the posterior is stale
-- the world moved and the evidence has not caught up yet. A fixed discount
factor eventually notices, but "eventually" is measured in thousands of
failed transactions. The controller notices in tens.

The error signal is a *calibration residual*, not a gap to a fixed target:
at selection time the router predicts the chosen arm's success probability,
and the outcome either confirms it or does not. Averaged, the difference is
zero whenever the posterior is well calibrated -- so the controller idles at
inflation 1.0 and costs nothing until the model is actually wrong.

Two setpoints that look reasonable and are not:

* A constant target SR (say 0.94) needs hand-tuning per merchant and per
  gateway mix, and it cannot distinguish "our model is stale" from "this
  merchant's traffic is just harder".
* ``max(posterior means)`` is the maximum of several noisy estimates, so it
  sits above the true best arm. That bias never washes out; the integrator
  winds up against it and the router over-explores permanently.

Because the residual is unbiased, the controller also stays quiet during
network-wide degradation -- when *every* gateway is down, predictions fall
with outcomes, the residual stays near zero, and spreading traffic around
would only make things worse.

What this buys you, measured (see ``sweep_gamma.py``): the controller is a
*substitute* for tuning the discount factor, not a complement to it. Against
an undiscounted posterior it is worth roughly +80 bps of success rate; against
a correctly-tuned ``gamma`` it is worth nothing, and stacking both slightly
over-explores. The catch is that the correct ``gamma`` depends on transaction
volume -- 0.999 at 20k/week, 0.9999 at 100k/week -- so a fleet-wide constant
is wrong for most merchants on it. The controller adapts to that mismatch at
run time instead of asking an operator to re-tune per merchant. Hence the
``gamma=1.0`` default here: the loop is doing the forgetting.
"""

from __future__ import annotations

import numpy as np

from ..gateways import Outcome, RoutingContext
from .thompson import ThompsonRouter


class PIDThompsonRouter(ThompsonRouter):
    name = "pid-thompson"

    def __init__(
        self,
        n_gateways: int,
        gamma: float = 1.0,
        kp: float = 25.0,
        ki: float = 0.8,
        kd: float = 5.0,
        ewma_alpha: float = 0.02,
        max_inflation: float = 25.0,
        integral_clamp: float = 2.0,
        seed: int = 0,
    ) -> None:
        super().__init__(n_gateways, gamma=gamma, seed=seed)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ewma_alpha = ewma_alpha
        self.max_inflation = max_inflation
        self.integral_clamp = integral_clamp

        self.realised_sr = float("nan")  # EWMA of actual outcomes
        self.predicted_sr = float("nan")  # EWMA of what we expected to get
        self._pending_prediction = float("nan")
        self.integral = 0.0
        self.prev_error = 0.0
        self.inflation = 1.0
        self.observations = 0

    def select(
        self,
        tick: int,
        context: RoutingContext | None = None,
        blocked: frozenset[int] = frozenset(),
    ) -> int:
        samples = self._sample(self.inflation)
        if blocked:
            samples[list(blocked)] = -np.inf
        gateway = int(np.argmax(samples))
        # Record the prediction now so update() can score it against reality.
        self._pending_prediction = float(self.counts.mean[gateway])
        return gateway

    def update(self, outcome: Outcome) -> None:
        super().update(outcome)
        self.observations += 1

        observed = 1.0 if outcome.success else 0.0
        predicted = self._pending_prediction
        if np.isnan(predicted):
            predicted = observed
        a = self.ewma_alpha
        if np.isnan(self.realised_sr):
            self.realised_sr, self.predicted_sr = observed, predicted
        else:
            self.realised_sr = (1 - a) * self.realised_sr + a * observed
            self.predicted_sr = (1 - a) * self.predicted_sr + a * predicted

        # Hold the loop open until the EWMAs carry enough evidence to mean
        # anything; otherwise cold-start noise slams the integrator.
        if self.observations < 50:
            return

        # > 0 means the fleet is underdelivering against its own forecast.
        error = self.predicted_sr - self.realised_sr

        self.integral = float(
            np.clip(self.integral + error, -self.integral_clamp, self.integral_clamp)
        )
        derivative = error - self.prev_error
        self.prev_error = error

        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        # Only ever *add* doubt: a negative u would mean sharpening the
        # posterior beyond what the evidence supports, which is how a router
        # talks itself into a stale arm.
        self.inflation = float(np.clip(1.0 + u, 1.0, self.max_inflation))

    def controller_state(self) -> dict[str, float]:
        """Exposed so the benchmark can plot what the controller was thinking."""
        return {
            "realised_sr": self.realised_sr,
            "predicted_sr": self.predicted_sr,
            "error": self.prev_error,
            "integral": self.integral,
            "inflation": self.inflation,
        }
