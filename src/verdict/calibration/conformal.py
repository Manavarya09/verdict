"""Split conformal prediction for classification.

Given calibrated probabilities on a held-out set, learn a threshold such that the
*prediction set* contains the true label with probability >= 1 - alpha on new data
from the same distribution. A set of size 1 is a confident answer; a bigger set means
"abstain / escalate".

Two scores are available:

* ``lac`` (default) — Least Ambiguous set-valued Classifier (Sadinle et al. 2019):
  score = 1 - p(true label). Produces the smallest average sets. Set = {labels with
  p >= 1 - qhat}.
* ``aps`` — Adaptive Prediction Sets (Romano et al. 2020): score = cumulative mass up
  to the true label. Better conditional coverage on hard examples, larger sets.

Distribution-free: the marginal coverage guarantee holds for any model and any data,
as long as calibration and test points are exchangeable."""

from __future__ import annotations

from typing import Literal

import numpy as np

Method = Literal["lac", "aps"]


class ConformalCalibrator:
    def __init__(self, alpha: float = 0.1, qhat: float | None = None, method: Method = "lac"):
        self.alpha = float(alpha)
        self.qhat = qhat
        self.method: Method = method
        self.n_calib = 0

    def _scores(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(y)
        if self.method == "lac":
            return 1.0 - probs[np.arange(n), y]
        order = np.argsort(-probs, axis=1)
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        pos = np.argmax(order == y[:, None], axis=1)
        return cum[np.arange(n), pos]

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> ConformalCalibrator:
        probs = np.asarray(probs, dtype=np.float64)
        y = np.asarray(labels)
        n = len(y)
        if n < 20:
            raise ValueError("need at least 20 calibration examples for a meaningful threshold")
        scores = self._scores(probs, y)
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat = float(np.quantile(scores, q_level, method="higher"))
        self.n_calib = n
        return self

    def prediction_set(self, probs: np.ndarray) -> list[list[int]]:
        """Indices, in descending probability order, that form the prediction set.
        Never empty: the top-1 label is always included."""
        P = np.atleast_2d(np.asarray(probs, dtype=np.float64))
        if self.qhat is None:
            return [[int(np.argmax(p))] for p in P]
        out: list[list[int]] = []
        for p in P:
            order = np.argsort(-p)
            if self.method == "lac":
                keep = [int(i) for i in order if p[i] >= 1.0 - self.qhat]
            else:
                cum = np.cumsum(p[order])
                k = int(np.searchsorted(cum, self.qhat, side="left")) + 1
                keep = [int(i) for i in order[: min(k, len(p))]]
            out.append(keep or [int(order[0])])
        return out

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "qhat": self.qhat, "method": self.method, "n_calib": self.n_calib}

    @classmethod
    def from_dict(cls, d: dict) -> ConformalCalibrator:
        c = cls(alpha=d.get("alpha", 0.1), qhat=d.get("qhat"), method=d.get("method", "lac"))
        c.n_calib = d.get("n_calib", 0)
        return c
