"""Temperature scaling (Guo et al. 2017): one scalar that makes probabilities honest.

Fit on held-out labelled decisions; apply to raw logits before softmax. We fit by
minimising negative log-likelihood with a bounded 1-D search, no torch needed at
inference time."""

from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _nll(logits: np.ndarray, y: np.ndarray, t: float) -> float:
    p = _softmax(logits / t)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> TemperatureScaler:
        """Golden-section search over log-temperature in [1/20, 20]."""
        logits = np.asarray(logits, dtype=np.float64)
        y = np.asarray(labels)
        lo, hi = np.log(0.05), np.log(20.0)
        phi = (np.sqrt(5) - 1) / 2
        a, b = lo, hi
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = _nll(logits, y, np.exp(c)), _nll(logits, y, np.exp(d))
        for _ in range(60):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = _nll(logits, y, np.exp(c))
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = _nll(logits, y, np.exp(d))
        self.temperature = float(np.exp((a + b) / 2))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return _softmax(np.asarray(logits, dtype=np.float64) / self.temperature)

    def to_dict(self) -> dict:
        return {"temperature": self.temperature}

    @classmethod
    def from_dict(cls, d: dict) -> TemperatureScaler:
        return cls(d.get("temperature", 1.0))
