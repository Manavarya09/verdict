"""The public API.

    from verdict import Verdict
    v = Verdict()                                   # small multilingual model, CPU is fine
    v.choose("I was charged twice", ["refund", "cancel", "other"])
    v.score("Great product, slow delivery", scale=(1, 5))
    v.check("Please call me back", claim="the customer asks for a human")

    d = v.compile(question)                          # a reusable Decider
    d.fit(examples)                                  # seconds, on CPU; beats hosted models
    d.calibrate(held_out, coverage=0.9)              # honest probabilities + abstain guarantee
    d.save("routing.verdict")
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from .calibration import ConformalCalibrator, TemperatureScaler
from .engines.base import Engine
from .engines.embed import DEFAULT_EMBED_MODEL, EmbedEngine
from .heads import Head, LinearHead, PrototypeHead, head_from_dict
from .types import (
    Answer,
    Check,
    Choice,
    Confidence,
    Decision,
    Example,
    Option,
    Question,
    Score,
    options_from,
)

FORMAT_VERSION = 1


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1)
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


class Decider:
    """A compiled question: fixed option set, optional trained head, own calibration.

    Cheap to call many times. Everything an answer needs is computed from one input
    embedding, so a batch of 1,000 inputs is one encoder pass."""

    def __init__(
        self,
        question: Question,
        engine: EmbedEngine,
        reranker: Engine | None = None,
        rerank_top_k: int = 8,
        rerank_weight: float = 4.0,
        coverage: float = 0.9,
    ):
        self.question = question
        self.engine = engine
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k
        self.rerank_weight = rerank_weight
        self.head: Head | None = None
        self.temperature = TemperatureScaler(1.0)
        self.conformal = ConformalCalibrator(alpha=1 - coverage)
        self.calibrated = False
        self.meta: dict = {}
        self._labels = self._option_labels()
        self._option_texts = [
            self.engine.question_text(question, t) for t in self.engine.option_texts(question)
        ]
        self._O = self.engine.embed_options(self._option_texts)  # [K, D]

    # ---- labels ------------------------------------------------------------
    def _option_labels(self) -> list[str]:
        q = self.question
        if q.kind == "choose":
            return q.option_labels
        if q.kind == "score":
            lo, hi = q.scale  # type: ignore[misc]
            return [str(i) for i in range(lo, hi + 1)]
        return ["false", "true"]

    @property
    def labels(self) -> list[str]:
        return self._labels

    def _label_index(self, answer: str | int | bool) -> int:
        q = self.question
        if q.kind == "check":
            return int(bool(answer)) if not isinstance(answer, str) else self._labels.index(answer.lower())
        if q.kind == "score":
            return int(answer) - q.scale[0]  # type: ignore[index]
        return self._labels.index(str(answer))

    # ---- raw logits --------------------------------------------------------
    def _embed(self, inputs: Sequence[str], context: Sequence[dict | None] | None = None) -> np.ndarray:
        rendered = []
        for i, t in enumerate(inputs):
            ctx = context[i] if context else None
            rendered.append(Decision(input=t, question=self.question, context=ctx).rendered_input())
        return self.engine.embed_inputs(rendered)

    def _logits_from_embeddings(self, X: np.ndarray, inputs: Sequence[str] | None = None) -> np.ndarray:
        Z = self.head.logits(X) if self.head is not None else (X @ self._O.T) * self.engine.scale
        if self.reranker is not None and inputs is not None:
            Z = self._rerank(Z, inputs)
        return Z

    def _rerank(self, Z: np.ndarray, inputs: Sequence[str]) -> np.ndarray:
        """Re-score the top-k options with the cross-encoder and fuse (z-scored sum).
        Options outside the top-k keep their order but sit strictly below the reranked ones.
        Measured on Banking77 zero-shot: the cross-encoder's own order beats any blend, so the
        default weight (4.0) lets it dominate inside the top-k while the bi-encoder decides k."""
        Z = Z.copy()
        k = min(self.rerank_top_k, Z.shape[1])
        for i, text in enumerate(inputs):
            top = np.argsort(-Z[i])[:k]
            d = Decision(input=text, question=self.question)
            r = self.reranker.score_subset(d, [int(j) for j in top])  # type: ignore[union-attr]
            fused = _zscore(Z[i][top]) + self.rerank_weight * _zscore(r)
            spread = Z[i][top].std() + 1e-6
            floor = Z[i][top].min() - 2 * spread
            Z[i] = np.minimum(Z[i], floor)
            Z[i][top] = floor + 2 * spread + (fused - fused.min()) * spread
        return Z

    def logits(self, inputs: Sequence[str], context: Sequence[dict | None] | None = None) -> np.ndarray:
        X = self._embed(inputs, context)
        return self._logits_from_embeddings(X, inputs)

    # ---- answers -----------------------------------------------------------
    def _answer(self, z: np.ndarray, latency_ms: float) -> Answer:
        p = self.temperature.transform(z[None, :])[0]
        order = np.argsort(-p)
        sets = self.conformal.prediction_set(p[None, :])[0]
        top = int(order[0])
        margin = float(p[order[0]] - (p[order[1]] if len(p) > 1 else 0.0))
        conf = Confidence(
            probability=float(p[top]),
            margin=margin,
            entropy=_entropy(p),
            abstain=len(sets) > 1 if self.calibrated else margin < 0.1,
            calibrated=self.calibrated,
            conformal_set=[self._labels[i] for i in sets] if self.calibrated else None,
        )
        q = self.question
        eng = self.engine.name if self.reranker is None else f"{self.engine.name}+{self.reranker.name}"
        if self.head is not None:
            eng += f"+{self.head.to_dict()['kind']}-head"
        if q.kind == "choose":
            return Choice(
                label=self._labels[top],
                confidence=conf,
                distribution={self._labels[i]: float(p[i]) for i in order},
                ranked=[self._labels[i] for i in order],
                engine=eng,
                latency_ms=latency_ms,
            )
        if q.kind == "score":
            lo = q.scale[0]  # type: ignore[index]
            values = np.arange(lo, lo + len(p))
            return Score(
                value=int(values[top]),
                expected=float((values * p).sum()),
                confidence=conf,
                distribution={int(v): float(pi) for v, pi in zip(values, p)},
                engine=eng,
                latency_ms=latency_ms,
            )
        return Check(
            verdict=bool(top == 1),
            probability=float(p[1]),
            confidence=conf,
            engine=eng,
            latency_ms=latency_ms,
        )

    def __call__(self, input: str, context: dict | None = None) -> Answer:
        return self.batch([input], [context])[0]

    def batch(self, inputs: Sequence[str], context: Sequence[dict | None] | None = None) -> list[Answer]:
        t0 = time.perf_counter()
        Z = self.logits(inputs, context)
        per = (time.perf_counter() - t0) * 1000 / max(1, len(inputs))
        return [self._answer(z, per) for z in Z]

    # ---- learning ------------------------------------------------------------
    def fit(
        self,
        examples: Sequence[Example] | Sequence[tuple[str, str | int | bool]],
        head: Literal["auto", "prototype", "linear"] = "auto",
        l2: float = 1e-5,
    ) -> Decider:
        """Train a head on frozen embeddings. Seconds on a CPU.

        ``auto`` picks a prototype head under ~8 examples per class, linear above."""
        pairs = [(e.decision.input, e.answer) if isinstance(e, Example) else e for e in examples]
        X = self._embed([p[0] for p in pairs])
        y = np.array([self._label_index(p[1]) for p in pairs])
        K = len(self._labels)
        per_class = np.bincount(y, minlength=K)
        kind = head
        if kind == "auto":
            kind = "linear" if np.median(per_class[per_class > 0]) >= 8 and len(y) >= 4 * K else "prototype"
        if kind == "prototype":
            self.head = PrototypeHead.fit(X, y, self._labels, self._O, scale=self.engine.scale)
        else:
            self.head = LinearHead.fit(X, y, self._labels, self._O, l2=l2)
        self.meta["fit"] = {"n": int(len(y)), "head": kind, "per_class": per_class.tolist()}
        self.calibrated = False  # calibration must be re-done on held-out data
        return self

    def calibrate(
        self,
        examples: Sequence[Example] | Sequence[tuple[str, str | int | bool]],
        coverage: float | None = None,
    ) -> Decider:
        """Fit temperature + conformal threshold on *held-out* labelled data.

        After this, ``probability`` is honest and ``abstain`` carries a guarantee: on data
        like the calibration set, the true answer is in ``conformal_set`` at least
        ``coverage`` of the time."""
        pairs = [(e.decision.input, e.answer) if isinstance(e, Example) else e for e in examples]
        if len(pairs) < 20:
            raise ValueError("calibrate needs at least 20 held-out examples (200+ recommended)")
        Z = self.logits([p[0] for p in pairs])
        y = np.array([self._label_index(p[1]) for p in pairs])
        self.temperature = TemperatureScaler().fit(Z, y)
        P = self.temperature.transform(Z)
        if coverage is not None:
            self.conformal.alpha = 1 - coverage
        self.conformal.fit(P, y)
        self.calibrated = True
        from .calibration import expected_calibration_error

        self.meta["calibration"] = {
            "n": int(len(y)),
            "temperature": self.temperature.temperature,
            "coverage": 1 - self.conformal.alpha,
            "qhat": self.conformal.qhat,
            "accuracy": float((P.argmax(1) == y).mean()),
            "ece": expected_calibration_error(P, y),
        }
        return self

    def evaluate(self, examples: Sequence[Example] | Sequence[tuple[str, str | int | bool]]) -> dict:
        """Accuracy, ECE, and the number everybody actually wants: how much you can
        automate at what accuracy."""
        from .calibration import expected_calibration_error

        pairs = [(e.decision.input, e.answer) if isinstance(e, Example) else e for e in examples]
        Z = self.logits([p[0] for p in pairs])
        y = np.array([self._label_index(p[1]) for p in pairs])
        P = self.temperature.transform(Z)
        pred = P.argmax(1)
        sets = self.conformal.prediction_set(P)
        commit = np.array([len(s) == 1 for s in sets])
        covered = np.array([yy in s for yy, s in zip(y, sets)])
        out = {
            "n": int(len(y)),
            "accuracy": float((pred == y).mean()),
            "ece": expected_calibration_error(P, y),
            "calibrated": self.calibrated,
        }
        if self.calibrated:
            out.update(
                {
                    "coverage_target": 1 - self.conformal.alpha,
                    "coverage_observed": float(covered.mean()),
                    "automation_rate": float(commit.mean()),
                    "accuracy_when_committed": float((pred[commit] == y[commit]).mean()) if commit.any() else None,
                    "mean_set_size": float(np.mean([len(s) for s in sets])),
                }
            )
        return out

    # ---- persistence -----------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        blob = {
            "format": FORMAT_VERSION,
            "engine": {"model": self.engine.model_name, "scale": self.engine.scale, "fingerprint": self.engine.fingerprint()},
            "reranker": getattr(self.reranker, "model_name", None),
            "rerank_top_k": self.rerank_top_k,
            "question": self.question.model_dump(),
            "head": self.head.to_dict() if self.head else None,
            "temperature": self.temperature.to_dict(),
            "conformal": self.conformal.to_dict(),
            "calibrated": self.calibrated,
            "meta": self.meta,
        }
        path.write_text(json.dumps(blob))
        return path

    @classmethod
    def load(cls, path: str | Path, verdict: Verdict | None = None) -> Decider:
        blob = json.loads(Path(path).read_text())
        v = verdict or Verdict(model=blob["engine"]["model"], reranker=blob.get("reranker"))
        if v.engine.model_name != blob["engine"]["model"]:
            raise ValueError(f"saved with {blob['engine']['model']}, loaded engine is {v.engine.model_name}")
        d = cls(Question.model_validate(blob["question"]), v.engine, v.reranker, blob.get("rerank_top_k", 8))
        d.head = head_from_dict(blob["head"]) if blob.get("head") else None
        d.temperature = TemperatureScaler.from_dict(blob["temperature"])
        d.conformal = ConformalCalibrator.from_dict(blob["conformal"])
        d.calibrated = blob.get("calibrated", False)
        d.meta = blob.get("meta", {})
        return d


class Verdict:
    """Entry point. Holds the encoder (and optional reranker); compiles questions."""

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        reranker: str | None = None,
        device: str | None = None,
        coverage: float = 0.9,
        rerank_top_k: int = 8,
    ):
        self.engine = EmbedEngine(model=model, device=device)
        self.reranker: Engine | None = None
        if reranker:
            from .engines.nli import NLIEngine

            self.reranker = NLIEngine(model=reranker, device=device)
        self.coverage = coverage
        self.rerank_top_k = rerank_top_k
        self._cache: dict[str, Decider] = {}

    # ---- compile -------------------------------------------------------------
    def compile(self, question: Question, rerank: bool | None = None) -> Decider:
        key = question.model_dump_json()
        if key not in self._cache:
            use_rr = self.reranker if (rerank if rerank is not None else self.reranker is not None) else None
            self._cache[key] = Decider(
                question, self.engine, use_rr, self.rerank_top_k, coverage=self.coverage
            )
        return self._cache[key]

    # ---- one-liners ------------------------------------------------------------
    def choose(
        self,
        input: str,
        options: Sequence[str] | Sequence[Option] | dict[str, str],
        prompt: str | None = None,
        context: dict | None = None,
    ) -> Choice:
        q = Question(kind="choose", prompt=prompt, options=options_from(options))
        return self.compile(q)(input, context)  # type: ignore[return-value]

    def score(
        self,
        input: str,
        scale: tuple[int, int] = (1, 5),
        rubric: dict[int, str] | None = None,
        prompt: str | None = None,
        context: dict | None = None,
    ) -> Score:
        q = Question(kind="score", prompt=prompt, scale=scale, rubric=rubric)
        return self.compile(q)(input, context)  # type: ignore[return-value]

    def check(self, input: str, claim: str, context: dict | None = None) -> Check:
        q = Question(kind="check", claim=claim)
        return self.compile(q)(input, context)  # type: ignore[return-value]

    def decide(self, decisions: Sequence[Decision]) -> list[Answer]:
        """Batch: any mix of questions. Groups by question so each is one encoder pass."""
        groups: dict[str, list[int]] = {}
        for i, d in enumerate(decisions):
            groups.setdefault(d.question.model_dump_json(), []).append(i)
        out: list[Answer | None] = [None] * len(decisions)
        for idx in groups.values():
            dec = self.compile(decisions[idx[0]].question)
            answers = dec.batch([decisions[i].input for i in idx], [decisions[i].context for i in idx])
            for i, a in zip(idx, answers):
                out[i] = a
        return out  # type: ignore[return-value]

    def load(self, path: str | Path) -> Decider:
        return Decider.load(path, self)
