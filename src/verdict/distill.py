"""Distill an LLM decision into a Decider.

You already have the training data: every time your LLM router, judge or classifier ran,
it produced (input, answer). Point ``distill`` at that log and get a Decider that agrees
with the LLM on held-out rows, answers in milliseconds, and says when it is unsure.

Log format: JSONL with ``input`` (or ``text``/``prompt``) and ``answer`` (or ``label``).
Optional ``context`` object per row."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .core import Decider, Verdict
from .types import Question


def read_traces(path: str | Path) -> list[tuple[str, str | int | bool]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        x = r.get("input", r.get("text", r.get("prompt")))
        y = r.get("answer", r.get("label", r.get("output")))
        if x is None or y is None:
            continue
        if isinstance(y, dict):  # e.g. {"choice": "billing"}
            y = y.get("choice", y.get("label", y.get("value", y.get("verdict"))))
        rows.append((str(x), y))
    return rows


def distill(
    verdict: Verdict,
    question: Question,
    traces: list[tuple[str, str | int | bool]],
    coverage: float = 0.9,
    calib_frac: float = 0.2,
    test_frac: float = 0.1,
    seed: int = 0,
) -> tuple[Decider, dict]:
    """Split traces into fit / calibrate / test, train, and report agreement with the LLM."""
    rng = random.Random(seed)
    rows = list(traces)
    rng.shuffle(rows)
    n = len(rows)
    n_test = max(20, int(n * test_frac))
    n_cal = max(20, int(n * calib_frac))
    if n < n_test + n_cal + 10:
        raise ValueError(f"need at least ~{n_test + n_cal + 10} traces, got {n}")
    test, cal, fit = rows[:n_test], rows[n_test : n_test + n_cal], rows[n_test + n_cal :]
    d = verdict.compile(question)
    d.fit(fit)
    d.calibrate(cal, coverage=coverage)
    ev = d.evaluate(test)
    report = {
        "traces": n,
        "fit": len(fit),
        "calibrate": len(cal),
        "test": len(test),
        "agreement_with_llm": ev["accuracy"],
        "agreement_when_committed": ev.get("accuracy_when_committed"),
        "automation_rate": ev.get("automation_rate"),
        "coverage_target": coverage,
        "coverage_observed": ev.get("coverage_observed"),
        "ece": ev["ece"],
        "head": d.meta.get("fit", {}).get("head"),
        "labels_seen": sorted({str(y) for _, y in fit}),
    }
    d.meta["distill"] = report
    return d, report


def label_histogram(traces) -> dict[str, int]:
    h: dict[str, int] = {}
    for _, y in traces:
        h[str(y)] = h.get(str(y), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


def suggest_question(traces) -> Question:
    """Infer a Question from the answers: bools -> check, ints on a small range -> score,
    otherwise choose over the observed labels."""
    from .types import options_from

    ys = [y for _, y in traces]
    if all(isinstance(y, bool) for y in ys):
        return Question(kind="check", claim="the LLM would answer yes")
    if all(isinstance(y, int) and not isinstance(y, bool) for y in ys):
        lo, hi = min(ys), max(ys)
        if 2 <= hi - lo + 1 <= 10:
            return Question(kind="score", scale=(lo, hi))
    labels = list(label_histogram(traces))
    return Question(kind="choose", options=options_from(labels))


