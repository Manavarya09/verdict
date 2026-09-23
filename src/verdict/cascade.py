"""The cascade: Verdict answers what it is sure about, your fallback handles the rest.

    from verdict.cascade import Cascade
    route = Cascade(decider, fallback=lambda text: llm_router(text))
    label = route("where is my parcel")        # Verdict if it commits, else the LLM
    route.stats()                              # {"calls": 1200, "committed": 0.79, ...}

``fallback`` receives the input and the Verdict answer (so it can use the shortlist:
``answer.confidence.conformal_set`` is the set of labels Verdict could not rule out) and
returns whatever your code expects. Works for any Decider kind."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from .core import Decider
from .types import Answer

T = TypeVar("T")


class Cascade(Generic[T]):
    def __init__(
        self,
        decider: Decider,
        fallback: Callable[[str, Answer], T] | Callable[[str], T],
        pick: Callable[[Answer], T] | None = None,
        fallback_takes_answer: bool | None = None,
    ):
        self.decider = decider
        self.fallback = fallback
        self.pick = pick or _default_pick
        if fallback_takes_answer is None:
            import inspect

            fallback_takes_answer = len(inspect.signature(fallback).parameters) >= 2
        self._two = fallback_takes_answer
        self.calls = 0
        self.committed = 0
        self.verdict_ms = 0.0
        self.fallback_ms = 0.0

    def __call__(self, text: str, context: dict | None = None) -> T:
        self.calls += 1
        t0 = time.perf_counter()
        a = self.decider(text, context)
        self.verdict_ms += (time.perf_counter() - t0) * 1000
        if not a.confidence.abstain:
            self.committed += 1
            return self.pick(a)
        t1 = time.perf_counter()
        out = self.fallback(text, a) if self._two else self.fallback(text)  # type: ignore[call-arg]
        self.fallback_ms += (time.perf_counter() - t1) * 1000
        return out

    def stats(self) -> dict[str, Any]:
        n = max(1, self.calls)
        return {
            "calls": self.calls,
            "committed": self.committed / n,
            "escalated": (self.calls - self.committed) / n,
            "verdict_ms_avg": self.verdict_ms / n,
            "fallback_ms_avg": self.fallback_ms / max(1, self.calls - self.committed),
            "coverage_target": 1 - self.decider.conformal.alpha if self.decider.calibrated else None,
        }


def _default_pick(a: Answer) -> Any:
    if hasattr(a, "label"):
        return a.label
    if hasattr(a, "value"):
        return a.value
    return a.verdict  # type: ignore[union-attr]
