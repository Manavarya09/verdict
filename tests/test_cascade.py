import numpy as np

from verdict.cascade import Cascade
from verdict.types import Check, Choice, Confidence


class FakeDecider:
    calibrated = True

    class conformal:
        alpha = 0.1

    def __init__(self, abstain_on):
        self.abstain_on = abstain_on

    def __call__(self, text, context=None):
        ab = text in self.abstain_on
        return Choice(
            label="a",
            confidence=Confidence(probability=0.5 if ab else 0.95, margin=0.0 if ab else 0.9, entropy=0.5, abstain=ab, calibrated=True, conformal_set=["a", "b"] if ab else ["a"]),
            distribution={"a": 0.5, "b": 0.5},
            ranked=["a", "b"],
            engine="fake",
            latency_ms=0.1,
        )


def test_cascade_routes_and_counts():
    seen = []
    c = Cascade(FakeDecider({"hard"}), fallback=lambda text, ans: seen.append((text, ans.confidence.conformal_set)) or "llm")
    assert c("easy") == "a"
    assert c("hard") == "llm"
    assert seen == [("hard", ["a", "b"])]
    s = c.stats()
    assert s["calls"] == 2 and np.isclose(s["committed"], 0.5) and s["coverage_target"] == 0.9


def test_single_arg_fallback_and_pick():
    c = Cascade(FakeDecider({"x"}), fallback=lambda t: "LLM:" + t, pick=lambda a: a.label.upper())
    assert c("y") == "A" and c("x") == "LLM:x"
    chk = Check(verdict=True, probability=0.9, confidence=Confidence(probability=0.9, margin=0.8, entropy=0.1), engine="f", latency_ms=0)
    from verdict.cascade import _default_pick

    assert _default_pick(chk) is True
