import numpy as np
import pytest

from verdict import Decider, Question, Verdict
from verdict.types import Decision, Option, options_from


def test_question_validation():
    with pytest.raises(ValueError):
        Question(kind="choose", options=[Option(label="a")])
    with pytest.raises(ValueError):
        Question(kind="score")
    with pytest.raises(ValueError):
        Question(kind="check")
    q = Question(kind="choose", options=options_from({"a": "first", "b": "second"}))
    assert q.option_labels == ["a", "b"]


def test_decision_context_rendering():
    d = Decision(input="hi", question=Question(kind="check", claim="x"), context={"tier": "gold"})
    assert d.rendered_input() == "tier: gold\n\nhi"


@pytest.fixture(scope="module")
def v():
    return Verdict()


@pytest.mark.slow
def test_choose_score_check(v):
    r = v.choose("I was billed twice, refund the duplicate", {"billing": "refunds, invoices", "tech": "bugs"})
    assert r.label == "billing"
    assert abs(sum(r.distribution.values()) - 1) < 1e-6
    assert r.ranked[0] == "billing"
    s = v.score("terrible, never again", scale=(1, 3), rubric={1: "negative", 2: "neutral", 3: "positive"})
    assert s.value in (1, 2, 3) and 1 <= s.expected <= 3
    c = v.check("please let me speak to a person", claim="the user asks for a human")
    assert 0 <= c.probability <= 1


@pytest.mark.slow
def test_fit_calibrate_save_load(v, tmp_path):
    q = Question(kind="choose", options=options_from(["cats", "dogs", "birds"]))
    d = v.compile(q)
    rng = np.random.default_rng(0)
    cats = ["my kitten purrs", "the cat scratched the sofa", "tabby cats sleep all day", "a cat meowed"] * 6
    dogs = ["the puppy barked", "walking my dog", "golden retrievers shed", "the dog fetched"] * 6
    birds = ["the parrot talks", "sparrows at the feeder", "an eagle soared", "budgies chirp"] * 6
    data = [(t, "cats") for t in cats] + [(t, "dogs") for t in dogs] + [(t, "birds") for t in birds]
    rng.shuffle(data)
    d.fit(data[:36])
    d.calibrate(data[36:], coverage=0.9)
    assert d.calibrated
    r = d("the hound howled at night")
    assert r.label == "dogs"
    assert r.confidence.calibrated and r.confidence.conformal_set is not None
    ev = d.evaluate(data)
    assert ev["accuracy"] > 0.9 and ev["coverage_observed"] >= 0.85
    p = d.save(tmp_path / "pets.verdict")
    d2 = Decider.load(p, v)
    assert d2("kitty on the windowsill").label == "cats"
    assert d2.calibrated and d2.conformal.qhat == d.conformal.qhat


@pytest.mark.slow
def test_batch_mixed_questions(v):
    qa = Question(kind="choose", options=options_from(["sports", "finance"]))
    qb = Question(kind="check", claim="the text mentions money")
    out = v.decide(
        [
            Decision(input="the striker scored twice", question=qa),
            Decision(input="the stock fell 4%", question=qa),
            Decision(input="the stock fell 4%", question=qb),
        ]
    )
    assert out[0].label == "sports" and out[1].label == "finance"
    assert hasattr(out[2], "verdict")
