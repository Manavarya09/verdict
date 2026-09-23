"""Teach Verdict a decision from labelled examples, calibrate it, and get a guarantee.

Run: python examples/fit_your_own.py
"""

from verdict import Question, Verdict
from verdict.types import options_from

v = Verdict()
q = Question(kind="choose", options=options_from(["refund", "shipping", "account", "other"]))
d = v.compile(q)

train = [
    ("I want my money back", "refund"),
    ("charged twice, please reverse", "refund"),
    ("where is my parcel", "shipping"),
    ("tracking says delivered but nothing arrived", "shipping"),
    ("reset my password", "account"),
    ("change the email on my profile", "account"),
    ("do you have a mobile app", "other"),
    ("what are your opening hours", "other"),
] * 4  # a few examples per class is enough for the prototype head

held_out = [
    ("refund the duplicate charge", "refund"),
    ("my order is late", "shipping"),
    ("I cannot log in", "account"),
    ("is there a student discount", "other"),
] * 6  # 20+ rows needed; use real held-out data in practice

d.fit(train)  # milliseconds on a laptop CPU
d.calibrate(held_out, coverage=0.9)  # honest probabilities + abstain with a guarantee
print(d.evaluate(held_out))

for text in ["send my money back now", "the courier lost it", "hello?"]:
    a = d(text)
    print(f"{text!r:35} -> {a.label:9} p={a.confidence.probability:.2f} abstain={a.confidence.abstain} set={a.confidence.conformal_set}")

d.save("support.verdict")  # one JSON file; load with Verdict().load("support.verdict")
