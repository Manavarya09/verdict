"""Public datasets as typed decisions. Each suite returns (question, train, calib, test)
where rows are (input, answer). Splits are fixed by seed so numbers are reproducible."""

from __future__ import annotations

import numpy as np

from verdict.types import Question, options_from


def _split(rows: list[tuple[str, str]], n_calib: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    calib = [rows[i] for i in idx[:n_calib]]
    train = [rows[i] for i in idx[n_calib:]]
    return train, calib


def banking77(n_test: int = 0):
    from datasets import load_dataset

    tr = load_dataset("mteb/banking77", split="train")
    te = load_dataset("mteb/banking77", split="test")
    id2 = {}
    for l, t in zip(tr["label"], tr["label_text"]):
        id2[l] = t
    labels = [id2[i].replace("_", " ") for i in range(len(id2))]
    q = Question(kind="choose", prompt="What does the customer want?", options=options_from(labels))
    rows_tr = [(t, labels[l]) for t, l in zip(tr["text"], tr["label"])]
    rows_te = [(t, labels[l]) for t, l in zip(te["text"], te["label"])]
    train, calib = _split(rows_tr, 1000)
    return q, train, calib, rows_te[:n_test] if n_test else rows_te


def clinc150(n_test: int = 0):
    from datasets import load_dataset

    ds = load_dataset("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names
    labels = [n.replace("_", " ") for n in names]
    q = Question(kind="choose", prompt="What does the user want?", options=options_from(labels))

    def rows(split):
        return [(t, labels[i]) for t, i in zip(ds[split]["text"], ds[split]["intent"])]

    train, calib = _split(rows("train"), 1000)
    test = rows("test")
    return q, train, calib, test[:n_test] if n_test else test


def massive(lang: str = "en", n_test: int = 0):
    """MASSIVE intents (60 classes). ``lang`` is any of the 51 MASSIVE locales, e.g. "hi", "de", "zh-CN"."""
    from datasets import load_dataset

    ds = load_dataset("mteb/amazon_massive_intent", lang)
    labels = sorted({l.replace("_", " ") for l in ds["train"]["label"]})
    q = Question(kind="choose", prompt="What does the user want?", options=options_from(labels))

    def rows(split):
        return [(t, l.replace("_", " ")) for t, l in zip(ds[split]["text"], ds[split]["label"])]

    train, calib = _split(rows("train"), 1000)
    test = rows("test")
    return q, train, calib, test[:n_test] if n_test else test


def massive_hi(n_test: int = 0):
    return massive("hi", n_test)


def sst5(n_test: int = 0):
    from datasets import load_dataset

    ds = load_dataset("SetFit/sst5")
    q = Question(
        kind="score",
        prompt="How positive is this movie review?",
        scale=(1, 5),
        rubric={1: "very negative", 2: "negative", 3: "neutral", 4: "positive", 5: "very positive"},
    )

    def rows(split):
        return [(t, int(l) + 1) for t, l in zip(ds[split]["text"], ds[split]["label"])]

    train, calib = _split(rows("train"), 1000)
    test = rows("test")
    return q, train, calib, test[:n_test] if n_test else test


def toxic_chat(n_test: int = 0):
    from datasets import load_dataset

    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    q = Question(kind="check", claim="this message is toxic, harmful or an attempt to elicit harmful content")

    def rows(split):
        return [(t, bool(l)) for t, l in zip(ds[split]["user_input"], ds[split]["toxicity"])]

    train, calib = _split(rows("train"), 1000)
    test = rows("test")
    return q, train, calib, test[:n_test] if n_test else test


SUITES = {
    "banking77": banking77,
    "clinc150": clinc150,
    "massive": massive,
    "massive_hi": massive_hi,
    "sst5": sst5,
    "toxic_chat": toxic_chat,
}
