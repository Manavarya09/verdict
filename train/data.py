"""Typed-decision training mix built from public datasets.

Every source becomes rows of ``{"input", "kind", "prompt", "options", "gold"}`` where
``options`` are the exact option *texts* the inference engine renders (so train and
inference see the same strings). Rendering is delegated to ``verdict.engines.embed``.

HELD OUT, never loaded here: Banking77, SST-5, ToxicChat. Those are the zero-shot evals.

Licences: CLINC (CC-BY-3.0), MASSIVE (CC-BY-4.0), HWU64/SNIPS (NLU-Evaluation-Data),
XNLI (CC-BY-NC, research), Yelp (Yelp agreement), Aegis-2.0 (CC-BY-4.0), OpenAI moderation
eval (MIT), PAWS (permissive), dair-ai/emotion, AG News, amazon_reviews_multi (research),
tweet_eval, go_emotions (Apache). Sources with non-commercial terms are flagged
``research_only`` and can be excluded with ``--commercial``."""

from __future__ import annotations

import random
from dataclasses import dataclass

from verdict.engines.embed import EmbedEngine
from verdict.types import Question, options_from


@dataclass
class Row:
    input: str
    options: list[str]  # rendered option texts (already include the prompt)
    gold: int
    task: str


def _render(q: Question) -> list[str]:
    return [EmbedEngine.question_text(q, t) for t in EmbedEngine.option_texts(q)]


def _choose(labels: list[str], prompt: str) -> tuple[Question, list[str]]:
    q = Question(kind="choose", prompt=prompt, options=options_from(labels))
    return q, _render(q)


def _score(lo: int, hi: int, rubric: dict[int, str], prompt: str) -> tuple[Question, list[str]]:
    q = Question(kind="score", prompt=prompt, scale=(lo, hi), rubric=rubric)
    return q, _render(q)


def _check(claim: str) -> list[str]:
    return _render(Question(kind="check", claim=claim))


def _cap(rows: list[Row], n: int, seed: int) -> list[Row]:
    if len(rows) <= n:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


# ---------------------------------------------------------------- sources
def clinc(cap: int, seed: int) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = [n.replace("_", " ") for n in ds.features["intent"].names]
    _, opts = _choose(names, "What does the user want?")
    return _cap([Row(t, opts, i, "clinc150") for t, i in zip(ds["text"], ds["intent"])], cap, seed)


def massive(langs: list[str], cap_per_lang: int, seed: int) -> list[Row]:
    from datasets import load_dataset

    out = []
    for lang in langs:
        ds = load_dataset("mteb/amazon_massive_intent", lang, split="train")
        labels = sorted({l.replace("_", " ") for l in ds["label"]})
        _, opts = _choose(labels, "What does the user want?")
        idx = {l: i for i, l in enumerate(labels)}
        rows = [Row(t, opts, idx[l.replace("_", " ")], f"massive_{lang}") for t, l in zip(ds["text"], ds["label"])]
        out += _cap(rows, cap_per_lang, seed)
    return out


def hwu64(cap: int, seed: int) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("DeepPavlov/hwu64", split="train")
    names = [n.replace("_", " ") for n in ds.features["label"].names]
    _, opts = _choose(names, "What does the user want?")
    return _cap([Row(t, opts, i, "hwu64") for t, i in zip(ds["utterance"], ds["label"])], cap, seed)


def snips(cap: int, seed: int) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("DeepPavlov/snips", split="train")
    names = [n.replace("_", " ") for n in ds.features["label"].names]
    _, opts = _choose(names, "What does the user want?")
    return _cap([Row(t, opts, i, "snips") for t, i in zip(ds["utterance"], ds["label"])], cap, seed)


def ag_news(cap: int, seed: int) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news", split="train")
    _, opts = _choose(["world news", "sports", "business", "science and technology"], "What is this article about?")
    return _cap([Row(t, opts, i, "ag_news") for t, i in zip(ds["text"], ds["label"])], cap, seed)


def emotion(cap: int, seed: int) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("dair-ai/emotion", split="train")
    names = ds.features["label"].names
    _, opts = _choose(names, "What emotion does the writer express?")
    return _cap([Row(t, opts, i, "emotion") for t, i in zip(ds["text"], ds["label"])], cap, seed)


def go_emotions(cap: int, seed: int) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    names = ds.features["labels"].feature.names
    _, opts = _choose(names, "What emotion does the writer express?")
    rows = [Row(t, opts, ls[0], "go_emotions") for t, ls in zip(ds["text"], ds["labels"]) if len(ls) == 1]
    return _cap(rows, cap, seed)


def yelp(cap: int, seed: int) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("Yelp/yelp_review_full", split="train")
    _, opts = _score(1, 5, {1: "very negative", 2: "negative", 3: "mixed", 4: "positive", 5: "very positive"}, "How positive is this review?")
    rows = [Row(t[:1200], opts, i, "yelp") for t, i in zip(ds["text"], ds["label"])]
    return _cap(rows, cap, seed)


def amazon_reviews(cap: int, seed: int) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("SetFit/amazon_reviews_multi_en", split="train")
    _, opts = _score(1, 5, {1: "very negative", 2: "negative", 3: "mixed", 4: "positive", 5: "very positive"}, "How positive is this product review?")
    rows = [Row(t[:1200], opts, int(i), "amazon_reviews") for t, i in zip(ds["text"], ds["label"])]
    return _cap(rows, cap, seed)


def xnli(langs: list[str], cap_per_lang: int, seed: int) -> list[Row]:
    """entailment -> claim true, contradiction -> claim false; neutral dropped."""
    from datasets import load_dataset

    out = []
    for lang in langs:
        ds = load_dataset("facebook/xnli", lang, split="train", streaming=True)
        rows = []
        for r in ds:
            if r["label"] == 1:
                continue
            rows.append(Row(r["premise"], _check(r["hypothesis"]), 1 if r["label"] == 0 else 0, f"xnli_{lang}"))
            if len(rows) >= cap_per_lang * 3:
                break
        out += _cap(rows, cap_per_lang, seed)
    return out


def paws(cap: int, seed: int) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/paws", "labeled_final", split="train")
    rows = [
        Row(s1, _check(f"this sentence means the same as: {s2}"), int(l), "paws")
        for s1, s2, l in zip(ds["sentence1"], ds["sentence2"], ds["label"])
    ]
    return _cap(rows, cap, seed)


def aegis(cap: int, seed: int) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
    opts = _check("this message is unsafe, harmful, or asks for harmful content")
    rows = [
        Row(p, opts, 1 if lab == "unsafe" else 0, "aegis")
        for p, lab in zip(ds["prompt"], ds["prompt_label"])
        if p and lab in ("safe", "unsafe")
    ]
    return _cap(rows, cap, seed)


def openai_moderation(cap: int, seed: int) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("mmathys/openai-moderation-api-evaluation", split="train")
    opts = _check("this message violates a content policy (sexual, hate, violence, harassment or self-harm)")
    cols = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]
    rows = []
    for r in ds:
        flagged = any((r.get(c) or 0) for c in cols)
        rows.append(Row(r["prompt"], opts, int(flagged), "openai_moderation"))
    return _cap(rows, cap, seed)


def tweet_offensive(cap: int, seed: int) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("cardiffnlp/tweet_eval", "offensive", split="train")
    opts = _check("this tweet is offensive")
    return _cap([Row(t, opts, int(l), "tweet_offensive") for t, l in zip(ds["text"], ds["label"])], cap, seed)


MASSIVE_LANGS = ["en", "de", "fr", "es", "hi", "ja", "zh-CN", "ar", "pt", "ru", "tr", "ko", "sw", "bn"]
XNLI_LANGS = ["en", "de", "fr", "es", "hi", "zh", "ar", "ru", "tr", "sw"]


def build_mix(scale: float = 1.0, seed: int = 0, commercial: bool = False) -> list[Row]:
    """``scale`` multiplies every cap. 1.0 is ~130k rows."""
    c = lambda n: int(n * scale)  # noqa: E731
    parts = [
        ("clinc150", lambda: clinc(c(15000), seed)),
        ("massive", lambda: massive(MASSIVE_LANGS, c(2500), seed)),
        ("hwu64", lambda: hwu64(c(8000), seed)),
        ("snips", lambda: snips(c(4000), seed)),
        ("ag_news", lambda: ag_news(c(8000), seed)),
        ("emotion", lambda: emotion(c(8000), seed)),
        ("go_emotions", lambda: go_emotions(c(8000), seed)),
        ("yelp", lambda: yelp(c(10000), seed)),
        ("paws", lambda: paws(c(8000), seed)),
        ("aegis", lambda: aegis(c(12000), seed)),
        ("openai_moderation", lambda: openai_moderation(c(1700), seed)),
        ("tweet_offensive", lambda: tweet_offensive(c(8000), seed)),
    ]
    if not commercial:
        parts += [
            ("xnli", lambda: xnli(XNLI_LANGS, c(1500), seed)),
            ("amazon_reviews", lambda: amazon_reviews(c(6000), seed)),
        ]
    rows: list[Row] = []
    for name, fn in parts:
        try:
            got = fn()
            print(f"  {name:18} {len(got):>7} rows", flush=True)
            rows += got
        except Exception as e:  # keep going; one gated/broken source must not kill the run
            print(f"  {name:18} SKIPPED: {repr(e)[:120]}", flush=True)
    random.Random(seed).shuffle(rows)
    return rows
