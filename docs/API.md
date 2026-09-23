# API reference (Python)

```python
from verdict import Verdict, Decider, Question, Option, Decision, Example
from verdict.types import options_from
from verdict.cascade import Cascade
```

## `Verdict(model=..., reranker=None, device=None, coverage=0.9, rerank_top_k=8, engine="torch")`

Holds the encoder. `model` is any sentence-transformers id or a local path (a trained
checkpoint). `engine="onnx"` uses onnxruntime + tokenizers with the int8 `Xenova/*` mirror
and no PyTorch. `reranker` is an NLI cross-encoder id; it rescores the top-k options.

| method | returns | notes |
|---|---|---|
| `choose(input, options, prompt=None, context=None)` | `Choice` | `options`: list of labels, list of `Option`, or `{label: description}` |
| `score(input, scale=(1,5), rubric=None, prompt=None, context=None)` | `Score` | `rubric={point: text}` gives the encoder something to read |
| `check(input, claim, context=None)` | `Check` | P(claim is true about input) |
| `decide(decisions)` | `list[Answer]` | any mix of questions; grouped so each question is one encoder pass |
| `compile(question)` | `Decider` | cached per question |
| `load(path)` | `Decider` | a saved `.verdict` file |

## `Decider`

| method | notes |
|---|---|
| `d(input, context=None)` / `d.batch(inputs, context=None)` | answers |
| `d.fit(examples, head="auto"\|"prototype"\|"linear", l2=1e-5)` | examples = `[(text, answer)]` or `Example`s; seconds on CPU |
| `d.calibrate(examples, coverage=0.9)` | held-out data; fits temperature + conformal threshold; 20 rows minimum, 200+ recommended |
| `d.evaluate(examples)` | accuracy, ECE, and when calibrated: coverage_observed, automation_rate, accuracy_when_committed; AUROC for `check` |
| `d.save(path)` / `Decider.load(path, verdict)` | one JSON file, encoder name checked on load |
| `d.logits(inputs)` | raw uncalibrated logits |

## Answers

* `Choice`: `label`, `ranked`, `distribution`, `confidence`, `engine`, `latency_ms`. `bool(choice)` is "did it commit".
* `Score`: `value` (argmax point), `expected` (probability-weighted), `distribution`, `confidence`.
* `Check`: `verdict`, `probability` = P(true), `confidence`.
* `Confidence`: `probability` (temperature-scaled), `margin`, `entropy`, `abstain`, `calibrated`, `conformal_set`.

Uncalibrated deciders set `abstain` from a margin heuristic (`margin < 0.1`) and
`calibrated=False`. Calibrated ones set it from the conformal set.

## `Cascade(decider, fallback, pick=None)`

Calls the decider; returns `pick(answer)` (label / value / verdict by default) when it
commits, else `fallback(text, answer)` (or `fallback(text)`). `stats()` reports the split.

## Server

`verdict serve --port 8000 [--model ...] [--reranker ...] [--decider a.verdict ...]`

* `POST /v1/systemone` — Jev-compatible: `{state, questions: {id: {type: choice|score|noul, instructions, criteria}}}`.
* `POST /v1/systemone/batch` — `{states: [...], questions}`; one encoder pass per question.
* `POST /v1/decide` — `{decisions: [{input, question, context}], decider?: name}` returns full answer objects.
* `GET /health`.

## CLI

`verdict ask | fit | evaluate | distill | serve | bench`. Run `verdict <cmd> --help`.
