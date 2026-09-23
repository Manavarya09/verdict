<h1 align="center">verdict</h1>
<p align="center"><b>Small, fast, honest decision models.</b><br>
Replace LLM calls for routing, guardrails, triage and policy checks with typed answers in milliseconds, on a CPU, with a probability you can trust.</p>

```python
from verdict import Verdict

v = Verdict()                                                      # 118M params, 100+ languages, CPU is fine
v.choose("Billed twice, refund or we cancel", ["billing", "technical", "sales"])   # -> billing  p=0.71
v.score("Great product, slow delivery", scale=(1, 5))                             # -> 4  expected 3.6
v.check("Can I talk to a person?", claim="the user asks for a human")             # -> True p=0.84
```

Three question kinds, one mechanism. No text generation, no schema errors, no hallucinated
labels: the answer is always one of the options you gave.

## Why another one

TypeSafe's **Jev** (closed, hosted) and **Laya** (open, 421M ModernBERT) made "System One"
decision models a category in September 2026. Both have the same three gaps, and Verdict is
built around closing them:

| | Jev | Laya | **Verdict** |
|---|---|---|---|
| Zero-shot, Banking77 (77 intents) | 0.80-0.87 | 0.425 | **0.594** |
| Trained on your labels | not possible | 4-5 h on 2xT4 | **0.9 s on a laptop CPU** (16 labels/class) |
| Accuracy after training, Banking77 | 0.80-0.87 (zero-shot only) | 0.425 | **0.858** with 16 labels/class, **0.913** full |
| Calibration (ECE, lower is better) | 0.144-0.246 | 0.466 as shipped | **0.014-0.020** |
| "I don't know" | opaque confidence | `act_probability` always 1.0 | **conformal abstain with a coverage guarantee** |
| Max options | 255 | ~20 before degrading | **unbounded** (options embedded once, cached) |
| Languages | English first | English ckpt fails silently on non-Latin | **100+** (multilingual encoder) |
| Latency | 70-500 ms network | 33 ms GPU, 329 ms+ CPU | **0.5 ms/example batched (M-series), single-digit ms CPU** |
| Weights, data, training code | closed | weights only | **all open, Apache-2.0** |
| Wire format | `/v1/systemone` | `/v1/systemone` | **`/v1/systemone` compatible** + native API |

Jev and Laya numbers are from their own docs and published third-party evals; see
[docs/RESEARCH.md](docs/RESEARCH.md). Verdict numbers are from [`bench/`](bench/) and
reproducible with one command below.

## The honest-probability part

Every answer carries a `Confidence`:

```python
a = d("the courier lost my parcel")
a.confidence.probability     # temperature-scaled, so 0.9 means right ~90% of the time
a.confidence.abstain         # True when the model cannot commit at your chosen coverage
a.confidence.conformal_set   # ["shipping"] or ["shipping", "refund"]: the labels it cannot rule out
```

`calibrate()` fits a temperature and a conformal threshold on held-out data. After that,
on data like your held-out set, the true label is inside `conformal_set` at least
`coverage` of the time. That is a distribution-free guarantee, not a heuristic. The
number you actually care about comes out of `evaluate()`:

```
Banking77, 16 labels per class, coverage 0.90:
  automation_rate 0.76   accuracy_when_committed 0.947   accuracy overall 0.858   ECE 0.020
```

Read: at 90% coverage Verdict commits on 76% of tickets, is right 94.7% of the time when it
does, and hands the rest to a human or an LLM. Build the cascade with one `if`.

## Teach it your decision

```python
from verdict import Verdict, Question
from verdict.types import options_from

v = Verdict()
d = v.compile(Question(kind="choose", options=options_from(["refund", "shipping", "account", "other"])))

d.fit(train_pairs)                       # [(text, label), ...]  seconds on a CPU
d.calibrate(held_out_pairs, coverage=0.9)
d.evaluate(test_pairs)                   # accuracy, ECE, automation_rate, accuracy_when_committed
d.save("support.verdict")                # one JSON file
```

`fit` trains a head on frozen embeddings: a prototype head from as little as one example
per class, a logistic head (initialised at the zero-shot solution, shrunk towards it) when
you have more. Nothing is fine-tuned, so it is fast, deterministic and cheap to redo.

## Drop-in for Jev / Laya clients

```bash
pip install "verdictml[serve]"
verdict serve --port 8000
```

Then point any Jev, Laya or impossibl client at `http://localhost:8000/v1/systemone`. The
`choice` / `score` / `noul` question shapes are accepted unchanged; answers add an
`abstain` flag. The native `POST /v1/decide` returns full `Choice` / `Score` / `Check`
objects.

## CLI

```bash
verdict ask "I want my money back" -o refund -o shipping -o account
verdict ask "Can I speak to someone?" --claim "the user asks for a human"
verdict fit question.json train.jsonl support.verdict --calibrate heldout.jsonl
verdict evaluate support.verdict test.jsonl
verdict bench banking77 --shots 0,16,-1
```

## Benchmarks

Protocol: heads fit on train minus a fixed 1,000-row calibration slice; temperature and
conformal threshold fit on that slice only; every number on the untouched test split.
`python -m bench.run all --shots 0,16,-1` reproduces the table.

| suite | shots/class | accuracy | ECE | automation @0.9 | acc. when committed | ms/example |
|---|---|---|---|---|---|---|
| Banking77 (77 intents) | 0 | 0.594 | 0.55* | | | 0.5 |
| Banking77 | 1 | 0.638 | 0.041 | 0.13 | 0.972 | 0.5 |
| Banking77 | 4 | 0.722 | 0.035 | 0.28 | 0.970 | 0.5 |
| Banking77 | 16 | 0.858 | 0.020 | 0.76 | 0.947 | 0.5 |
| Banking77 | all (8,993) | 0.913 | 0.014 | 0.98 | 0.923 | 0.6 |
| CLINC150 (150 intents) | 0 / 16 / all | _running_ | | | | |
| MASSIVE en (60 intents) | 0 / 16 / all | _running_ | | | | |
| SST-5 (score 1-5) | 0 / 16 / all | _running_ | | | | |
| ToxicChat (check) | 0 / 16 / all | _running_ | | | | |

\* zero-shot probabilities are uncalibrated by definition; `calibrate()` fixes ECE without
changing accuracy. Apple M5, `intfloat/multilingual-e5-small`, batched.

We publish the rows we lose too. Zero-shot on fine-grained intent sets is where a bigger
cross-encoder still wins; that is the trained-encoder track in [docs/PLAN.md](docs/PLAN.md).

## How it works

```
input ──► bi-encoder (options embedded once, cached) ──► logits
                  └─ optional cross-encoder rerank of top-k ─┘
          head: none | prototype | logistic   ◄── fit(examples)
          temperature scaling                 ◄── calibrate(held_out)
          conformal set (LAC)  ──► commit or abstain
```

The default encoder is `intfloat/multilingual-e5-small` (118M, MIT). Swap any
sentence-transformers model with `Verdict(model=...)`. Add an NLI cross-encoder reranker with
`Verdict(reranker="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")` for
`check` questions and subtle option sets.

## Install

```bash
pip install verdictml            # import verdict
pip install "verdictml[serve]"   # + HTTP server
pip install "verdictml[bench]"   # + datasets for the benchmark suite
```

## Roadmap

1. ONNX int8 export and a browser playground (ORT Web).
2. TypeScript SDK with literal-union return types from the options array.
3. Trained multilingual encoder on an open, commercially clean typed-decision mix (open data list, open script, open weights).
4. Long inputs: chunk + max-pool.

## License

Apache-2.0.
