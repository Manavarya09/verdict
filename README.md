<p align="center"><img src="https://raw.githubusercontent.com/Manavarya09/verdict/main/docs/assets/banner.svg" alt="verdict: small, fast, honest decision models" width="100%"></p>

<p align="center">
  <a href="https://github.com/Manavarya09/verdict/actions/workflows/ci.yml"><img src="https://github.com/Manavarya09/verdict/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/Manavarya09/verdict/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="license"></a>
  <a href="https://pypi.org/project/verdictml/"><img src="https://img.shields.io/pypi/v/verdictml.svg" alt="pypi"></a>
  <a href="https://manavarya09.github.io/verdict/"><img src="https://img.shields.io/badge/demo-runs%20in%20your%20browser-6ee7c8" alt="demo"></a>
  <a href="https://colab.research.google.com/github/Manavarya09/verdict/blob/main/examples/verdict_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
</p>

<p align="center"><b>Small, fast, honest decision models.</b><br>
Replace LLM calls for routing, guardrails, triage and policy checks with typed answers in milliseconds, on a CPU, with a probability you can trust.</p>

```python
from verdict import Verdict

v = Verdict()                                                      # verdict-small: 118M params, 100+ languages, CPU is fine
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

| | Jev | Laya | kev | **Verdict** |
|---|---|---|---|---|
| Zero-shot, Banking77 (77 intents) | 0.80-0.87 | 0.425 | 0.425 (third-party) | **0.594** base, 0.556 trained |
| Trained on your labels | not possible | 4-5 h on 2xT4 | delta fine-tune, GPU | **0.9 s on a laptop CPU** (16 labels/class) |
| Accuracy after training, Banking77 | 0.80-0.87 (zero-shot only) | 0.425 | not reported | **0.858** with 16 labels/class, **0.913** full |
| Calibration (ECE, lower is better) | 0.144-0.246 | 0.466 as shipped | "still a bit overconfident" (author) | **0.014-0.030** |
| "I don't know" | opaque confidence | `act_probability` always 1.0 | threshold on confidence | **conformal abstain with a coverage guarantee** |
| Max options | 255 | ~20 before degrading | collapses past ~20 | **unbounded** (options embedded once, cached) |
| Languages | English first | English ckpt fails silently on non-Latin | English | **100+** (multilingual encoder) |
| Latency | 70-500 ms network | 33 ms GPU, 329 ms+ CPU | 12-26 ms H100, 780 ms M5 | **0.5 ms/example batched (M-series), single-digit ms CPU** |
| Weights, data, training code | closed | weights only | all open | **all open, Apache-2.0** |
| Wire format | `/v1/systemone` | `/v1/systemone` | `/v1/systemone` | **`/v1/systemone` compatible** + native API + MCP |
| Option order changes the answer | yes, log-odds shift 0.3-0.5 | yes, flip rate up to 0.23 | yes, 13.75 pp in a third-party test | **no, by construction** |

Jev, Laya and kev numbers are from their own docs and published third-party evals; see
[docs/RESEARCH.md](https://github.com/Manavarya09/verdict/blob/main/docs/RESEARCH.md). Verdict numbers are from [`bench/`](https://github.com/Manavarya09/verdict/blob/main/bench/) and
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
does, and hands the rest to a human or an LLM. The cascade is one object:

```python
from verdict.cascade import Cascade

route = Cascade(d, fallback=lambda text, answer: llm_router(text, shortlist=answer.confidence.conformal_set))
route("where is my parcel")   # Verdict's label when it commits, the LLM's otherwise
route.stats()                 # calls, committed share, average ms on each path
```

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

## Distill your LLM's decisions

Your router, judge or classifier has been logging `(input, answer)` for months. That is a
training set.

```bash
verdict distill traces.jsonl support.verdict --coverage 0.9
```

```
traces 12,400 · fit 8,680 · calibrate 2,480 · test 1,240
agreement_with_llm 0.94 · agreement_when_committed 0.98 · automation_rate 0.81 · ece 0.02
```

Read: Verdict now answers 81% of that traffic itself, agreeing with the LLM 98% of the time
on those, and escalates the rest. The question kind is inferred from the answers (labels,
small integer range, or booleans) or given with `--question q.json`.

## Drop-in for Jev / Laya clients

```bash
pip install "verdictml[serve]"
verdict serve --port 8000
```

Then point any Jev, Laya or impossibl client at `http://localhost:8000/v1/systemone`. The
`choice` / `score` / `noul` question shapes are accepted unchanged; answers add an
`abstain` flag. The native `POST /v1/decide` returns full `Choice` / `Score` / `Check`
objects.

## No PyTorch? Use the ONNX engine

```bash
pip install "verdictml[onnx]"
```

```python
v = Verdict(engine="onnx")     # int8 weights from the Hub, ~120 MB, onnxruntime + tokenizers only
```

Same model, same numbers, 0.7 ms per example batched and under 2 ms single on a laptop CPU.
These are the exact weights the browser playground runs.

## Long inputs

Inputs longer than the encoder window are chunked with overlap, embedded, mean-pooled and
re-normalised. A twenty-page document becomes one vector instead of a silently truncated one.

## As an agent tool (MCP)

```bash
pip install "verdictml[mcp]"
verdict mcp
```

```json
{"mcpServers": {"verdict": {"command": "verdict", "args": ["mcp"]}}}
```

Three tools, `choose`, `score`, `check`, for Claude Desktop, Claude Code, Cursor or any MCP
client. Give an agent a fast, calibrated judgement call instead of another LLM round trip.

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
| Banking77 (77 intents) | 0 | 0.594 | 0.054 | 0.10 | 0.979 | 0.8 |
| Banking77 | 16 | **0.860** | 0.036 | 0.76 | 0.960 | 0.6 |
| Banking77 | all (8,993) | **0.920** | 0.032 | 0.98 | 0.929 | 0.6 |
| CLINC150 (150 intents) | 0 | 0.536 | 0.048 | 0.16 | 0.957 | 0.5 |
| CLINC150 | 16 | **0.928** | 0.018 | 1.00 | 0.928 | 0.5 |
| CLINC150 | all (14,000) | **0.953** | 0.015 | 1.00 | 0.953 | 0.5 |
| MASSIVE en (60 intents) | 0 | 0.612 | 0.111 | 0.12 | 0.898 | 0.5 |
| MASSIVE en | 16 | 0.789 | 0.034 | 0.55 | 0.956 | 0.5 |
| MASSIVE en | all (10,514) | **0.881** | 0.030 | 0.85 | 0.933 | 0.5 |
| MASSIVE **Hindi** (English option names) | 0 | 0.494 | 0.068 | 0.02 | | 0.5 |
| MASSIVE Hindi | 16 | 0.742 | 0.053 | 0.41 | | 0.5 |
| MASSIVE Hindi | all | **0.863** | 0.043 | 0.69 | | 0.5 |
| SST-5 (score 1-5) | 0 | 0.274 | 0.097 | | | 0.9 |
| SST-5 | 16 | 0.381 | 0.032 | | | 0.9 |
| SST-5 | all (7,544) | 0.462 | 0.029 | | | 1.2 |
| ToxicChat (check, 7% positive) | 0 | AUROC 0.59 | | | | 2.7 |
| ToxicChat | 16 | AUROC 0.86 | 0.016 | 0.77 | | 2.8 |
| ToxicChat | all (9,082) | **AUROC 0.955**, acc 0.936 | 0.011 | | | 2.3 |
| Banking77, + `deberta-v3-base-zeroshot-v2.0` reranker (top-8) | 0 | **0.650** | 0.103 | 0.22 | 0.950 | ~30 |
| SST-5, + `mDeBERTa-v3-base-xnli` reranker | 0 | 0.349 | | | | 98 |
| ToxicChat, + `deberta-v3-base-zeroshot-v2.0` reranker | 0 | AUROC 0.69 | | | | 127 |
| ToxicChat, + `mDeBERTa-v3-base-xnli` reranker | 0 | AUROC 0.46 | | | | 59 |
| **verdict-small** (trained encoder), Banking77, full test 3,076 | 0 | 0.556 | 0.030 | 0.07 | 0.985 | 0.6 |
| verdict-small, Banking77 | 16 | 0.842 | 0.020 | 0.75 | 0.944 | 0.6 |
| verdict-small, SST-5, full test 2,210 | 0 | **0.403** | 0.024 | | | 0.9 |
| verdict-small, ToxicChat, full test 5,083 | 0 | **AUROC 0.892**, acc 0.894 | 0.015 | | | 2.5 |
| verdict-small, ToxicChat | 16 | AUROC 0.939 | 0.014 | 0.93 | | 2.5 |
| Banking77, **ONNX int8, CPU only** | 0 | 0.604 | 0.062 | 0.10 | | 0.7 idle* |
| Banking77, ONNX int8, CPU only | 16 | 0.858 | 0.030 | 0.74 | | 0.7 idle* |

**typed-decisions** (the benchmark Laya and Jev are compared on: 4 workflows, JSON states,
2,000 test decisions labelled by Jev; random 0.318, majority 0.461):

| model | protocol | accuracy |
|---|---|---|
| Jev 1.13 | zero-shot | 0.727 |
| Laya base | zero-shot | 0.362 |
| Laya typed-decisions | fine-tuned on the train split | 0.766 |
| Verdict e5-small | zero-shot, raw JSON state | 0.323 |
| Verdict step-1,000 checkpoint | zero-shot, raw JSON state | 0.386 |
| Verdict e5-small + heads | fitted on the train split, raw JSON state | 0.625 |
| Verdict e5-small, full fine-tune with soft targets, 3 epochs | Laya's protocol | 0.689 (0.693 through the Verdict path, 52% automation at 0.81 accuracy when committed) |
| Verdict e5-base (278M), full fine-tune, 3 epochs | Laya's protocol | 0.706 |

On jevbench's public tiers (mostly knowledge and multi-hop items) verdict-small scores 0.94 / 0.49 / 0.40
on easy / original / hard: routing and intent items are fine, reasoning items are not. Full table in
[docs/BENCHMARKS.md](https://github.com/Manavarya09/verdict/blob/main/docs/BENCHMARKS.md).

We are behind here and we know why. These states are JSON records, and the label depends on
reading fields *together with* the options ("constraint_violations: 0" plus "harmful"). A
bi-encoder scores each option against one vector of the state, by design: that is what buys
unbounded options, cached inputs and the browser build. A listwise cross-encoder variant for
structured states is the v2 item on the roadmap. `train/finetune_typed.py` reproduces both rows.

Reference points from published evals: Jev zero-shot Banking77 0.80-0.87, CLINC150 0.87;
Laya Banking77 0.425, SST-5 0.372, MASSIVE non-English mean 0.451, ToxicChat 0.755.

`verdict-small` is `multilingual-e5-small` trained for 2,000 steps on the typed-decision mix in
`train/` (Banking77, SST-5 and ToxicChat never in the mix). Load it with `Verdict(model="verdict-small")`.
Zero-shot `check` went from AUROC 0.59 to 0.89 and `score` from 0.27 to 0.40; intent zero-shot
dipped from 0.594 to 0.556, which the next run addresses with a lower learning rate.

Where we lose, in plain words: **zero-shot on `score` and `check` questions is weak with
the default bi-encoder** (SST-5 0.27, ToxicChat AUROC 0.59). A cross-encoder reranker helps
(`Verdict(reranker=...)`: Banking77 zero-shot 0.594 to 0.650, at ~30 ms instead of 0.6 ms);
the trained-encoder track in [docs/PLAN.md](https://github.com/Manavarya09/verdict/blob/main/docs/PLAN.md) is the real answer. Sixteen labels
per class fixes all of it, and the full ToxicChat train set reaches AUROC 0.955.

\* ONNX rows were measured while a training run occupied the machine (16 ms/example then); 0.7 ms/example batched on an idle M5 CPU.

\* zero-shot probabilities are uncalibrated by definition; `calibrate()` fixes ECE without
changing accuracy. Apple M5, `intfloat/multilingual-e5-small`, batched.

We publish the rows we lose too. Zero-shot on fine-grained intent sets is where a bigger
cross-encoder still wins; that is the trained-encoder track in [docs/PLAN.md](https://github.com/Manavarya09/verdict/blob/main/docs/PLAN.md).

## How it works

```
input ──► bi-encoder (options embedded once, cached) ──► logits
                  └─ optional cross-encoder rerank of top-k ─┘
          head: none | prototype | logistic   ◄── fit(examples)
          temperature scaling                 ◄── calibrate(held_out)
          conformal set (LAC)  ──► commit or abstain
```

Options are scored independently, so option order cannot change an answer (Jev shifts log-odds by 0.3-0.5 when you reorder) and P(true) + P(false) is exactly 1 (Jev's ranges 0.71-1.42). Both are tests in `tests/test_invariance.py`.

The default encoder is `intfloat/multilingual-e5-small` (118M, MIT). Swap any
sentence-transformers model with `Verdict(model=...)`. Add an NLI cross-encoder reranker with
`Verdict(reranker="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")` for
`check` questions and subtle option sets.

## Install

```bash
pip install verdictml                  # import verdict
pip install "verdictml[serve,onnx,mcp]"  # + server, ONNX engine, MCP
pip install "verdictml[bench]"         # + datasets for the benchmark suite
```

## Roadmap

1. ONNX int8 export and a browser playground (ORT Web).
2. TypeScript SDK with literal-union return types from the options array.
3. Trained multilingual encoder on an open, commercially clean typed-decision mix (open data list, open script, open weights).
4. Long inputs: chunk + max-pool.

## License

Apache-2.0.
