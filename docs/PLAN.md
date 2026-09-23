# Verdict: plan

_Last updated 2026-09-23. Research sources in `docs/RESEARCH.md`._

## 1. What the category is, in one paragraph

A **decision model** answers a typed question about an input and returns a probability,
not text: *which of these N?*, *how much on this scale?*, *is this claim true?*
TypeSafe AI's **Jev** (15 Sep 2026, closed, hosted, $0.042/M tokens) named the category
"System One models". **Laya** (18 Sep, Apache-2.0, 19k stars in 5 days) cloned the wire
format with a fine-tuned ModernBERT. **kev** (5k stars) did it with a LoRA on Qwen3.5.
Four "awesome-jev" lists and 370+ integrations appeared in a week. The wire format
(`POST /v1/systemone`, `state` + `{id: {type, instructions, criteria}}`) is now de facto.

## 2. Why the incumbents are beatable (evidence)

| Weakness | Jev | Laya | Source |
|---|---|---|---|
| Zero-shot on real intent sets | 0.80-0.87 Banking77 | **0.425** Banking77; base ckpt 0.362 on typed-decisions vs 0.461 majority | Laya BENCHMARKS.md, MindStudio audit |
| Calibration | ECE 0.144-0.246; confidence is a deterministic function of p_max; saturates at 1.0 on 51% of items | ECE 0.466 as shipped; `act_probability` always 1.0 (broken head) | jev-baselines-eval, Laya issue #185 |
| Option count | hard cap 255 | degrades above ~20 (token budget) | docs, README |
| Context | 32k state | 512 tokens (English), silent truncation | docs, issue #174 |
| Languages | English first, CJK "less reliable", Russian -11 pts | English ckpt 0.000 on Khmer at 0.952 confidence | robustness audit, BENCHMARKS.md |
| Fine-tuning | none | notebook, 4-5 h on 2xT4; training data + scripts **not released** (issue #4) | docs |
| Abstain / "I don't know" | opaque `confidence`, thresholds "test with your own data" | `confidence = 1 - H/log k`, uncalibrated | docs |
| CPU | n/a (hosted) | 329 ms p50 with pinned threads, 9.4 s default; 49 s on a 4-vCPU VPS | BENCHMARKS.md, Flowtivity |
| Browser | none | community ONNX ports, 340-930 MB download | r4ai/laya-web |

And the number that matters most, from three independent audits: **a logistic-regression
head on a small encoder, trained on a few hundred labels, beats Jev on the same task**
(93.3% vs 83.2% Banking77, jev-baselines-eval; 93.2% at 8 ms CPU vs 80.1%, MindStudio).

## 3. The thesis

> **Verdict is the decision layer, not another checkpoint.**
> Zero-shot out of the box. Fit on your own labels in seconds. Every answer carries a
> probability that is honest and an abstain flag with a coverage guarantee.
> Small enough for a CPU or a browser tab. Wire-compatible with Jev.

Five claims, each one measurable and each one a table on the README:

1. **Honest.** Temperature-scaled probabilities and conformal prediction sets. "At 90%
   coverage, Verdict commits on X% of inputs and is right Y% of the time when it does."
   Nobody in the category ships a coverage guarantee. This answers the top HN complaint on
   both launches.
2. **Learns your decision in seconds.** `decider.fit(examples)` trains a head on frozen
   embeddings, on a CPU, in under a second per thousand examples. 16 labels per class is
   enough to beat Jev on Banking77. Everything (data recipe, scripts) is in the repo.
3. **Zero-shot that works.** Embedding retrieval over options plus an optional
   cross-encoder rerank. Beats Laya zero-shot on every public intent set from day one.
4. **No limits.** Options are embedded once and cached: 10 or 10,000 options cost the
   same. Long inputs are chunked and max-pooled.
5. **Runs anywhere.** Default model is 118M params (multilingual-e5-small), ~120 MB int8,
   100+ languages, CPU-ready. Same weights run in the browser via ONNX Runtime Web.

## 4. Architecture (v0.1)

```
input text ──► EmbedEngine (bi-encoder, cached option vectors) ──► logits
                       │                                            │
                       └── optional NLIEngine rerank of top-k ──────┤
                                                                    ▼
             Head (none | prototype | linear)  ◄── fit(examples)   logits
                                                                    ▼
             TemperatureScaler ◄── calibrate(held_out)            probs
                                                                    ▼
             ConformalCalibrator (LAC) ──► prediction set ──► abstain / commit
```

* `Verdict` holds the encoder; `compile(question)` returns a `Decider`.
* `Decider` = question + option embeddings + optional head + own calibration. Saved as
  one JSON file (`.verdict`) that loads anywhere the encoder is available.
* Three question kinds map to one mechanism: `choose` (K options), `score` (K scale
  points, rubric text per point, expected value reported), `check` ([false, true]).

## 5. What is built (this session) and what is next

Built: types, calibration (temperature + LAC/APS conformal + ECE), EmbedEngine,
NLIEngine, prototype + linear heads, `Verdict`/`Decider` API with fit / calibrate /
evaluate / save / load, tests.

Next, in order:

1. `bench/`: Banking77, CLINC150, MASSIVE (multilingual), ToxicChat, typed-decisions.
   Zero-shot and 16-shot rows, ECE and automation-rate columns. A `verdict bench` command
   anyone can rerun. Publish numbers where we lose too.
2. `verdict serve`: FastAPI, Jev wire format at `POST /v1/systemone` so every awesome-jev
   integration works with a base-URL change; native `/v1/decide` alongside.
3. CLI: `verdict ask`, `verdict fit`, `verdict calibrate`, `verdict eval`, `verdict serve`.
4. ONNX export + int8; browser demo page (ORT Web) with a live typed-question playground.
5. TypeScript SDK (`@verdict-ai/sdk` or similar): literal-union return types from the
   options array.
6. Training track (needs a GPU): multi-task fine-tune of mmBERT-small/base on the
   commercially-clean typed-decision mix (Banking77, CLINC, MASSIVE, PAWS, MNLI,
   WildGuardMix, Aegis-2.0, xlam-60k, BFCL, MT-Bench judgments) as the default zero-shot
   encoder. Open data list, open script, open weights.

## 6. Naming risk (decision needed)

`verdict` is taken on PyPI (haizelabs/verdict, LLM-judge pipelines, 348 stars) and on npm
(a 2018 rules engine). Two repos already use the name in this exact space:
`Heman10x-NGU/Verdict-open-jev` and `openJev-verdict-2.0` (274 stars). The package here is
published as `verdictml` with import name `verdict`. A distinct name would avoid a
permanent discoverability tax; candidates worth checking: `decide`, `sysone`, `ruling`,
`adjudicate`, `qbit`, `nod`.

## 7. Launch plan

* Title with a falsifiable number: "Show HN: Verdict, an open decision model that beats
  Jev on your task with 100 labels, on a CPU, with a coverage guarantee".
* Landing README: one 6-line snippet, one head-to-head table (Jev, Laya, Verdict), one
  live browser demo link, one "automation rate at 95% accuracy" chart.
* Honest-numbers section: every public benchmark including the ones we lose.
* Repeat the Show HN with each milestone (ONNX/browser, TS SDK, trained encoder).
