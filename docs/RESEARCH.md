# Research notes (23 Sep 2026)

Condensed from three research passes. Primary sources linked.

## Jev (TypeSafe AI)
* Launched 15 Sep 2026; $40M seed (DCVC); CEO Diogo Almeida (ex-OpenAI). Model `jev-1.13.0`.
* API: `POST https://api.typesafe.ai/v1/systemone` with `state` + `questions: {id: {type: choice|score|noul, instructions, criteria}}`. Choice max 255 options; score 2-10 levels; 64k tokens/request, 32k for state; text only. Response: `{model, answers: {id: {choice|score|noul, confidence, probabilities, legend}}, usage}`.
* Price $0.042/M input tokens, output free. Latency 70-500 ms. Headline "193.6x faster / 444.6x cheaper" is vs the slowest comparator in their own workflow evals; independent runs measured 4-7x / 30-65x.
* Own evals score against "average of GPT-6 Astra and Fable 5.1", agreement 67.8%. No calibration curves published.
* Independent: Banking77 zero-shot 0.80-0.87; **BGE-small + logistic regression 93.3%** (jev-baselines-eval); **22M encoder + LR 93.2% at 8 ms CPU** (MindStudio). Confidence exactly 1.0 on 51% of items; option-order shifts log-odds 0.3-0.5; Russian -11 pts; ECE 0.144-0.246.
* Closed weights, no fine-tuning, English-first. HN thread 1,929 pts: wanted open weights/local, calibration benchmarks ("what % can we automate at 90% accuracy"), fine-tuning.
* Sources: https://docs.typesafe.ai/api.md, https://typesafe.ai/blog/introducing-system-one-models-and-jev, https://news.ycombinator.com/item?id=49717558, https://github.com/ickma2311/jev-baselines-eval, https://www.mindstudio.ai/blog/jev-vs-classic-classifiers-benchmark, https://github.com/Yifan-Lan/awesome-jev-robustness

## Laya (Convai Innovations)
* Repo created 18 Sep; 19,076 stars in 5 days; Apache-2.0; `pip install laya`; HN 1,343 pts.
* Checkpoints: `convaiinnovations/laya` (ModernBERT-large 421M, 512 ctx), `laya-multilingual` (mmBERT-base 322M, 1024 ctx), `laya-typed-decisions`.
* Architecture: one packed sequence `[CLS] type+instructions [SEP] [MASK]opt0 [MASK]opt1 ... [SEP] state [SEP]`, 2 extra transformer layers, MLP scorer at each option's `[MASK]`. Options capped at 48 tokens each within a 192-256 token head budget, so >20 options degrade. Trained with "RLCD" (proper-scoring-rule reward, REINFORCE). Training data and scripts not released (issue #4).
* Numbers: typed-decisions 0.766 (fine-tuned on that benchmark's train split; base ckpt 0.362 vs 0.461 majority), Banking77 0.425, SST-5 0.372, ToxicChat 0.755, MASSIVE non-English 0.451. ECE 0.466 as shipped. `act_probability` head broken (always 1.0, issue #185). `noul` follows label words not state (issue #156). Khmer 0.000 acc at 0.952 confidence.
* CPU: 329 ms p50 with pinned threads, 9.4 s default; 49 s on a 4-vCPU VPS. Browser: community ONNX ports 340-930 MB.
* Serves Jev wire format (`laya-serve`, impossibl).
* Sources: https://github.com/NandhaKishorM/laya, https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md, https://huggingface.co/convaiinnovations/laya, https://news.ycombinator.com/item?id=49765348

## Others in the category (23 Sep)
kev (jaredpalmer, LoRA on Qwen3.5, 5.3k stars), Verdict-open-jev / openJev-verdict-2.0 (Heman10x-NGU, ModernBERT-base + GLiClass head, 274 stars, browser playground), von, AnyJev, agent-jev, LLM2Jev, jevbench. Four awesome-jev lists (1.4k / 677 / 364 / 169).

## Backbones
mmBERT-small (140M, 1,833 langs, XNLI 73.6) / mmBERT-base (307M, XNLI 77.1), ModernBERT-base/large (English), mDeBERTa-v3-base (XNLI 79.8, 512 ctx), Ettin encoders (17M-1B, open data), multilingual-e5-small/large (MIT), bge-m3, gte-multilingual-base (Apache). jina-v3 is CC-BY-NC: do not use.

## Zero-shot reference points
BTZSC (22 datasets): Qwen3-Reranker-0.6B is the best sub-1B zero-shot classifier (acc 0.64); NLI cross-encoders ~0.62; gte-large embeddings F1 0.62. deberta-v3-large-zeroshot-v2.0 Banking77 macro-F1 0.513. GLiClass-large Banking77 0.557. 41 open LLMs 135M-9B average Banking77 0.388, none above 0.80.

Our measurements (Banking77 test, first 300/1000 rows, Apple M5):
| method | acc@1 | acc@10 |
|---|---|---|
| multilingual-e5-small, embed only | 0.567 / 0.594 (n=1000) | 0.883 |
| multilingual-e5-large, embed only | 0.623 | 0.943 |
| bge-m3, embed only | 0.577 | 0.943 |
| e5-small + mDeBERTa-xnli rerank top-10 | 0.347 | |

## Datasets (commercially clean core)
Banking77 (CC-BY-4.0), CLINC150 (CC-BY-3.0), MASSIVE (CC-BY-4.0, 52 langs), PAWS, MNLI, WildGuardMix (ODC-BY), Aegis-2.0 (CC-BY-4.0), OpenAI moderation eval (MIT), xlam-function-calling-60k (CC-BY-4.0), BFCL (Apache), APIBench (Apache), MT-Bench human judgments (CC-BY-4.0), arena-55k. Eval-only: ANLI, XNLI, BeaverTails, ToxicChat (CC-BY-NC).

## Calibration
Temperature scaling (Guo 2017), LAC / APS / RAPS conformal (Sadinle 2019, Romano 2020, Angelopoulos & Bates 2021). Libraries: MAPIE (BSD), crepes (BSD), TorchCP (LGPL, avoid vendoring).

## Launch tactics that worked
Falsifiable number in the title (Outlines 854 pts, RouteLLM 244, model2vec), live browser demo (transformers.js 378 + 239), head-to-head table vs the incumbent, wrap an object people already hold (Jev wire format), repeat the Show HN per milestone. Encoder projects without these plateaued at 0.5-4k (GLiNER, SetFit, fastembed).
