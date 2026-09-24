# Launch plan

## Gate (do not post before all four are true)

1. Trained encoder beats the e5-small baseline on all three held-out suites and the README
   table has a `verdict-small` row.
2. `pip install verdictml` works from PyPI. Done 24 Sep (0.1.0).
3. Playground at https://manavarya09.github.io/verdict/ loads the trained weights.
4. CI green, `python -m bench.run all` reproduces every README number from a clean clone.

## Show HN title (pick one, keep the number)

* Show HN: Verdict, an open decision model that beats Jev on your task with 16 labels per class, on a CPU
* Show HN: Verdict, decision models with a coverage guarantee (open alternative to Jev/Laya)
* Show HN: Verdict, replace your LLM router with a 118M model that knows when to abstain

## First comment (the honest one, posted immediately)

Where it loses: zero-shot `score` and `check` with the bi-encoder alone (SST-5 0.27,
ToxicChat AUROC 0.59 before training). Jev is still ahead zero-shot on fine-grained intent
sets (0.80-0.87 vs 0.59-0.65 on Banking77). The fix is 16 labels per class (0.86) or the
trained encoder. Every number is `python -m bench.run`, protocol in docs/BENCHMARKS.md.

## What people will ask (have the answer ready)

* "Isn't this just an embedding model + logistic regression?" Yes, deliberately. That
  combination beats a $40M hosted model on its own benchmark; the product is the calibration
  guarantee, the abstain set, the wire compatibility and the honesty of the numbers.
* "Why not ModernBERT-large like Laya?" 3x smaller, 100+ languages, runs in a browser tab,
  and Laya's base checkpoint is near chance zero-shot anyway. Bigger encoders are one flag.
* "Where is the training data?" `train/data.py` lists every source and cap. Nothing hidden.
* "Coverage guarantee, really?" Split conformal, LAC score. Marginal coverage on exchangeable
  data. `evaluate()` prints observed coverage next to the target on every run.

## Repeat posts

One Show HN per milestone: trained encoder, browser build of the trained weights, TS SDK on
npm, `verdict distill` case study with real LLM logs.


## Final post text (24 Sep 2026)

**Title:** Show HN: Verdict – open decision models that say "I don't know" (Jev/Laya alternative)

**URL:** https://github.com/Manavarya09/verdict

**First comment (post immediately after submitting):**

Hi HN. Verdict is a small open-source "decision model": give it text plus a typed question (choose one of N, score on a scale, is this claim true) and it returns a probability, not prose. Same category as TypeSafe's Jev (closed) and Laya (open), same wire format, so existing clients work by changing a base URL.

What is different:

- Honest probabilities. Temperature scaling plus split conformal prediction, so `abstain` comes with a coverage guarantee. On Banking77 at 90% coverage it commits on 76% of inputs and is right 94.7% of the time when it does. ECE 0.01-0.03; Jev is 0.14-0.25, Laya ships at 0.47.
- Learns your decision in seconds. Heads on frozen embeddings: 16 labels per class gives 0.86 on Banking77 in 0.9 s on a laptop CPU. No GPU, no fine-tune.
- Option order cannot change the answer, by construction. Jev, Laya and kev all flip.
- 118M multilingual encoder, int8 ONNX, runs on a CPU or in a browser tab (demo on the page, no server). 100+ languages: Hindi input with English labels gets 0.86 on MASSIVE.
- `verdict distill traces.jsonl out.verdict` turns your LLM router's logs into a calibrated replacement with an agreement report. `verdict mcp` exposes it as agent tools.

Where it loses, so you do not have to find out: zero-shot on fine-grained intent sets is behind Jev (0.59-0.65 vs 0.80-0.87 on Banking77). On the typed-decisions benchmark our fine-tuned bi-encoder gets 0.71 vs Laya's 0.77 and Jev's 0.73; that benchmark is JSON states where fields must be read jointly with the options, which a bi-encoder cannot do by design (a cross-encoder variant is next). Every number is `python -m bench.run` with a no-leakage protocol, and the training mix, scripts and weights are in the repo.

Apache-2.0. Would love benchmark PRs and failures.
