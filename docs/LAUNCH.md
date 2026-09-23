# Launch plan

## Gate (do not post before all four are true)

1. Trained encoder beats the e5-small baseline on all three held-out suites and the README
   table has a `verdict-small` row.
2. `pip install verdictml` works from PyPI (needs a PyPI token; not done yet).
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
