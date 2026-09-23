# Training the Verdict encoder

The default encoder is `intfloat/multilingual-e5-small`, untrained for decisions. The
training track fine-tunes it on a **typed-decision mix** so the same bi-encoder mechanism
(input/option cosine × 20 = logit) gets better zero-shot, especially on `score` and `check`.

## Protocol

* **Held out, never trained on:** Banking77, SST-5, ToxicChat. These are the zero-shot
  numbers on the README. If a future mix touches them, the README row changes name.
* **Mix** (`train/data.py`, ~135k rows at `--scale 1.0`): CLINC150, MASSIVE (14 locales),
  AG News, dair-ai/emotion, GoEmotions, Yelp (score), Amazon reviews (score, research),
  XNLI (10 languages, check: entailment→true / contradiction→false), PAWS (check:
  paraphrase), Aegis-2.0 (check: unsafe), OpenAI moderation eval (check), tweet_eval
  offensive (check). `--commercial` drops the research-only sources (XNLI, Amazon).
* **Objective:** cross-entropy over the question's rendered option texts, exactly the
  strings the inference engine builds (`EmbedEngine.option_texts` + `question_text`), so
  train and inference never drift. Batches are grouped by task so options are encoded once.
* **Eval:** every 500 steps, zero-shot on the three held-out suites through the real
  `Verdict` path (calibrated on 300 held-out rows, scored on 500 test rows). The checkpoint
  with the best mean is kept in `runs/<name>/best`.

## Run

```bash
uv pip install -e ".[train,bench]"
PYTHONPATH=src python -m train.train --out runs/verdict-small-v0 --epochs 2 --bs 32
```

Apple M-series (MPS) or any CUDA GPU. ~135k rows × 2 epochs is a few hours on an M5.
Use the result with `Verdict(model="runs/verdict-small-v0/best")`.

## Baseline (before training)

| suite | metric | e5-small |
|---|---|---|
| Banking77 | accuracy | 0.580 |
| SST-5 | accuracy | 0.294 |
| ToxicChat | AUROC | 0.583 |

Training logs land in `runs/<name>/log.json`; the trained checkpoint's numbers go into
`docs/BENCHMARKS.md` with the model name in the `engine` column.
