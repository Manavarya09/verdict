"""Train the Verdict encoder: a bi-encoder whose input/option cosine, scaled by 20, is a
calibrated logit over the question's options. Same mechanism as inference, so the trained
model drops into ``Verdict(model=path)`` and into the browser build unchanged.

    python -m train.train --out runs/verdict-small-v0 --epochs 2

Held-out zero-shot evals (Banking77, SST-5, ToxicChat) run every ``--eval-every`` steps and
the best checkpoint by mean held-out score is kept."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .data import Row, build_mix

SCALE = 20.0


def batches_by_task(rows: list[Row], bs: int, seed: int):
    """Yield batches that share one task (so options are encoded once per batch)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r.task].append(r)
    out = []
    for rs in by.values():
        rng.shuffle(rs)
        for i in range(0, len(rs), bs):
            out.append(rs[i : i + bs])
    rng.shuffle(out)
    return out


def encode(model: SentenceTransformer, texts: list[str], max_len: int) -> torch.Tensor:
    model.max_seq_length = max_len
    feats = model.tokenize(texts)
    feats = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in feats.items()}
    return F.normalize(model(feats)["sentence_embedding"], dim=-1)


def step_loss(model, batch: list[Row], q_prefix: str, p_prefix: str, max_len: int):
    X = encode(model, [q_prefix + r.input for r in batch], max_len)
    # option texts: shared per task for choose/score, per-example for check-with-claim
    uniq: dict[str, int] = {}
    for r in batch:
        for o in r.options:
            uniq.setdefault(o, len(uniq))
    O = encode(model, [p_prefix + o for o in uniq], 96)
    losses = []
    correct = 0
    for i, r in enumerate(batch):
        idx = torch.tensor([uniq[o] for o in r.options], device=X.device)
        logits = (O[idx] @ X[i]) * SCALE
        losses.append(F.cross_entropy(logits[None], torch.tensor([r.gold], device=X.device)))
        correct += int(logits.argmax().item() == r.gold)
    return torch.stack(losses).mean(), correct / len(batch)


@torch.no_grad()
def heldout_eval(model_path_or_obj, n: int = 500) -> dict:
    """Zero-shot on the three held-out suites using the real inference path."""
    from bench.suites import SUITES

    from verdict import Verdict

    v = Verdict(model=model_path_or_obj)
    out = {}
    for name in ["banking77", "sst5", "toxic_chat"]:
        q, _, calib, test = SUITES[name](n_test=n)
        d = v.compile(q)
        d.calibrate(calib[:300])
        ev = d.evaluate(test)
        out[name] = round(ev.get("auroc", ev["accuracy"]), 4)
    out["mean"] = round(float(np.mean([out["banking77"], out["sst5"], out["toxic_chat"]])), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out", default="runs/verdict-small-v0")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--scale", type=float, default=1.0, help="data mix multiplier")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--commercial", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} base={a.base}", flush=True)

    print("building mix...", flush=True)
    rows = build_mix(scale=a.scale, seed=a.seed, commercial=a.commercial)
    print(f"total {len(rows)} rows", flush=True)
    (out / "mix.json").write_text(json.dumps({"n": len(rows), "tasks": dict(sorted(((t, sum(r.task == t for r in rows)) for t in {r.task for r in rows}), key=lambda x: -x[1]))}, indent=2))

    model = SentenceTransformer(a.base, device=dev)
    from verdict.engines.embed import _prefixes_for

    q_prefix, p_prefix = _prefixes_for(a.base)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(rows) / a.bs)
    total = int(steps_per_epoch * a.epochs)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)) if s >= warm else (s + 1) / warm)

    print("baseline held-out:", flush=True)
    base_eval = heldout_eval(a.base, n=a.eval_n)
    print(json.dumps(base_eval), flush=True)
    best = base_eval["mean"]
    log = [{"step": 0, "eval": base_eval}]
    (out / "log.json").write_text(json.dumps(log, indent=2))

    step = 0
    t0 = time.time()
    run_loss, run_acc, seen = 0.0, 0.0, 0
    epoch = 0
    while step < total:
        for batch in batches_by_task(rows, a.bs, a.seed + epoch):
            if step >= total:
                break
            model.train()
            loss, acc = step_loss(model, batch, q_prefix, p_prefix, a.max_len)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            run_acc += acc
            seen += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"step {step}/{total} loss {run_loss/seen:.4f} acc {run_acc/seen:.3f} lr {sched.get_last_lr()[0]:.2e} {el/step:.2f}s/step eta {(total-step)*el/step/60:.0f}m", flush=True)
                run_loss, run_acc, seen = 0.0, 0.0, 0
            if step % a.eval_every == 0 or step == total:
                model.eval()
                tmp = out / "latest"
                model.save(str(tmp))
                ev = heldout_eval(str(tmp), n=a.eval_n)
                log.append({"step": step, "eval": ev})
                (out / "log.json").write_text(json.dumps(log, indent=2))
                flag = ""
                if ev["mean"] > best:
                    best = ev["mean"]
                    model.save(str(out / "best"))
                    flag = "  <- best"
                print(f"EVAL step {step}: {json.dumps(ev)}{flag}", flush=True)
        epoch += 1
    print(f"done. best held-out mean {best}. saved to {out/'best'}", flush=True)


if __name__ == "__main__":
    main()
