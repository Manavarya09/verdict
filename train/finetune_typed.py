"""Fine-tune the full encoder on the typed-decisions train split with the teacher's soft
probabilities as targets (soft cross-entropy). This is the protocol Laya used for its 0.766.

    python -m train.finetune_typed --base runs/verdict-small-v0/best --out runs/verdict-typed-v0 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from bench.typed_decisions import _gold, load
from sentence_transformers import SentenceTransformer

from verdict.engines.embed import EmbedEngine, _prefixes_for
from verdict.server import jev_question_to_verdict, render_state

from .train import encode

SCALE = 20.0


def build(split):
    rows = []
    for r in split:
        qs = json.loads(r["questions"])
        gold = json.loads(r["gold"])
        state = render_state(r["state"])
        for qid, raw in qs.items():
            q = jev_question_to_verdict(raw)
            opts = [EmbedEngine.question_text(q, t) for t in EmbedEngine.option_texts(q)]
            g = gold[qid]
            probs = g.get("probabilities", {})
            if q.kind == "choose":
                labels = q.option_labels
                soft = [float(probs.get(l, 0.0)) for l in labels]
            elif q.kind == "check":
                soft = [float(probs.get("false", 0.0)), float(probs.get("true", 0.0))]
            else:
                soft = [float(probs.get(str(i), 0.0)) for i in range(len(opts))]
            s = sum(soft)
            soft = [x / s for x in soft] if s > 0 else None
            rows.append({"input": state, "options": opts, "gold": _gold(q.kind, g), "soft": soft, "kind": q.kind, "key": f"{r['workflow']}/{qid}"})
    return rows


def gold_index(row):
    if row["kind"] == "check":
        return int(bool(row["gold"]))
    if row["kind"] == "score":
        return int(row["gold"])
    # choose: label -> index in options order (options texts include prompt; recover by position)
    return row["_gold_idx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out", default="runs/verdict-typed-v0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--device", default=None)
    ap.add_argument("--hard", action="store_true", help="use argmax labels instead of soft targets")
    ap.add_argument("--train-embeddings", action="store_true", help="also train the token embedding matrix (memory-heavy on 250k vocabs)")
    a = ap.parse_args()
    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    train, test = load()
    tr_rows, te_rows = build(train), build(test)
    # choose gold -> index
    for rows, split in [(tr_rows, train), (te_rows, test)]:
        i = 0
        for r in split:
            qs = json.loads(r["questions"])
            for raw in qs.values():
                q = jev_question_to_verdict(raw)
                if q.kind == "choose":
                    rows[i]["_gold_idx"] = q.option_labels.index(rows[i]["gold"])
                i += 1
    print(f"train decisions {len(tr_rows)}  test decisions {len(te_rows)}  device {dev}", flush=True)
    model = SentenceTransformer(a.base, device=dev)
    qp, pp = _prefixes_for(a.base)
    if not a.train_embeddings:
        emb = model[0].auto_model.get_input_embeddings()
        emb.weight.requires_grad_(False)
        print(f"frozen token embeddings: {emb.weight.numel()/1e6:.0f}M params", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    total = (len(tr_rows) // a.bs + 1) * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.06 * total))) * max(0.0, (total - s) / total))

    @torch.no_grad()
    def evaluate():
        model.eval()
        # cache state embeddings once
        states = list(dict.fromkeys(r["input"] for r in te_rows))
        E = {}
        for i in range(0, len(states), 32):
            X = encode(model, [qp + s for s in states[i : i + 32]], a.max_len)
            for s, x in zip(states[i : i + 32], X):
                E[s] = x
        ok = {"all": [0, 0], "choose": [0, 0], "check": [0, 0], "score": [0, 0]}
        ocache = {}
        for r in te_rows:
            key = tuple(r["options"])
            if key not in ocache:
                ocache[key] = encode(model, [pp + o for o in r["options"]], 96)
            logits = ocache[key] @ E[r["input"]]
            pred = int(logits.argmax())
            hit = pred == gold_index(r)
            for k in ("all", r["kind"]):
                ok[k][0] += int(hit)
                ok[k][1] += 1
        model.train()
        return {k: round(v[0] / max(1, v[1]), 4) for k, v in ok.items()}

    print("before:", evaluate(), flush=True)
    step = 0
    best = 0.0
    out = Path(a.out)
    for ep in range(a.epochs):
        random.Random(ep).shuffle(tr_rows)
        t0 = time.time()
        for i in range(0, len(tr_rows), a.bs):
            batch = tr_rows[i : i + a.bs]
            X = encode(model, [qp + r["input"] for r in batch], a.max_len)
            uniq = {}
            for r in batch:
                for o in r["options"]:
                    uniq.setdefault(o, len(uniq))
            O = encode(model, [pp + o for o in uniq], 96)
            losses = []
            for j, r in enumerate(batch):
                idx = torch.tensor([uniq[o] for o in r["options"]], device=X.device)
                logits = (O[idx] @ X[j]) * SCALE
                if r["soft"] is not None and not a.hard:
                    target = torch.tensor(r["soft"], device=X.device)
                    losses.append(-(target * F.log_softmax(logits, -1)).sum())
                else:
                    losses.append(F.cross_entropy(logits[None], torch.tensor([gold_index(r)], device=X.device)))
            loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"ep {ep} step {step}/{total} loss {loss.item():.4f} {(time.time()-t0)/(i//a.bs+1):.2f}s/step", flush=True)
        ev = evaluate()
        print(f"EVAL epoch {ep}: {json.dumps(ev)}", flush=True)
        if ev["all"] > best:
            best = ev["all"]
            model.save(str(out / "best"))
            (out / "eval.json").write_text(json.dumps({"epoch": ep, **ev}, indent=2))
    print(f"done. best typed-decisions accuracy {best}", flush=True)


if __name__ == "__main__":
    main()
