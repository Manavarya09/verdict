"""The typed-decisions benchmark (LocalLLaMA/typed-decisions): 4 workflows, 5 Jev-format
questions per case, 1,200 train / 400 test cases = 2,000 test decisions. Laya reports 0.766
(fine-tuned on this train split), Jev 0.727, majority 0.461, random 0.318.

We report two rows: zero-shot (nothing fit) and fitted (heads on the train split minus a
calibration slice, calibration on that slice, scored on test), which is Laya's protocol.

    python -m bench.typed_decisions --model intfloat/multilingual-e5-small
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np

from verdict import Verdict
from verdict.server import jev_question_to_verdict


def load():
    from datasets import load_dataset

    ds = load_dataset("LocalLLaMA/typed-decisions", "all")
    return ds["train"], ds["test"]


def _gold(kind: str, g: dict):
    if kind == "check":
        return g["label"] in ("true", True)
    if kind == "score":
        return int(g["label"])
    return g["label"]


def run(model: str | None = None, device: str | None = None, coverage: float = 0.9, calib_per_wf: int = 60) -> list[dict]:
    train, test = load()
    v = Verdict(**({"model": model} if model else {}), device=device)
    # group rows by (workflow, qid): one Decider each
    groups: dict[tuple[str, str], dict] = {}
    for split_name, split in [("train", train), ("test", test)]:
        for r in split:
            qs = json.loads(r["questions"])
            gold = json.loads(r["gold"])
            for qid, raw in qs.items():
                key = (r["workflow"], qid)
                g = groups.setdefault(key, {"raw": raw, "train": [], "test": []})
                q = jev_question_to_verdict(raw)
                g["q"] = q
                g[split_name].append((r["state"], _gold(q.kind, gold[qid])))
    rows = []
    for mode in ["zero-shot", "fitted"]:
        correct = defaultdict(int)
        total = defaultdict(int)
        committed = 0
        correct_committed = 0
        t0 = time.perf_counter()
        n_dec = 0
        for (wf, _qid), g in groups.items():
            d = v.compile(g["q"])
            d.head = None
            d.calibrated = False
            tr = list(g["train"])
            rng = np.random.default_rng(0)
            rng.shuffle(tr)
            cal, fit = tr[:calib_per_wf], tr[calib_per_wf:]
            if mode == "fitted":
                d.fit(fit)
            d.calibrate(cal, coverage=coverage)
            answers = d.batch([s for s, _ in g["test"]])
            for a, (_, y) in zip(answers, g["test"]):
                pred = a.label if g["q"].kind == "choose" else (a.value if g["q"].kind == "score" else a.verdict)
                ok = pred == y
                for k in ("all", f"wf:{wf}", f"kind:{g['q'].kind}"):
                    total[k] += 1
                    correct[k] += int(ok)
                n_dec += 1
                if not a.confidence.abstain:
                    committed += 1
                    correct_committed += int(ok)
        ms = (time.perf_counter() - t0) * 1000 / n_dec
        row = {
            "suite": "typed_decisions",
            "mode": mode,
            "engine": v.engine.name,
            "decisions": n_dec,
            "accuracy": correct["all"] / total["all"],
            "automation_rate": committed / n_dec,
            "accuracy_when_committed": correct_committed / max(1, committed),
            "ms_per_decision": round(ms, 2),
            **{k: round(correct[k] / total[k], 4) for k in sorted(total) if k != "all"},
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--device")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = run(a.model, a.device)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
