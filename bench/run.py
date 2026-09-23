"""Reproducible benchmark. ``python -m bench.run banking77 --shots 0,16,-1``.

Protocol (no leakage): heads are fit on the train split minus a fixed 1,000-row calibration
slice; temperature and conformal threshold are fit on that calibration slice only; every
number is reported on the untouched test split."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from verdict import Verdict

from .suites import SUITES


def _few_shot(train, shots: int):
    by = defaultdict(list)
    for x, y in train:
        by[y].append(x)
    return [(x, y) for y, xs in by.items() for x in xs[:shots]]


def run_bench(suite: str, n: int = 1000, shots=(0, 16), model=None, reranker=None, coverage=0.9) -> list[dict]:
    names = list(SUITES) if suite == "all" else [suite]
    v = Verdict(**({"model": model} if model else {}), reranker=reranker)
    results = []
    for name in names:
        q, train, calib, test = SUITES[name](n_test=n)
        d = v.compile(q)
        for s in shots:
            d.head = None
            d.calibrated = False
            fit_s = 0.0
            if s != 0:
                sub = train if s < 0 else _few_shot(train, s)
                t0 = time.perf_counter()
                d.fit(sub)
                fit_s = time.perf_counter() - t0
            d.calibrate(calib, coverage=coverage)
            t0 = time.perf_counter()
            ev = d.evaluate(test)
            ms = (time.perf_counter() - t0) * 1000 / len(test)
            row = {"suite": name, "shots": s, "engine": v.engine.name, "fit_s": round(fit_s, 2), "ms_per_example": round(ms, 2), **ev}
            print(json.dumps(row), flush=True)
            results.append(row)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", default="banking77")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--shots", default="0,16")
    ap.add_argument("--model")
    ap.add_argument("--reranker")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = run_bench(a.suite, a.n, [int(x) for x in a.shots.split(",")], a.model, a.reranker)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
