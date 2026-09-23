"""Turn bench JSON files into the Markdown table used in README / docs/BENCHMARKS.md.

    python -m bench.report results/*.json > docs/BENCHMARKS.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLS = [
    ("suite", "suite"),
    ("shots", "shots/class"),
    ("engine", "engine"),
    ("accuracy", "accuracy"),
    ("auroc", "AUROC"),
    ("ece", "ECE"),
    ("automation_rate", "automation @0.9"),
    ("accuracy_when_committed", "acc. when committed"),
    ("fit_s", "fit (s)"),
    ("ms_per_example", "ms/example"),
]


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    if v == -1:
        return "all"
    return str(v)


def main(paths: list[str]) -> None:
    rows = []
    for p in paths:
        rows += json.loads(Path(p).read_text())
    rows.sort(key=lambda r: (r["suite"], r.get("engine", ""), r["shots"] if r["shots"] >= 0 else 10**6))
    print("| " + " | ".join(h for _, h in COLS) + " |")
    print("|" + "---|" * len(COLS))
    for r in rows:
        print("| " + " | ".join(fmt(r.get(k)) for k, _ in COLS) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
