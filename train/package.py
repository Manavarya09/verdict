"""Package a trained checkpoint for release: tar.gz + sha256, and print the registry entry.

    python -m train.package runs/verdict-small-v0/best verdict-small-v0
    gh release create models-v0 dist/verdict-small-v0.tar.gz --title "verdict-small v0" --notes-file dist/verdict-small-v0.md
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path


def main(src: str, name: str) -> None:
    src_p = Path(src)
    out = Path("dist")
    out.mkdir(exist_ok=True)
    tar_p = out / f"{name}.tar.gz"
    with tarfile.open(tar_p, "w:gz") as tf:
        for p in sorted(src_p.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(src_p)))
    sha = hashlib.sha256(tar_p.read_bytes()).hexdigest()
    log = json.loads((src_p.parent / "log.json").read_text()) if (src_p.parent / "log.json").exists() else []
    best = max(log, key=lambda r: r["eval"]["mean"]) if log else None
    notes = out / f"{name}.md"
    notes.write_text(
        f"# {name}\n\n"
        f"Base: intfloat/multilingual-e5-small. Trained with `python -m train.train` on the mix in `train/data.py` "
        f"(Banking77, SST-5, ToxicChat held out).\n\n"
        + (f"Best held-out (zero-shot, n=500): `{json.dumps(best['eval'])}` at step {best['step']}.\n\n" if best else "")
        + f"sha256: `{sha}`\n\nLoad with `Verdict(model=\"{name.rsplit('-v', 1)[0]}\")`.\n"
    )
    print(json.dumps({"tar": str(tar_p), "sha256": sha, "size_mb": round(tar_p.stat().st_size / 1e6, 1), "best": best}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
