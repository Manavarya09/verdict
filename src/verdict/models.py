"""Named Verdict checkpoints and where to get them.

``Verdict(model="verdict-small")`` resolves through this registry: a Hugging Face repo when
one is listed, otherwise a tarball on GitHub Releases, cached under ``~/.cache/verdict``."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from pathlib import Path

import httpx

REGISTRY: dict[str, dict] = {
    "verdict-small": {
        "description": "multilingual-e5-small fine-tuned on the typed-decision mix (train/). 118M params, 100+ languages.",
        "hf": None,  # set when published to the Hub
        "url": "https://github.com/Manavarya09/verdict/releases/download/models-v0/verdict-small-v0.tar.gz",
        "sha256": None,  # filled by train/package.py at release time
        "base": "intfloat/multilingual-e5-small",
    },
}

ALIASES = {"verdict-small-v0": "verdict-small", "small": "verdict-small"}


def cache_dir() -> Path:
    p = Path(os.environ.get("VERDICT_HOME", Path.home() / ".cache" / "verdict"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve(name: str) -> str:
    """Return something sentence-transformers can load: a Hub id or a local directory."""
    key = ALIASES.get(name, name)
    if key not in REGISTRY:
        return name  # a Hub id or a path; pass through
    spec = REGISTRY[key]
    if spec.get("hf"):
        return spec["hf"]
    target = cache_dir() / key
    if (target / "config.json").exists() or (target / "modules.json").exists():
        return str(target)
    url = spec["url"]
    print(f"[verdict] downloading {key} from {url}", flush=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        buf = io.BytesIO()
        h = hashlib.sha256()
        for chunk in r.iter_bytes():
            buf.write(chunk)
            h.update(chunk)
    if spec.get("sha256") and h.hexdigest() != spec["sha256"]:
        raise RuntimeError(f"checksum mismatch for {key}: {h.hexdigest()} != {spec['sha256']}")
    buf.seek(0)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        tf.extractall(target, filter="data")
    # tarballs may contain a single top-level folder
    inner = [p for p in target.iterdir() if p.is_dir()]
    if not (target / "modules.json").exists() and len(inner) == 1 and (inner[0] / "modules.json").exists():
        for p in inner[0].iterdir():
            p.rename(target / p.name)
        inner[0].rmdir()
    return str(target)
