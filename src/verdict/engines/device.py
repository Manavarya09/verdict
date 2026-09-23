from __future__ import annotations

import os


def pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    env = os.environ.get("VERDICT_DEVICE")
    if env:
        return env
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover
        pass
    return "cpu"
