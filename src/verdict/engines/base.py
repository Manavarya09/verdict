from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..types import Decision


@dataclass
class EngineOutput:
    """Raw, uncalibrated option logits for a batch of decisions.

    ``logits[i]`` has one entry per option of ``decisions[i]`` (for check: [false, true];
    for score: one per scale point)."""

    logits: list[np.ndarray]
    engine: str


class Engine(ABC):
    name: str = "engine"

    @abstractmethod
    def score(self, decisions: list[Decision]) -> EngineOutput: ...

    def warmup(self) -> None:  # pragma: no cover
        pass
