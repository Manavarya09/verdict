"""Cross-encoder engine using a natural-language-inference model.

Scores P(entailment) of "<input> entails <hypothesis about option>". Slower than the
bi-encoder (one forward pass per option) but reads the option *together with* the input,
which matters for ``check`` questions and for options that differ in subtle ways.

Used as a re-ranker over the bi-encoder's top-k, or on its own for small option sets.
"""

from __future__ import annotations

import numpy as np

from ..types import Decision, Question
from .base import Engine, EngineOutput
from .device import pick_device

DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


class NLIEngine(Engine):
    name = "nli"

    def __init__(
        self,
        model: str = DEFAULT_NLI_MODEL,
        device: str | None = None,
        template: str = "This example is about {}.",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.device).eval()
        l2i = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.ent = l2i.get("entailment", 0)
        self.con = l2i.get("contradiction", None)
        self.template = template
        self.batch_size = batch_size
        self.max_length = max_length
        self._torch = torch
        self.name = f"nli:{model.split('/')[-1]}"

    def hypotheses(self, q: Question) -> list[str]:
        if q.kind == "choose":
            return [self.template.format(o.text) for o in q.options]
        if q.kind == "score":
            lo, hi = q.scale  # type: ignore[misc]
            rub = q.rubric or {}
            return [self.template.format(rub.get(i, f"{i} out of {hi}")) for i in range(lo, hi + 1)]
        return [f"It is not the case that {q.claim}", str(q.claim)]

    def _entail_logits(self, premises: list[str], hyps: list[str]) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        for i in range(0, len(premises), self.batch_size):
            enc = self.tok(
                premises[i : i + self.batch_size],
                hyps[i : i + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                lg = self.model(**enc).logits.float()
            e = lg[:, self.ent]
            if self.con is not None:  # entailment vs contradiction log-odds is better calibrated
                e = e - lg[:, self.con]
            out.append(e.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def score(self, decisions: list[Decision]) -> EngineOutput:
        premises: list[str] = []
        hyps: list[str] = []
        spans: list[tuple[int, int]] = []
        for d in decisions:
            h = self.hypotheses(d.question)
            spans.append((len(premises), len(premises) + len(h)))
            premises.extend([d.rendered_input()] * len(h))
            hyps.extend(h)
        flat = self._entail_logits(premises, hyps)
        return EngineOutput(logits=[flat[a:b] for a, b in spans], engine=self.name)

    def score_subset(self, decision: Decision, option_idx: list[int]) -> np.ndarray:
        h = self.hypotheses(decision.question)
        sub = [h[i] for i in option_idx]
        return self._entail_logits([decision.rendered_input()] * len(sub), sub)
