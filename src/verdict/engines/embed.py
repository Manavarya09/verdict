"""Bi-encoder engine: embed the input once, embed every option once, score by cosine.

Why this is the default:
* option count is unbounded (10 or 10,000 options cost the same per input);
* options are embedded once and cached, so a decision is one forward pass;
* a small multilingual model (118M) fits in ~120 MB int8 and runs on CPU or in a browser;
* a supervised head (``verdict.heads``) trained on the same embeddings turns this into a
  fast, strong classifier from a few dozen examples.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from ..types import Decision, Question
from .base import Engine, EngineOutput
from .device import pick_device

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-small"

# Prefixes expected by the e5 family; other models get empty prefixes.
_PREFIXES = {
    "e5": ("query: ", "passage: "),
}


def _prefixes_for(model_name: str) -> tuple[str, str]:
    for key, pair in _PREFIXES.items():
        if key in model_name.lower():
            return pair
    return ("", "")


class EmbedEngine(Engine):
    name = "embed"

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        device: str | None = None,
        scale: float = 20.0,
        batch_size: int = 32,
        max_seq_length: int | None = 512,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.device = pick_device(device)
        self.model = SentenceTransformer(model, device=self.device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.scale = float(scale)  # cosine in [-1,1] -> logits; 20 is the usual SBERT scale
        self.batch_size = batch_size
        self.q_prefix, self.p_prefix = _prefixes_for(model)
        self._option_cache: dict[str, np.ndarray] = {}
        self.name = f"embed:{model.split('/')[-1]}"

    # ---- embedding ------------------------------------------------------------
    def _chunks(self, text: str) -> list[str]:
        """Split a long input into windows that fit the encoder; short inputs pass through.
        Windows overlap by ~10% so a decision-bearing sentence is not cut in half."""
        limit = self.model.max_seq_length or 512
        tok = self.model.tokenizer
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        budget = limit - 8
        if len(ids) <= budget:
            return [text]
        step = int(budget * 0.9)
        return [tok.decode(ids[i : i + budget]) for i in range(0, len(ids), step)]

    def embed_inputs(self, texts: list[str]) -> np.ndarray:
        """Long inputs are chunked, embedded, mean-pooled and re-normalised, so a 20-page
        document is one vector instead of a silently truncated one."""
        chunked: list[str] = []
        spans: list[tuple[int, int]] = []
        for t in texts:
            cs = self._chunks(t)
            spans.append((len(chunked), len(chunked) + len(cs)))
            chunked.extend(cs)
        E = self.model.encode(
            [self.q_prefix + t for t in chunked],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if len(chunked) == len(texts):
            return E
        out = np.stack([E[a:b].mean(0) for a, b in spans])
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)

    def embed_options(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._option_cache]
        if missing:
            vecs = self.model.encode(
                [self.p_prefix + t for t in missing],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for t, v in zip(missing, vecs):
                self._option_cache[t] = v
        return np.stack([self._option_cache[t] for t in texts])

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    # ---- option rendering -------------------------------------------------------
    @staticmethod
    def option_texts(q: Question) -> list[str]:
        if q.kind == "choose":
            return [o.text for o in q.options]
        if q.kind == "score":
            lo, hi = q.scale  # type: ignore[misc]
            rub = q.rubric or {}
            return [rub.get(i, f"{i} out of {hi}") for i in range(lo, hi + 1)]
        # check: [false, true]
        return [f"It is false that {q.claim}", f"It is true that {q.claim}"]

    @staticmethod
    def question_text(q: Question, option_text: str) -> str:
        return f"{q.prompt} {option_text}" if q.prompt else option_text

    # ---- scoring -----------------------------------------------------------------
    def score(self, decisions: list[Decision]) -> EngineOutput:
        inputs = self.embed_inputs([d.rendered_input() for d in decisions])
        out: list[np.ndarray] = []
        for d, x in zip(decisions, inputs):
            opts = [self.question_text(d.question, t) for t in self.option_texts(d.question)]
            O = self.embed_options(opts)
            out.append((O @ x) * self.scale)
        return EngineOutput(logits=out, engine=self.name)

    def score_with_embeddings(self, decisions: list[Decision]) -> tuple[EngineOutput, np.ndarray]:
        """Same as :meth:`score` but also returns the input embeddings (for heads)."""
        inputs = self.embed_inputs([d.rendered_input() for d in decisions])
        out: list[np.ndarray] = []
        for d, x in zip(decisions, inputs):
            opts = [self.question_text(d.question, t) for t in self.option_texts(d.question)]
            O = self.embed_options(opts)
            out.append((O @ x) * self.scale)
        return EngineOutput(logits=out, engine=self.name), inputs

    def fingerprint(self) -> str:
        return hashlib.sha1(f"{self.model_name}|{self.dim}".encode()).hexdigest()[:12]


@lru_cache(maxsize=4)
def shared_embed_engine(model: str = DEFAULT_EMBED_MODEL, device: str | None = None) -> EmbedEngine:
    return EmbedEngine(model=model, device=device)
