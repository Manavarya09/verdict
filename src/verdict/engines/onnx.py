"""ONNX Runtime engine: the same bi-encoder without PyTorch.

``pip install "verdictml[onnx]"`` pulls only ``onnxruntime`` + ``tokenizers``. Uses the
int8 export that ships in the ``Xenova/*`` mirrors on the Hub (``onnx/model_quantized.onnx``),
so the download is ~120 MB and a decision costs a few milliseconds on a laptop CPU. The
browser playground uses exactly these weights, so numbers match across Python and JS."""

from __future__ import annotations

import contextlib

import numpy as np

from .embed import EmbedEngine, _prefixes_for

DEFAULT_ONNX_MODEL = "Xenova/multilingual-e5-small"


class OnnxEmbedEngine(EmbedEngine):
    name = "onnx"

    def __init__(
        self,
        model: str = DEFAULT_ONNX_MODEL,
        file: str = "onnx/model_quantized.onnx",
        scale: float = 20.0,
        batch_size: int = 32,
        max_seq_length: int = 512,
        threads: int | None = None,
    ):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.model_name = model
        self.device = "cpu"
        self.scale = float(scale)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.q_prefix, self.p_prefix = _prefixes_for(model)
        self._option_cache: dict[str, np.ndarray] = {}
        from collections import OrderedDict

        self._input_cache = OrderedDict()
        self.input_cache_size = 4096
        self.name = f"onnx:{model.split('/')[-1]}"
        from pathlib import Path

        local = Path(model)
        if local.is_dir():  # an exported checkpoint (train/export_onnx.py) or unpacked release
            path = str(local / file)
            tok_src = str(local / "tokenizer.json")
        else:
            path = hf_hub_download(model, file)
            # the quantized graph may reference external data next to it
            with contextlib.suppress(Exception):
                hf_hub_download(model, file + "_data")
            tok_src = None
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tok = Tokenizer.from_file(tok_src) if tok_src else Tokenizer.from_pretrained(model)
        self.tok.enable_truncation(max_seq_length)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("<pad>") or 0, pad_token="<pad>")
        self._dim: int | None = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok.encode_batch(texts[i : i + self.batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feed)[0]  # [B, L, D]
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9
            out.append(pooled.astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def _embed_inputs_uncached(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.q_prefix + t for t in texts])

    def embed_options(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._option_cache]
        if missing:
            for t, v in zip(missing, self._encode([self.p_prefix + t for t in missing])):
                self._option_cache[t] = v
        return np.stack([self._option_cache[t] for t in texts])

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._encode(["x"]).shape[1])
        return self._dim
