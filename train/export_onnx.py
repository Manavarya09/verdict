"""Export a Verdict checkpoint to ONNX (fp32 + dynamic int8) in the layout transformers.js
and the ONNX engine expect: ``<out>/onnx/model.onnx``, ``<out>/onnx/model_quantized.onnx``,
plus tokenizer and config files at the root.

    python -m train.export_onnx runs/verdict-small-v0/best dist/verdict-small-v0-onnx
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main(src: str, out: str) -> None:
    src_p, out_p = Path(src), Path(out)
    onnx_dir = out_p / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "optimum.exporters.onnx", "--model", str(src_p), "--task", "feature-extraction", "--opset", "17", str(onnx_dir)],
        check=True,
    )
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(onnx_dir / "model.onnx"), str(onnx_dir / "model_quantized.onnx"), weight_type=QuantType.QInt8)
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"]:
        for cand in [src_p / f, onnx_dir / f]:
            if cand.exists():
                shutil.copy(cand, out_p / f)
                break
    for f in onnx_dir.iterdir():
        if f.suffix != ".onnx" and not f.name.endswith(".onnx_data"):
            f.unlink()
    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in onnx_dir.iterdir()}
    print({"out": str(out_p), "onnx_mb": sizes})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
