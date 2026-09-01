from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def ensure_cross_encoder(
    repository: str,
    revision: str,
    local_dir: Path,
) -> Path:
    required = [
        local_dir / "config.json",
        local_dir / "tokenizer.json",
        local_dir / "openvino" / "openvino_model_qint8_quantized.xml",
        local_dir / "openvino" / "openvino_model_qint8_quantized.bin",
    ]
    if not all(path.exists() for path in required):
        from huggingface_hub import snapshot_download

        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=[
                "config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
                "openvino/openvino_model_qint8_quantized.xml",
                "openvino/openvino_model_qint8_quantized.bin",
            ],
        )
    manifest = {
        "repository": repository,
        "revision": revision,
        "license": "Apache-2.0",
        "purpose": "Optional pretrained MS MARCO cross-encoder relevance feature",
    }
    (local_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return local_dir


@dataclass
class OpenVINOCrossEncoder:
    model_dir: str = "models/cross_encoder"
    max_length: int = 64
    batch_size: int = 512
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _compiled: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._compiled is not None:
            return
        import openvino as ov
        from transformers import AutoTokenizer

        directory = Path(self.model_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        quantized = directory / "openvino" / "openvino_model_qint8_quantized.xml"
        fine_tuned = directory / "openvino" / "model.xml"
        xml = fine_tuned if fine_tuned.exists() else quantized
        if not xml.exists():
            raise FileNotFoundError(f"OpenVINO cross-encoder model not found under {directory}")
        self._compiled = ov.Core().compile_model(xml, "CPU")

    def score_pairs(self, queries: Sequence[str], documents: Sequence[str]) -> np.ndarray:
        if len(queries) != len(documents):
            raise ValueError("queries and documents must have equal length")
        self._load()
        scores = np.empty(len(queries), dtype=np.float32)
        input_aliases = {
            alias: port.any_name for port in self._compiled.inputs for alias in port.names
        }
        for start in range(0, len(queries), self.batch_size):
            stop = min(start + self.batch_size, len(queries))
            encoded = self._tokenizer(
                list(queries[start:stop]),
                list(documents[start:stop]),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            inputs = {
                input_aliases[name]: value
                for name, value in encoded.items()
                if name in input_aliases
            }
            output = self._compiled(inputs)[self._compiled.output(0)]
            scores[start:stop] = np.asarray(output).reshape(-1)
        return scores

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        state["_compiled"] = None
        return state
