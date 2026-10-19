"""ONNX inference for the SMS classifier.

There is no torch at runtime. The tokenizer is the Rust ``tokenizers`` library reading the
tokenizer.json saved at training time, and the model runs on ONNX Runtime. The model
directory holds model.onnx, tokenizer.json and labels.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class Classifier:
    def __init__(self, model_dir: Path, max_len: int = 128):
        self.labels: list[str] = json.loads((model_dir / "labels.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_len)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        feeds = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        (logits,) = self.session.run(["logits"], feeds)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, texts: list[str]) -> list[tuple[str, dict[str, float]]]:
        probs = self.predict_proba(texts)
        return [
            (self.labels[int(row.argmax())], dict(zip(self.labels, map(float, row), strict=True)))
            for row in probs
        ]
