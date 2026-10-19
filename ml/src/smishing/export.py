"""Export the fine-tuned model to ONNX, check it against PyTorch, and quantise it to int8.

The output is a "bundle" directory in the layout the model service loads:
model.onnx, tokenizer.json and labels.json.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class _LogitsOnly(torch.nn.Module):
    """Wrap the HF model so the exported graph has plain tensor inputs and one output."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_onnx(model_dir: Path, bundle_dir: Path) -> Path:
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sample = tokenizer(
        ["short one", "a second, noticeably longer example message"],
        padding=True,
        return_tensors="pt",
    )

    bundle_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = bundle_dir / "model.onnx"
    torch.onnx.export(
        _LogitsOnly(model),
        (sample["input_ids"], sample["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    shutil.copy(model_dir / "tokenizer.json", bundle_dir / "tokenizer.json")
    id2label = model.config.id2label
    labels = [id2label[i] for i in range(len(id2label))]
    (bundle_dir / "labels.json").write_text(json.dumps(labels))
    return onnx_path


def quantize_int8(fp32_bundle: Path, int8_bundle: Path) -> Path:
    """Dynamic int8 quantisation: weights stored as int8, activations quantised at runtime."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_bundle.mkdir(parents=True, exist_ok=True)
    out = int8_bundle / "model.onnx"
    quantize_dynamic(str(fp32_bundle / "model.onnx"), str(out), weight_type=QuantType.QInt8)
    for name in ("tokenizer.json", "labels.json"):
        shutil.copy(fp32_bundle / name, int8_bundle / name)
    return out


def logits_parity(model_dir: Path, bundle_dir: Path, texts: list[str], max_len: int = 128) -> float:
    """Largest absolute difference between PyTorch and ONNX Runtime logits on the same input."""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    with torch.no_grad():
        expected = model(**enc).logits.numpy()

    session = ort.InferenceSession(
        str(bundle_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(
        ["logits"],
        {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()},
    )
    return float(np.abs(expected - actual).max())


def tokenizer_parity(
    model_dir: Path, bundle_dir: Path, texts: list[str], max_len: int = 128
) -> int:
    """Number of texts where the service tokenizer (tokenizers + tokenizer.json) disagrees with
    the training tokenizer (transformers). Must be zero."""
    hf = AutoTokenizer.from_pretrained(model_dir)
    rust = Tokenizer.from_file(str(bundle_dir / "tokenizer.json"))
    rust.enable_truncation(max_length=max_len)
    mismatches = 0
    for text in texts:
        expected = hf(text, truncation=True, max_length=max_len)["input_ids"]
        if rust.encode(text).ids != expected:
            mismatches += 1
    return mismatches
