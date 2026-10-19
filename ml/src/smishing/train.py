"""Fine-tune DistilBERT for 3-class SMS classification.

A plain PyTorch loop rather than the HF Trainer, so every step is visible: tokenise,
batch with dynamic padding, AdamW with linear warmup and decay, mixed precision on GPU,
validation after each epoch, keep the epoch with the best validation macro F1.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from smishing import LABELS
from smishing.metrics import evaluate


@dataclass
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    max_len: int = 128
    batch_size: int = 32
    epochs: int = 3
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(tokenizer, texts, labels, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    items = list(zip(texts, labels, strict=True))

    def collate(batch):
        enc = tokenizer(
            [t for t, _ in batch],
            truncation=True,
            max_length=cfg.max_len,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor([y for _, y in batch])
        return enc

    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(
        items, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate, generator=generator
    )


@torch.no_grad()
def predict_proba(model, tokenizer, texts, max_len: int = 128, batch_size: int = 64) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            list(texts[i : i + batch_size]),
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def train(
    train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: TrainConfig, out_dir: Path
) -> tuple[torch.nn.Module, object, pd.DataFrame]:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label2id = {label: i for i, label in enumerate(LABELS)}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=label2id,
    ).to(device)

    train_loader = _loader(
        tokenizer, train_df["text"], train_df["label"].map(label2id), cfg, shuffle=True
    )

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_f1, best_state = [], -1.0, None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        probs = predict_proba(model, tokenizer, list(val_df["text"]), cfg.max_len)
        preds = [LABELS[i] for i in probs.argmax(axis=1)]
        result = evaluate(val_df["label"], preds)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / len(train_loader),
                "val_macro_f1": result["macro_f1"],
                "val_fpr": result["fpr"],
            }
        )
        if result["macro_f1"] > best_f1:
            best_f1 = result["macro_f1"]
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    pd.Series(asdict(cfg)).to_json(out_dir / "train_config.json", indent=2)
    return model, tokenizer, pd.DataFrame(history)
