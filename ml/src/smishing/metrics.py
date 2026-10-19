"""Evaluation metrics shared by the notebook and mirrored by the Go regression gate.

False-positive rate (FPR) here means: the share of legitimate (ham) messages that the model
labels as spam or smishing. In a messaging security product that is the number of real
messages a customer loses, so it is tracked next to macro F1 rather than folded into it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from smishing import LABELS


def evaluate(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    ham = LABELS.index("ham")
    ham_total = cm[ham].sum()
    fpr = float((ham_total - cm[ham, ham]) / ham_total) if ham_total else 0.0

    return {
        "macro_f1": float(np.mean(f1)),
        "fpr": fpr,
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "confusion": cm.tolist(),
    }


def summary_row(name: str, result: dict) -> dict:
    """One flat row for a results table."""
    return {
        "model": name,
        "macro_f1": round(result["macro_f1"], 4),
        "fpr": round(result["fpr"], 4),
        "smishing_recall": round(result["per_class"]["smishing"]["recall"], 4),
        "smishing_precision": round(result["per_class"]["smishing"]["precision"], 4),
        "spam_f1": round(result["per_class"]["spam"]["f1"], 4),
    }
