"""TF-IDF + logistic regression baseline.

Word n-grams catch phrases like "verify your account"; character n-grams catch obfuscated
spellings and link fragments ("bit.do", "fr33"). The regularisation strength C is picked on
the validation set; the test set is not touched here.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from smishing.metrics import evaluate


def build(c: float = 10.0) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
                ),
            ),
        ]
    )
    clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def fit_with_validation(
    train: pd.DataFrame, val: pd.DataFrame, cs: tuple[float, ...] = (1.0, 10.0, 100.0)
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit one model per C, keep the one with the best validation macro F1."""
    rows, best, best_f1 = [], None, -1.0
    for c in cs:
        model = build(c).fit(train["text"], train["label"])
        result = evaluate(val["label"], model.predict(val["text"]))
        rows.append({"C": c, "val_macro_f1": result["macro_f1"], "val_fpr": result["fpr"]})
        if result["macro_f1"] > best_f1:
            best, best_f1 = model, result["macro_f1"]
    return best, pd.DataFrame(rows)
