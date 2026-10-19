"""Runs against the real exported model. Set MODEL_DIR to a directory with model.onnx,
tokenizer.json and labels.json, for example ``MODEL_DIR=model uv run pytest -m model``."""

import os
from pathlib import Path

import numpy as np
import pytest

from app.inference import Classifier

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif("MODEL_DIR" not in os.environ, reason="MODEL_DIR not set"),
]


@pytest.fixture(scope="module")
def classifier() -> Classifier:
    return Classifier(Path(os.environ["MODEL_DIR"]))


def test_probabilities_sum_to_one(classifier):
    probs = classifier.predict_proba(["hi, are we still on for lunch?", "WIN a free iPhone now"])
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_obvious_messages(classifier):
    (ham_label, _), (phish_label, _) = classifier.predict(
        [
            "Ok, see you at 7 at the usual place",
            "Your bank account has been suspended. Verify your details now at http://secure-bank-verify.co/login",
        ]
    )
    assert ham_label == "ham"
    assert phish_label == "smishing"
