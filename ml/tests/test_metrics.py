import pytest

from smishing.metrics import evaluate


def test_fpr_counts_ham_flagged_as_anything_else():
    y_true = ["ham", "ham", "ham", "ham", "spam", "smishing"]
    y_pred = ["ham", "ham", "spam", "smishing", "spam", "smishing"]

    result = evaluate(y_true, y_pred)

    assert result["fpr"] == pytest.approx(0.5)


def test_macro_f1_is_unweighted_mean_over_classes():
    y_true = ["ham", "ham", "spam", "smishing"]
    y_pred = ["ham", "ham", "spam", "spam"]

    result = evaluate(y_true, y_pred)

    # ham: P=1 R=1 F1=1; spam: P=0.5 R=1 F1=2/3; smishing: F1=0
    assert result["per_class"]["ham"]["f1"] == pytest.approx(1.0)
    assert result["per_class"]["spam"]["f1"] == pytest.approx(2 / 3)
    assert result["per_class"]["smishing"]["f1"] == pytest.approx(0.0)
    assert result["macro_f1"] == pytest.approx((1 + 2 / 3 + 0) / 3)
    assert result["confusion"] == [[2, 0, 0], [0, 1, 0], [0, 1, 0]]


def test_perfect_predictions():
    labels = ["ham", "spam", "smishing", "ham"]
    result = evaluate(labels, labels)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["fpr"] == 0.0
