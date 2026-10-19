package evalmetrics

import (
	"math"
	"reflect"
	"testing"
)

func approx(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("%s = %v, want %v", name, got, want)
	}
}

// The same cases as ml/tests/test_metrics.py, so both implementations agree.

func TestFPRCountsHamFlaggedAsAnythingElse(t *testing.T) {
	yTrue := []string{"ham", "ham", "ham", "ham", "spam", "smishing"}
	yPred := []string{"ham", "ham", "spam", "smishing", "spam", "smishing"}

	res, err := Evaluate(yTrue, yPred)
	if err != nil {
		t.Fatal(err)
	}
	approx(t, "fpr", res.FPR, 0.5)
}

func TestMacroF1IsUnweightedMeanOverClasses(t *testing.T) {
	yTrue := []string{"ham", "ham", "spam", "smishing"}
	yPred := []string{"ham", "ham", "spam", "spam"}

	res, err := Evaluate(yTrue, yPred)
	if err != nil {
		t.Fatal(err)
	}
	approx(t, "ham f1", res.PerClass["ham"].F1, 1)
	approx(t, "spam f1", res.PerClass["spam"].F1, 2.0/3)
	approx(t, "smishing f1", res.PerClass["smishing"].F1, 0)
	approx(t, "macro f1", res.MacroF1, (1+2.0/3+0)/3)

	want := [][]int{{2, 0, 0}, {0, 1, 0}, {0, 1, 0}}
	if !reflect.DeepEqual(res.Confusion, want) {
		t.Errorf("confusion = %v, want %v", res.Confusion, want)
	}
}

func TestPerfectPredictions(t *testing.T) {
	labels := []string{"ham", "spam", "smishing", "ham"}
	res, err := Evaluate(labels, labels)
	if err != nil {
		t.Fatal(err)
	}
	approx(t, "macro f1", res.MacroF1, 1)
	approx(t, "fpr", res.FPR, 0)
}

func TestRejectsUnknownLabelsAndLengthMismatch(t *testing.T) {
	if _, err := Evaluate([]string{"ham"}, []string{"phish"}); err == nil {
		t.Error("expected error for unknown predicted label")
	}
	if _, err := Evaluate([]string{"ham", "spam"}, []string{"ham"}); err == nil {
		t.Error("expected error for length mismatch")
	}
}
