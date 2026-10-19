package gate

import (
	"path/filepath"
	"testing"

	"github.com/mgorgiev/sms-phishing-detector/internal/evalmetrics"
)

var tol = Tolerance{MaxF1Drop: 0.01, MaxFPRRise: 0.005}

func released() evalmetrics.Result {
	return evalmetrics.Result{MacroF1: 0.95, FPR: 0.010}
}

func TestPassesWhenMetricsHold(t *testing.T) {
	checks := Compare(released(), evalmetrics.Result{MacroF1: 0.95, FPR: 0.010}, tol)
	if !AllPassed(checks) {
		t.Fatalf("expected pass, got %+v", checks)
	}
}

func TestPassesWithinTolerance(t *testing.T) {
	checks := Compare(released(), evalmetrics.Result{MacroF1: 0.945, FPR: 0.013}, tol)
	if !AllPassed(checks) {
		t.Fatalf("expected pass within tolerance, got %+v", checks)
	}
}

func TestFailsWhenF1Drops(t *testing.T) {
	checks := Compare(released(), evalmetrics.Result{MacroF1: 0.93, FPR: 0.010}, tol)
	if AllPassed(checks) {
		t.Fatal("expected failure on F1 drop")
	}
	if checks[0].Passed || !checks[1].Passed {
		t.Errorf("only macro_f1 should fail: %+v", checks)
	}
}

func TestFailsWhenFPRRises(t *testing.T) {
	checks := Compare(released(), evalmetrics.Result{MacroF1: 0.96, FPR: 0.02}, tol)
	if AllPassed(checks) {
		t.Fatal("expected failure on FPR rise")
	}
	if !checks[0].Passed || checks[1].Passed {
		t.Errorf("only fpr should fail: %+v", checks)
	}
}

func TestReleaseRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "release-metrics.json")
	want := Release{ModelVersion: "v1", Dataset: "test.csv", Metrics: released()}
	if err := WriteRelease(path, want); err != nil {
		t.Fatal(err)
	}
	got, err := LoadRelease(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.ModelVersion != want.ModelVersion || got.Metrics.MacroF1 != want.Metrics.MacroF1 {
		t.Errorf("round trip mismatch: got %+v", got)
	}
}
