// Package gate decides whether a candidate model may replace the released one.
//
// It compares the candidate's metrics with the metrics recorded for the last release
// (release-metrics.json). Small movements are tolerated so that a harmless retrain does not
// break the build; anything beyond the tolerance fails it.
package gate

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/mgorgiev/sms-phishing-detector/internal/evalmetrics"
)

// Release is the content of release-metrics.json.
type Release struct {
	ModelVersion string             `json:"model_version"`
	Dataset      string             `json:"dataset"`
	Metrics      evalmetrics.Result `json:"metrics"`
}

// Tolerance is how far the candidate may move in the wrong direction.
type Tolerance struct {
	MaxF1Drop  float64
	MaxFPRRise float64
}

// Check is the outcome of one comparison.
type Check struct {
	Name     string  `json:"name"`
	Released float64 `json:"released"`
	Current  float64 `json:"current"`
	Limit    float64 `json:"limit"`
	Passed   bool    `json:"passed"`
}

// Compare returns one check per gated metric.
func Compare(released, current evalmetrics.Result, tol Tolerance) []Check {
	f1Limit := released.MacroF1 - tol.MaxF1Drop
	fprLimit := released.FPR + tol.MaxFPRRise
	return []Check{
		{
			Name:     "macro_f1",
			Released: released.MacroF1,
			Current:  current.MacroF1,
			Limit:    f1Limit,
			Passed:   current.MacroF1 >= f1Limit,
		},
		{
			Name:     "fpr",
			Released: released.FPR,
			Current:  current.FPR,
			Limit:    fprLimit,
			Passed:   current.FPR <= fprLimit,
		},
	}
}

// AllPassed reports whether every check passed.
func AllPassed(checks []Check) bool {
	for _, c := range checks {
		if !c.Passed {
			return false
		}
	}
	return true
}

// LoadRelease reads release-metrics.json.
func LoadRelease(path string) (Release, error) {
	var r Release
	b, err := os.ReadFile(path)
	if err != nil {
		return r, err
	}
	if err := json.Unmarshal(b, &r); err != nil {
		return r, fmt.Errorf("parse %s: %w", path, err)
	}
	return r, nil
}

// WriteRelease writes release-metrics.json.
func WriteRelease(path string, r Release) error {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}
