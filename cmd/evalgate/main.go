// Command evalgate is the model regression gate.
//
// It sends every message of the held-out test set through the running gateway, computes
// macro F1 and the false-positive rate, and compares them with release-metrics.json.
// Exit codes: 0 pass, 1 regression, 2 could not run.
//
// With -write-release it records the current metrics as the new release baseline instead.
package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"text/tabwriter"
	"time"

	"github.com/mgorgiev/sms-phishing-detector/internal/evalmetrics"
	"github.com/mgorgiev/sms-phishing-detector/internal/gate"
)

type sample struct {
	Text  string
	Label string
}

func main() {
	var (
		dataPath     = flag.String("data", "data/splits/test.csv", "held-out test set (CSV with text,label)")
		gatewayURL   = flag.String("url", "http://localhost:8080", "gateway base URL")
		releasePath  = flag.String("release", "release-metrics.json", "metrics of the released model")
		reportPath   = flag.String("report", "report.json", "where to write the evaluation report")
		workers      = flag.Int("workers", 8, "concurrent requests")
		maxF1Drop    = flag.Float64("max-f1-drop", 0.01, "allowed drop in macro F1")
		maxFPRRise   = flag.Float64("max-fpr-rise", 0.005, "allowed rise in false-positive rate")
		waitFor      = flag.Duration("wait", 60*time.Second, "how long to wait for the gateway to come up")
		writeRelease = flag.Bool("write-release", false, "record current metrics as the release baseline")
		modelVersion = flag.String("model-version", "", "model version to record with -write-release")
	)
	flag.Parse()

	if err := run(*dataPath, *gatewayURL, *releasePath, *reportPath, *workers,
		gate.Tolerance{MaxF1Drop: *maxF1Drop, MaxFPRRise: *maxFPRRise},
		*waitFor, *writeRelease, *modelVersion); err != nil {
		var regression errRegression
		if errors.As(err, &regression) {
			fmt.Fprintln(os.Stderr, "GATE FAILED:", err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stderr, "evalgate:", err)
		os.Exit(2)
	}
}

type errRegression struct{ failed []string }

func (e errRegression) Error() string { return fmt.Sprintf("regression in %v", e.failed) }

func run(dataPath, gatewayURL, releasePath, reportPath string, workers int, tol gate.Tolerance,
	waitFor time.Duration, writeRelease bool, modelVersion string) error {
	samples, err := readSamples(dataPath)
	if err != nil {
		return err
	}
	client := &http.Client{Timeout: 10 * time.Second}
	if err := waitHealthy(client, gatewayURL, waitFor); err != nil {
		return err
	}

	start := time.Now()
	preds, err := screenAll(client, gatewayURL, samples, workers)
	if err != nil {
		return err
	}
	elapsed := time.Since(start)

	yTrue := make([]string, len(samples))
	for i, s := range samples {
		yTrue[i] = s.Label
	}
	current, err := evalmetrics.Evaluate(yTrue, preds)
	if err != nil {
		return err
	}
	fmt.Printf("screened %d messages in %s with %d workers\n\n", len(samples), elapsed.Round(time.Millisecond), workers)
	printPerClass(current)

	if writeRelease {
		if modelVersion == "" {
			return errors.New("-write-release needs -model-version")
		}
		rel := gate.Release{ModelVersion: modelVersion, Dataset: dataPath, Metrics: current}
		if err := gate.WriteRelease(releasePath, rel); err != nil {
			return err
		}
		fmt.Printf("\nwrote %s for model %s\n", releasePath, modelVersion)
		return nil
	}

	released, err := gate.LoadRelease(releasePath)
	if err != nil {
		return err
	}
	checks := gate.Compare(released.Metrics, current, tol)
	if err := writeReport(reportPath, released.ModelVersion, current, checks); err != nil {
		return err
	}
	printChecks(released.ModelVersion, checks)

	if !gate.AllPassed(checks) {
		var failed []string
		for _, c := range checks {
			if !c.Passed {
				failed = append(failed, c.Name)
			}
		}
		return errRegression{failed: failed}
	}
	fmt.Println("\nGATE PASSED")
	return nil
}

func readSamples(path string) ([]sample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	if len(rows) < 2 || rows[0][0] != "text" || rows[0][1] != "label" {
		return nil, fmt.Errorf("%s: expected a header row text,label and at least one sample", path)
	}
	samples := make([]sample, 0, len(rows)-1)
	for _, r := range rows[1:] {
		samples = append(samples, sample{Text: r[0], Label: r[1]})
	}
	return samples, nil
}

func waitHealthy(client *http.Client, base string, maxWait time.Duration) error {
	deadline := time.Now().Add(maxWait)
	for {
		resp, err := client.Get(base + "/healthz")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("gateway at %s not healthy after %s", base, maxWait)
		}
		time.Sleep(time.Second)
	}
}

// screenAll posts every sample through a fixed pool of workers and returns the predicted
// labels in input order. The first failure cancels the remaining work.
func screenAll(client *http.Client, base string, samples []sample, workers int) ([]string, error) {
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)

	preds := make([]string, len(samples))
	jobs := make(chan int)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				label, err := screenOne(ctx, client, base, samples[i].Text)
				if err != nil {
					cancel(fmt.Errorf("sample %d: %w", i, err))
					return
				}
				preds[i] = label
			}
		}()
	}

feed:
	for i := range samples {
		select {
		case jobs <- i:
		case <-ctx.Done():
			break feed
		}
	}
	close(jobs)
	wg.Wait()

	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	return preds, nil
}

func screenOne(ctx context.Context, client *http.Client, base, text string) (string, error) {
	body, _ := json.Marshal(map[string]string{"text": text})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/v1/screen", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("status %d: %s", resp.StatusCode, bytes.TrimSpace(msg))
	}
	var out struct {
		Label string `json:"label"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	return out.Label, nil
}

func writeReport(path, releasedVersion string, current evalmetrics.Result, checks []gate.Check) error {
	b, err := json.MarshalIndent(map[string]any{
		"compared_with": releasedVersion,
		"passed":        gate.AllPassed(checks),
		"checks":        checks,
		"metrics":       current,
	}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func printPerClass(r evalmetrics.Result) {
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "class\tprecision\trecall\tf1\tsupport")
	for _, l := range evalmetrics.Labels {
		m := r.PerClass[l]
		fmt.Fprintf(tw, "%s\t%.4f\t%.4f\t%.4f\t%d\n", l, m.Precision, m.Recall, m.F1, m.Support)
	}
	tw.Flush()
	fmt.Printf("\nmacro F1 %.4f   FPR %.4f\n", r.MacroF1, r.FPR)
}

func printChecks(releasedVersion string, checks []gate.Check) {
	fmt.Printf("\ncompared with release %s:\n", releasedVersion)
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "metric\treleased\tcurrent\tlimit\tresult")
	for _, c := range checks {
		result := "ok"
		if !c.Passed {
			result = "FAIL"
		}
		fmt.Fprintf(tw, "%s\t%.4f\t%.4f\t%.4f\t%s\n", c.Name, c.Released, c.Current, c.Limit, result)
	}
	tw.Flush()
}
