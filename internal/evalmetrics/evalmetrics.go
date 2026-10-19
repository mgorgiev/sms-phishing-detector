// Package evalmetrics computes the classification metrics used by the regression gate.
//
// The definitions match ml/src/smishing/metrics.py, so the numbers the gate reports can be
// compared with the notebook. FPR is the share of legitimate (ham) messages that the model
// labels as spam or smishing.
package evalmetrics

import "fmt"

// Labels is the fixed class order. It must match the model's label order.
var Labels = []string{"ham", "spam", "smishing"}

// ClassMetrics holds per-class precision, recall and F1.
type ClassMetrics struct {
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	Support   int     `json:"support"`
}

// Result is the full evaluation of one model on one dataset.
type Result struct {
	MacroF1   float64                 `json:"macro_f1"`
	FPR       float64                 `json:"fpr"`
	PerClass  map[string]ClassMetrics `json:"per_class"`
	Confusion [][]int                 `json:"confusion"`
}

// Evaluate compares true and predicted labels.
func Evaluate(yTrue, yPred []string) (Result, error) {
	if len(yTrue) != len(yPred) {
		return Result{}, fmt.Errorf("length mismatch: %d true labels, %d predictions", len(yTrue), len(yPred))
	}
	index := make(map[string]int, len(Labels))
	for i, l := range Labels {
		index[l] = i
	}

	n := len(Labels)
	confusion := make([][]int, n)
	for i := range confusion {
		confusion[i] = make([]int, n)
	}
	for i := range yTrue {
		t, ok := index[yTrue[i]]
		if !ok {
			return Result{}, fmt.Errorf("unknown true label %q at row %d", yTrue[i], i)
		}
		p, ok := index[yPred[i]]
		if !ok {
			return Result{}, fmt.Errorf("unknown predicted label %q at row %d", yPred[i], i)
		}
		confusion[t][p]++
	}

	res := Result{PerClass: make(map[string]ClassMetrics, n), Confusion: confusion}
	var f1Sum float64
	for c, label := range Labels {
		tp := confusion[c][c]
		var predicted, actual int
		for k := 0; k < n; k++ {
			predicted += confusion[k][c]
			actual += confusion[c][k]
		}
		m := ClassMetrics{
			Precision: ratio(tp, predicted),
			Recall:    ratio(tp, actual),
			Support:   actual,
		}
		if m.Precision+m.Recall > 0 {
			m.F1 = 2 * m.Precision * m.Recall / (m.Precision + m.Recall)
		}
		res.PerClass[label] = m
		f1Sum += m.F1
	}
	res.MacroF1 = f1Sum / float64(n)

	ham := index["ham"]
	hamTotal := 0
	for k := 0; k < n; k++ {
		hamTotal += confusion[ham][k]
	}
	res.FPR = ratio(hamTotal-confusion[ham][ham], hamTotal)
	return res, nil
}

func ratio(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}
