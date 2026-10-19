// Package screen is the message-screening API: it sends a message to the model service,
// turns the predicted class into a verdict and records Prometheus metrics.
package screen

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Prediction is the model service's answer for one message.
type Prediction struct {
	Label  string             `json:"label"`
	Scores map[string]float64 `json:"scores"`
}

// Predictor classifies one message.
type Predictor interface {
	Predict(ctx context.Context, text string) (Prediction, error)
}

// ModelClient calls the Python model service over HTTP.
type ModelClient struct {
	baseURL string
	http    *http.Client
}

// NewModelClient returns a client with a per-request timeout.
func NewModelClient(baseURL string, timeout time.Duration) *ModelClient {
	return &ModelClient{baseURL: baseURL, http: &http.Client{Timeout: timeout}}
}

// Predict posts the text to /predict.
func (c *ModelClient) Predict(ctx context.Context, text string) (Prediction, error) {
	var p Prediction
	body, err := json.Marshal(map[string]string{"text": text})
	if err != nil {
		return p, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/predict", bytes.NewReader(body))
	if err != nil {
		return p, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return p, fmt.Errorf("model service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return p, fmt.Errorf("model service: status %d: %s", resp.StatusCode, bytes.TrimSpace(msg))
	}
	if err := json.NewDecoder(resp.Body).Decode(&p); err != nil {
		return p, fmt.Errorf("model service: decode response: %w", err)
	}
	return p, nil
}
