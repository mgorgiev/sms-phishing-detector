package screen

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// Verdict is what the messaging system should do with a message.
type Verdict string

const (
	Allow Verdict = "allow"
	Flag  Verdict = "flag"
	Block Verdict = "block"
)

// VerdictFor maps a predicted class to a verdict.
func VerdictFor(label string) (Verdict, error) {
	switch label {
	case "ham":
		return Allow, nil
	case "spam":
		return Flag, nil
	case "smishing":
		return Block, nil
	}
	return "", fmt.Errorf("unknown label %q", label)
}

const (
	maxBodyBytes = 16 << 10
	maxTextRunes = 5000
)

type screenRequest struct {
	Text string `json:"text"`
}

type screenResponse struct {
	Verdict Verdict            `json:"verdict"`
	Label   string             `json:"label"`
	Scores  map[string]float64 `json:"scores"`
}

// Handler serves POST /v1/screen.
type Handler struct {
	Model   Predictor
	Metrics *Metrics
	Log     *slog.Logger
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	defer func() { h.Metrics.Latency.Observe(time.Since(start).Seconds()) }()

	var req screenRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	if err := dec.Decode(&req); err != nil {
		h.fail(w, http.StatusBadRequest, "bad_request", "invalid JSON body")
		return
	}
	text := strings.TrimSpace(req.Text)
	if text == "" {
		h.fail(w, http.StatusBadRequest, "bad_request", "text must not be empty")
		return
	}
	if len([]rune(text)) > maxTextRunes {
		h.fail(w, http.StatusBadRequest, "bad_request", fmt.Sprintf("text longer than %d characters", maxTextRunes))
		return
	}

	pred, err := h.Model.Predict(r.Context(), text)
	if err != nil {
		h.Log.Error("model call failed", "err", err)
		h.fail(w, http.StatusBadGateway, "model", "model service unavailable")
		return
	}
	verdict, err := VerdictFor(pred.Label)
	if err != nil {
		h.Log.Error("model returned an unknown label", "label", pred.Label)
		h.fail(w, http.StatusBadGateway, "model", "model returned an unknown label")
		return
	}

	h.Metrics.Requests.WithLabelValues(string(verdict)).Inc()
	h.Metrics.SmishingScore.Observe(pred.Scores["smishing"])
	writeJSON(w, http.StatusOK, screenResponse{Verdict: verdict, Label: pred.Label, Scores: pred.Scores})
}

func (h *Handler) fail(w http.ResponseWriter, status int, reason, msg string) {
	h.Metrics.Errors.WithLabelValues(reason).Inc()
	writeJSON(w, status, map[string]string{"error": msg})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v) // the client may have gone; nothing useful to do then
}
