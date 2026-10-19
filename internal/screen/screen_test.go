package screen

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

type fakePredictor struct {
	pred Prediction
	err  error
	got  string
}

func (f *fakePredictor) Predict(_ context.Context, text string) (Prediction, error) {
	f.got = text
	return f.pred, f.err
}

func newHandler(p Predictor) *Handler {
	return &Handler{
		Model:   p,
		Metrics: NewMetrics(prometheus.NewRegistry()),
		Log:     slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
}

func post(h http.Handler, body string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/screen", strings.NewReader(body)))
	return rec
}

func TestVerdictFor(t *testing.T) {
	cases := map[string]Verdict{"ham": Allow, "spam": Flag, "smishing": Block}
	for label, want := range cases {
		got, err := VerdictFor(label)
		if err != nil || got != want {
			t.Errorf("VerdictFor(%q) = %q, %v; want %q", label, got, err, want)
		}
	}
	if _, err := VerdictFor("phish"); err == nil {
		t.Error("expected error for unknown label")
	}
}

func TestScreenBlocksSmishing(t *testing.T) {
	fake := &fakePredictor{pred: Prediction{
		Label:  "smishing",
		Scores: map[string]float64{"ham": 0.01, "spam": 0.02, "smishing": 0.97},
	}}
	h := newHandler(fake)

	rec := post(h, `{"text":"  Your parcel is held, pay at bit.do/x  "}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body %s", rec.Code, rec.Body)
	}
	var resp screenResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Verdict != Block || resp.Label != "smishing" {
		t.Errorf("got %+v", resp)
	}
	if fake.got != "Your parcel is held, pay at bit.do/x" {
		t.Errorf("text not trimmed before the model call: %q", fake.got)
	}
	if v := testutil.ToFloat64(h.Metrics.Requests.WithLabelValues("block")); v != 1 {
		t.Errorf("screen_requests_total{verdict=block} = %v, want 1", v)
	}
}

func TestScreenRejectsBadInput(t *testing.T) {
	h := newHandler(&fakePredictor{})
	for _, body := range []string{`not json`, `{"text":""}`, `{"text":"   "}`} {
		if rec := post(h, body); rec.Code != http.StatusBadRequest {
			t.Errorf("body %q: status = %d, want 400", body, rec.Code)
		}
	}
	long := `{"text":"` + strings.Repeat("a", maxTextRunes+1) + `"}`
	if rec := post(h, long); rec.Code != http.StatusBadRequest {
		t.Errorf("over-long text: status = %d, want 400", rec.Code)
	}
	if v := testutil.ToFloat64(h.Metrics.Errors.WithLabelValues("bad_request")); v != 4 {
		t.Errorf("screen_errors_total{reason=bad_request} = %v, want 4", v)
	}
}

func TestScreenReturns502WhenModelFails(t *testing.T) {
	h := newHandler(&fakePredictor{err: errors.New("connection refused")})
	if rec := post(h, `{"text":"hello"}`); rec.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", rec.Code)
	}
	h = newHandler(&fakePredictor{pred: Prediction{Label: "unknown"}})
	if rec := post(h, `{"text":"hello"}`); rec.Code != http.StatusBadGateway {
		t.Errorf("unknown label: status = %d, want 502", rec.Code)
	}
}

func TestModelClient(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]string
		_ = json.NewDecoder(r.Body).Decode(&body)
		if r.URL.Path != "/predict" || body["text"] != "hi" {
			http.Error(w, "unexpected request", http.StatusBadRequest)
			return
		}
		_, _ = w.Write([]byte(`{"label":"ham","scores":{"ham":0.9,"spam":0.05,"smishing":0.05}}`))
	}))
	defer srv.Close()

	pred, err := NewModelClient(srv.URL, time.Second).Predict(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if pred.Label != "ham" || pred.Scores["ham"] != 0.9 {
		t.Errorf("got %+v", pred)
	}
}

func TestModelClientErrorsOnNon200AndTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not loaded", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	if _, err := NewModelClient(srv.URL, time.Second).Predict(context.Background(), "x"); err == nil {
		t.Error("expected error on 503")
	}

	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
	}))
	defer slow.Close()
	if _, err := NewModelClient(slow.URL, 50*time.Millisecond).Predict(context.Background(), "x"); err == nil {
		t.Error("expected timeout error")
	}
}
