package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestReadSamplesHandlesQuotedFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.csv")
	content := "text,label\n" +
		"\"Hi, are we still on?\",ham\n" +
		"\"line one\nline two\",spam\n" +
		"plain text,smishing\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	samples, err := readSamples(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 3 {
		t.Fatalf("got %d samples, want 3", len(samples))
	}
	if samples[0].Text != "Hi, are we still on?" || samples[1].Text != "line one\nline two" {
		t.Errorf("quoted fields not parsed: %+v", samples)
	}
}

func TestReadSamplesRejectsWrongHeader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.csv")
	if err := os.WriteFile(path, []byte("message,class\nhi,ham\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := readSamples(path); err == nil {
		t.Error("expected error for wrong header")
	}
}

// fakeGateway labels a message by a keyword in it, and fails on "boom".
func fakeGateway() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]string
		_ = json.NewDecoder(r.Body).Decode(&body)
		text := body["text"]
		switch {
		case strings.Contains(text, "boom"):
			http.Error(w, "model service unavailable", http.StatusBadGateway)
		case strings.Contains(text, "prize"):
			_, _ = w.Write([]byte(`{"verdict":"flag","label":"spam"}`))
		case strings.Contains(text, "verify"):
			_, _ = w.Write([]byte(`{"verdict":"block","label":"smishing"}`))
		default:
			_, _ = w.Write([]byte(`{"verdict":"allow","label":"ham"}`))
		}
	}))
}

func TestScreenAllKeepsInputOrder(t *testing.T) {
	srv := fakeGateway()
	defer srv.Close()

	var samples []sample
	var want []string
	for i := 0; i < 50; i++ {
		switch i % 3 {
		case 0:
			samples, want = append(samples, sample{Text: "see you soon"}), append(want, "ham")
		case 1:
			samples, want = append(samples, sample{Text: "you won a prize"}), append(want, "spam")
		case 2:
			samples, want = append(samples, sample{Text: "verify your account"}), append(want, "smishing")
		}
	}

	got, err := screenAll(&http.Client{Timeout: time.Second}, srv.URL, samples, 8)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("prediction %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestScreenAllFailsOnGatewayError(t *testing.T) {
	srv := fakeGateway()
	defer srv.Close()

	samples := []sample{{Text: "hello"}, {Text: "boom"}, {Text: "hello again"}}
	if _, err := screenAll(&http.Client{Timeout: time.Second}, srv.URL, samples, 2); err == nil {
		t.Error("expected an error when the gateway fails")
	}
}
