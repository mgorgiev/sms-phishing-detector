// Command gateway is the message-screening service.
//
//	POST /v1/screen  {"text": "..."} -> {"verdict": "allow|flag|block", "label": ..., "scores": ...}
//	GET  /healthz    liveness
//	GET  /metrics    Prometheus
//
// Configuration comes from the environment: ADDR (default :8080), MODEL_URL (default
// http://localhost:8000) and MODEL_TIMEOUT (default 2s).
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/mgorgiev/sms-phishing-detector/internal/screen"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	addr := env("ADDR", ":8080")
	modelURL := env("MODEL_URL", "http://localhost:8000")
	timeout, err := time.ParseDuration(env("MODEL_TIMEOUT", "2s"))
	if err != nil {
		log.Error("invalid MODEL_TIMEOUT", "err", err)
		os.Exit(1)
	}

	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))

	mux := http.NewServeMux()
	mux.Handle("POST /v1/screen", &screen.Handler{
		Model:   screen.NewModelClient(modelURL, timeout),
		Metrics: screen.NewMetrics(reg),
		Log:     log,
	})
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok\n"))
	})
	mux.Handle("GET /metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))

	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      timeout + 5*time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Info("gateway listening", "addr", addr, "model_url", modelURL)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("server failed", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Error("shutdown", "err", err)
	}
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
