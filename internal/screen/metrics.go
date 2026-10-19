package screen

import "github.com/prometheus/client_golang/prometheus"

// Metrics are the gateway's Prometheus series.
//
// The smishing-score histogram is a cheap drift signal: if the distribution of scores on
// live traffic moves away from what the test set produced, the model is seeing something
// new, even before anyone has labels for it.
type Metrics struct {
	Requests      *prometheus.CounterVec
	Errors        *prometheus.CounterVec
	Latency       prometheus.Histogram
	SmishingScore prometheus.Histogram
}

// NewMetrics registers the series on reg.
func NewMetrics(reg prometheus.Registerer) *Metrics {
	m := &Metrics{
		Requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "screen_requests_total",
			Help: "Screened messages by verdict.",
		}, []string{"verdict"}),
		Errors: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "screen_errors_total",
			Help: "Failed screening requests by reason.",
		}, []string{"reason"}),
		Latency: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "screen_latency_seconds",
			Help:    "End-to-end latency of a screening request, model call included.",
			Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5},
		}),
		SmishingScore: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "model_smishing_score",
			Help:    "Distribution of the model's smishing probability on screened messages.",
			Buckets: prometheus.LinearBuckets(0.1, 0.1, 9),
		}),
	}
	reg.MustRegister(m.Requests, m.Errors, m.Latency, m.SmishingScore)
	return m
}
