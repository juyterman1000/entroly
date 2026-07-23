# Grafana dashboard for the Entroly proxy

An importable dashboard over the proxy's built-in Prometheus endpoint.
No extra services: the proxy already serves `GET /metrics` in Prometheus
text format.

![panels] tokens saved · savings ratio · optimized share · circuit breaker ·
request/token rates · pipeline latency · recorded outcomes

## 1. Scrape the proxy

```yaml
# prometheus.yml
scrape_configs:
  - job_name: entroly-proxy
    metrics_path: /metrics
    static_configs:
      - targets: ["localhost:9377"]   # entroly proxy default port
```

## 2. Import the dashboard

Grafana → Dashboards → New → Import → upload
[`entroly-proxy-dashboard.json`](entroly-proxy-dashboard.json), then select
your Prometheus datasource when prompted.

## Exported metrics

| Metric | Type | Meaning |
|---|---|---|
| `entroly_requests_total` | counter | all proxy requests |
| `entroly_requests_optimized` | counter | requests entroly optimized |
| `entroly_requests_bypassed` | counter | requests passed through untouched |
| `entroly_requests_subscription_blocked` | counter | OAuth-bearer requests blocked from the public API |
| `entroly_tokens_original_total` | counter | input tokens before optimization |
| `entroly_tokens_optimized_total` | counter | input tokens actually sent |
| `entroly_pipeline_latency_ms` | gauge | mean optimization latency |
| `entroly_circuit_breaker` | gauge | 0 = closed (healthy), 1 = open |
| `entroly_outcome_success` / `entroly_outcome_failure` | counter | recorded downstream outcomes |

Honest-measurement note: token counters measure **input-side reduction at the
proxy**. Provider invoices remain the billing source of truth — see
[docs/limitations.md](../limitations.md).
