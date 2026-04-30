"""
Prometheus AI Observability Metrics.

Import this module once at startup (e.g. in main.py or ai_service.py) so
metrics are registered before the first scrape.

Usage in ai_service.py::

    from app.core.ai_metrics import (
        record_ai_cache_hit, record_ai_cache_miss,
        record_ai_latency, record_ai_tokens, record_ai_cost,
    )
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Cache hit/miss counters ─────────────────────────────────────────
ai_cache_hits = Counter(
    "ai_cache_hits_total",
    "Total AI cache hits",
    ["endpoint"],
)

ai_cache_misses = Counter(
    "ai_cache_misses_total",
    "Total AI cache misses",
    ["endpoint"],
)

# ── Latency histogram ───────────────────────────────────────────────
ai_latency = Histogram(
    "ai_request_duration_seconds",
    "AI request latency in seconds",
    ["endpoint", "model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 12.0, float("inf")),
)

# ── Token consumption counters ──────────────────────────────────────
ai_tokens_consumed = Counter(
    "ai_tokens_total",
    "Total tokens consumed (prompt + completion)",
    ["direction", "model"],  # direction: "in" | "out"
)

# ── Running cost estimate ───────────────────────────────────────────
# Gemini 2.0 Flash pricing (as of 2025): $0.075/1M input, $0.30/1M output
ai_cost_estimate = Gauge(
    "ai_estimated_cost_usd",
    "Estimated Gemini API cost in USD (running total, approximate)",
)

# Internal state for cost accumulation (Gauge doesn't accumulate)
_cumulative_cost: float = 0.0

# Gemini Flash cost per token (USD)
_COST_PER_INPUT_TOKEN = 0.075 / 1_000_000
_COST_PER_OUTPUT_TOKEN = 0.300 / 1_000_000


# ── Helper functions ────────────────────────────────────────────────

def record_ai_cache_hit(endpoint: str = "portfolio_analysis") -> None:
    ai_cache_hits.labels(endpoint=endpoint).inc()


def record_ai_cache_miss(endpoint: str = "portfolio_analysis") -> None:
    ai_cache_misses.labels(endpoint=endpoint).inc()


def record_ai_latency(
    latency_seconds: float,
    endpoint: str = "portfolio_analysis",
    model: str = "gemini-2.0-flash",
) -> None:
    ai_latency.labels(endpoint=endpoint, model=model).observe(latency_seconds)


def record_ai_tokens(
    tokens_in: int,
    tokens_out: int,
    model: str = "gemini-2.0-flash",
) -> None:
    global _cumulative_cost
    ai_tokens_consumed.labels(direction="in", model=model).inc(tokens_in)
    ai_tokens_consumed.labels(direction="out", model=model).inc(tokens_out)
    _cumulative_cost += (
        tokens_in * _COST_PER_INPUT_TOKEN + tokens_out * _COST_PER_OUTPUT_TOKEN
    )
    ai_cost_estimate.set(_cumulative_cost)


def record_ai_cost(extra_usd: float) -> None:
    """Manually add to cost estimate (e.g. for PDF extraction calls)."""
    global _cumulative_cost
    _cumulative_cost += extra_usd
    ai_cost_estimate.set(_cumulative_cost)
