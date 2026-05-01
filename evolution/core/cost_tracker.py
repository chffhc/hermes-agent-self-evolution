"""API cost tracking for self-evolution runs.

Tracks token usage and estimated cost for all LLM calls during optimization.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime

# Approximate per-1M-token costs for common models (USD)
# Prices as of mid-2026; adjust as needed.
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M)
    "qwen3.6-plus": (0.4, 1.2),
    "qwen-plus": (0.4, 1.2),
    "qwen-max": (2.0, 6.0),
    "qwen-turbo": (0.3, 0.6),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
}


@dataclass
class CallRecord:
    """A single LLM API call."""
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: str
    purpose: str = ""  # e.g. "judge", "reflection", "dataset_gen"


@dataclass
class CostSummary:
    """Aggregated cost summary."""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    per_model: dict[str, dict] = field(default_factory=dict)
    per_purpose: dict[str, dict] = field(default_factory=dict)


class APICostTracker:
    """Thread-safe LLM API cost tracker.

    Wraps DSPy's LM to intercept usage metadata from responses.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._calls: list[CallRecord] = []

    def record(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        purpose: str = "",
    ):
        """Record an API call."""
        record = CallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=datetime.now().isoformat(),
            purpose=purpose,
        )
        with self._lock:
            self._calls.append(record)

    @property
    def calls(self) -> list[CallRecord]:
        with self._lock:
            return list(self._calls)

    def summary(self) -> CostSummary:
        """Compute aggregated cost summary."""
        s = CostSummary()
        for c in self._calls:
            s.total_calls += 1
            s.total_input_tokens += c.input_tokens
            s.total_output_tokens += c.output_tokens
            cost = _estimate_cost(c.model, c.input_tokens, c.output_tokens)
            s.total_cost_usd += cost

            # Per-model
            pm = s.per_model.setdefault(c.model, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            })
            pm["calls"] += 1
            pm["input_tokens"] += c.input_tokens
            pm["output_tokens"] += c.output_tokens
            pm["cost_usd"] += cost

            # Per-purpose
            pp = s.per_purpose.setdefault(c.purpose or "unknown", {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            })
            pp["calls"] += 1
            pp["input_tokens"] += c.input_tokens
            pp["output_tokens"] += c.output_tokens
            pp["cost_usd"] += cost

        return s

    def print_summary(self, console=None):
        """Print cost summary to console."""
        s = self.summary()
        lines = []
        lines.append(f"\n{'='*50}")
        lines.append("  API Cost Summary")
        lines.append(f"{'='*50}")
        lines.append(f"  Total calls: {s.total_calls}")
        lines.append(f"  Input tokens:  {s.total_input_tokens:>10,}")
        lines.append(f"  Output tokens: {s.total_output_tokens:>10,}")
        lines.append(f"  Est. cost:     ${s.total_cost_usd:>10.4f}")
        lines.append("")

        if s.per_model:
            lines.append("  Per model:")
            for model, m in sorted(s.per_model.items()):
                lines.append(
                    f"    {model}: {m['calls']} calls, "
                    f"{m['input_tokens'] + m['output_tokens']:,} tokens, "
                    f"${m['cost_usd']:.4f}"
                )

        if s.per_purpose:
            lines.append("")
            lines.append("  Per purpose:")
            for purpose, p in sorted(s.per_purpose.items()):
                lines.append(
                    f"    {purpose}: {p['calls']} calls, "
                    f"{p['input_tokens'] + p['output_tokens']:,} tokens, "
                    f"${p['cost_usd']:.4f}"
                )

        lines.append(f"{'='*50}\n")
        text = "\n".join(lines)

        if console:
            console.print(text)
        else:
            print(text)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for a single API call."""
    # Normalize model name for lookup
    key = model
    if "/" in key:
        key = key.split("/")[-1]
    if key.startswith("openai/"):
        key = key[len("openai/"):]

    prices = _MODEL_PRICES.get(key)
    if prices is None:
        return 0.0  # Unknown model, can't estimate

    input_cost, output_cost = prices
    return (
        input_tokens / 1_000_000 * input_cost
        + output_tokens / 1_000_000 * output_cost
    )


# Global singleton for easy import
tracker = APICostTracker()
