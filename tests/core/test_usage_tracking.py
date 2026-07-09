"""Regression tests for LM usage tracking and budget enforcement."""

from __future__ import annotations

import pytest

from evolution.core.errors import BudgetExceededError


def test_usage_tracked_lm_records_calls(monkeypatch):
    from evolution.core import config as config_module
    from evolution.core.cost_tracker import APICostTracker

    tracker = APICostTracker()
    monkeypatch.setattr("evolution.core.cost_tracker.tracker", tracker)

    class FakeLM:
        def __init__(self):
            self.history = []
            self.custom_attr = "delegated"

        def __call__(self, prompt):
            self.history.append({"usage": {"prompt_tokens": 123, "completion_tokens": 45}})
            return f"ok: {prompt}"

    wrapped = config_module.install_usage_tracking(FakeLM(), "openai/qwen3.6-plus")

    assert wrapped("hello") == "ok: hello"
    assert wrapped.custom_attr == "delegated"
    summary = tracker.summary()
    assert summary.total_calls == 1
    assert summary.total_input_tokens == 123
    assert summary.total_output_tokens == 45
    assert summary.total_cost_usd > 0


def test_budget_exceeded_raises_and_still_records_the_call():
    from evolution.core.cost_tracker import APICostTracker

    tracker = APICostTracker(max_cost_usd=0.001)

    # qwen3.6-plus: 1M input tokens = $0.40 — well past a $0.001 budget.
    with pytest.raises(BudgetExceededError):
        tracker.record("qwen3.6-plus", 1_000_000, 0)

    # The over-budget call is still recorded (the spend already happened).
    assert tracker.summary().total_calls == 1
    assert tracker.total_cost_usd > 0.001


def test_no_budget_never_raises():
    from evolution.core.cost_tracker import APICostTracker

    tracker = APICostTracker()
    for _ in range(3):
        tracker.record("qwen3.6-plus", 10_000_000, 10_000_000)
    assert tracker.summary().total_calls == 3


def test_set_budget_applies_to_subsequent_calls():
    from evolution.core.cost_tracker import APICostTracker

    tracker = APICostTracker()
    tracker.record("qwen3.6-plus", 1_000_000, 0)
    tracker.set_budget(0.001)
    with pytest.raises(BudgetExceededError):
        tracker.record("qwen3.6-plus", 1, 0)


def test_unknown_model_charged_at_conservative_default():
    from evolution.core.cost_tracker import _estimate_cost

    # Unknown models must not be free — otherwise a mistyped model name
    # silently bypasses budget enforcement.
    cost = _estimate_cost("unknown-model-xyz", 1_000_000, 1_000_000)
    known_max = _estimate_cost("claude-opus-4-20250514", 1_000_000, 1_000_000)
    assert cost >= known_max > 0


def test_budget_from_env_parsing(monkeypatch):
    from evolution.core.cost_tracker import _budget_from_env

    monkeypatch.setenv("EVOLUTION_MAX_COST_USD", "12.5")
    assert _budget_from_env() == 12.5

    monkeypatch.setenv("EVOLUTION_MAX_COST_USD", "not-a-number")
    assert _budget_from_env() is None

    monkeypatch.setenv("EVOLUTION_MAX_COST_USD", "-1")
    assert _budget_from_env() is None

    monkeypatch.delenv("EVOLUTION_MAX_COST_USD")
    assert _budget_from_env() is None


def test_budget_error_propagates_through_usage_tracking_wrapper(monkeypatch):
    from evolution.core import config as config_module
    from evolution.core.cost_tracker import APICostTracker

    tracker = APICostTracker(max_cost_usd=0.000001)
    monkeypatch.setattr("evolution.core.cost_tracker.tracker", tracker)

    class FakeLM:
        def __init__(self):
            self.history = []

        def __call__(self, prompt):
            self.history.append({"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000}})
            return "ok"

    wrapped = config_module.install_usage_tracking(FakeLM(), "openai/qwen3.6-plus")

    # The wrapper must not swallow the budget error like ordinary
    # tracking failures — a hard budget has to fail closed.
    with pytest.raises(BudgetExceededError):
        wrapped("hello")
