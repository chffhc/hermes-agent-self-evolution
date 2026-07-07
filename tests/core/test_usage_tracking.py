"""Regression tests for LM usage tracking."""

from __future__ import annotations


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
