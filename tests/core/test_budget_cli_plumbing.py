"""Regression tests for --max-cost-usd CLI/config budget plumbing.

The hard budget lives on the global cost tracker. These tests verify each
entrypoint threads the option through (or applies it directly) without
breaking the EVOLUTION_MAX_COST_USD env default when the flag is omitted.
"""

from __future__ import annotations

import inspect

import pytest
from click.testing import CliRunner


@pytest.fixture
def global_budget_guard():
    """Snapshot and restore the global tracker's budget around a test."""
    from evolution.core.cost_tracker import tracker

    before = tracker._max_cost_usd
    yield tracker
    tracker.set_budget(before)


def test_run_evolution_cli_threads_max_cost_usd(monkeypatch):
    import run_evolution

    captured = {}
    monkeypatch.setattr(run_evolution, "evolve", lambda **kw: captured.update(kw))

    result = CliRunner().invoke(run_evolution.main, ["--skill", "arxiv", "--max-cost-usd", "5.0"])

    assert result.exit_code == 0, result.output
    assert captured["max_cost_usd"] == 5.0


def test_run_evolution_cli_defaults_budget_to_none(monkeypatch):
    import run_evolution

    captured = {}
    monkeypatch.setattr(run_evolution, "evolve", lambda **kw: captured.update(kw))

    result = CliRunner().invoke(run_evolution.main, ["--skill", "arxiv"])

    assert result.exit_code == 0, result.output
    # None means "not specified" — evolve() then leaves the env default alone.
    assert captured["max_cost_usd"] is None


def test_run_evolution_cli_rejects_nonpositive_budget(monkeypatch):
    import run_evolution

    monkeypatch.setattr(run_evolution, "evolve", lambda **kw: None)

    result = CliRunner().invoke(run_evolution.main, ["--skill", "arxiv", "--max-cost-usd", "0"])

    assert result.exit_code != 0


def test_evolve_skill_cli_threads_max_cost_usd(monkeypatch):
    from evolution.skills import evolve_skill

    captured = {}
    monkeypatch.setattr(evolve_skill, "evolve", lambda **kw: captured.update(kw))

    result = CliRunner().invoke(evolve_skill.main, ["--skill", "arxiv", "--max-cost-usd", "2.5"])

    assert result.exit_code == 0, result.output
    assert captured["max_cost_usd"] == 2.5


def test_evolve_function_applies_budget_before_billable_work():
    from evolution.skills.evolve_skill import evolve

    source = inspect.getsource(evolve)
    assert "set_budget_from_option(config.max_cost_usd)" in source
    # Must precede the optimizer (and everything else billable, e.g. dataset
    # generation) so the very first LLM call is already budget-enforced.
    assert source.index("set_budget_from_option") < source.index("dspy.GEPA(")


def test_phase2_cli_sets_budget_on_global_tracker(monkeypatch, global_budget_guard):
    from evolution.tools import evolve_tool_descriptions as mod

    monkeypatch.setattr(mod, "evolve_tool_descriptions", lambda **kw: None)

    result = CliRunner().invoke(mod.main, ["--max-cost-usd", "3.0"])

    assert result.exit_code == 0, result.output
    assert global_budget_guard._max_cost_usd == 3.0


def test_phase3_cli_sets_budget_on_global_tracker(monkeypatch, global_budget_guard):
    from evolution.prompts import evolve_prompt_section as mod

    monkeypatch.setattr(mod, "evolve_prompt_section", lambda **kw: None)

    result = CliRunner().invoke(mod.main, ["--max-cost-usd", "3.5"])

    assert result.exit_code == 0, result.output
    assert global_budget_guard._max_cost_usd == 3.5


def test_session_importer_cli_sets_budget(monkeypatch, global_budget_guard):
    from evolution.core import external_importers as mod

    # --dry-run makes no LLM calls, but the budget must already be applied
    # before main gets anywhere near billable relevance scoring.
    monkeypatch.setattr(mod, "_load_skill_text", lambda name: ("arxiv", "skill text"))
    for importer in (mod.ClaudeCodeImporter, mod.CopilotImporter, mod.HermesSessionImporter):
        monkeypatch.setattr(importer, "extract_messages", staticmethod(lambda limit=0: []))

    result = CliRunner().invoke(
        mod.main, ["--skill", "arxiv", "--dry-run", "--max-cost-usd", "1.5"]
    )

    assert result.exit_code == 0, result.output
    assert global_budget_guard._max_cost_usd == 1.5


def test_session_importer_cli_rejects_nonpositive_budget(global_budget_guard):
    from evolution.core import external_importers as mod

    before = global_budget_guard._max_cost_usd

    result = CliRunner().invoke(mod.main, ["--skill", "arxiv", "--dry-run", "--max-cost-usd", "0"])

    assert result.exit_code != 0
    assert global_budget_guard._max_cost_usd == before


def test_continuous_evolution_cli_sets_budget(monkeypatch, global_budget_guard, capsys):
    from evolution.monitor import continuous_evolution as mod

    # No --cycle/--setup-cron: main applies the budget then prints help.
    monkeypatch.setattr("sys.argv", ["continuous_evolution", "--max-cost-usd", "7.0"])
    mod.main()
    capsys.readouterr()

    assert global_budget_guard._max_cost_usd == 7.0
