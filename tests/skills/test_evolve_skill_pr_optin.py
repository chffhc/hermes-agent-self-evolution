"""Regression tests for the opt-in Phase 1 PR step.

The invariants: no PR-related git operation can ever happen by default, the
opt-in path goes through PRBuilder's safety semantics, gate-failing or
non-improving runs are refused, and --pr-dry-run produces a redacted preview
with zero git/GitHub side effects.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from click.testing import CliRunner

from evolution.core.pr_builder import PRResult
from evolution.skills import evolve_skill


def _run_metrics(**overrides) -> dict:
    metrics = {
        "skill_name": "arxiv",
        "deployable": True,
        "improvement": 0.05,
        "baseline_score": 0.40,
        "evolved_score": 0.45,
        "iterations": 10,
        "optimizer_model": "qwen3.6-plus",
        "train_examples": 6,
        "val_examples": 2,
        "holdout_examples": 2,
        "elapsed_seconds": 120.0,
    }
    metrics.update(overrides)
    return metrics


def _forbid_subprocess(monkeypatch):
    """Any subprocess call from pr_builder means a git/gh side effect leaked."""
    import evolution.core.pr_builder as pr_builder_mod

    def _boom(*args, **kwargs):
        raise AssertionError(f"unexpected subprocess call: {args} {kwargs}")

    monkeypatch.setattr(pr_builder_mod.subprocess, "run", _boom)


class FakePRBuilder:
    """Captures create_pr inputs and returns a canned PRResult."""

    instances: list[FakePRBuilder] = []
    result = PRResult(
        success=True,
        branch_name="evolve/arxiv-x",
        pr_url="https://github.com/x/pull/1",
        branch_pushed=True,
        pr_created=True,
    )

    def __init__(self, hermes_agent_path: Path, target_repo: str = "NousResearch/hermes-agent"):
        self.hermes_agent_path = hermes_agent_path
        self.create_pr_calls: list[tuple] = []
        FakePRBuilder.instances.append(self)

    def create_pr(self, changes, metrics, title_prefix="evolve"):
        self.create_pr_calls.append((changes, metrics, title_prefix))
        return FakePRBuilder.result


@pytest.fixture(autouse=True)
def _reset_fake_builder():
    FakePRBuilder.instances = []
    FakePRBuilder.result = PRResult(
        success=True,
        branch_name="evolve/arxiv-x",
        pr_url="https://github.com/x/pull/1",
        branch_pushed=True,
        pr_created=True,
    )
    yield
    FakePRBuilder.instances = []


def test_evolve_defaults_never_request_pr():
    sig = inspect.signature(evolve_skill.evolve)
    assert sig.parameters["create_pr"].default is False
    assert sig.parameters["pr_dry_run"].default is False


def test_cli_defaults_pr_flags_off(monkeypatch):
    captured = {}
    monkeypatch.setattr(evolve_skill, "evolve", lambda **kw: captured.update(kw))

    result = CliRunner().invoke(evolve_skill.main, ["--skill", "arxiv"])

    assert result.exit_code == 0, result.output
    assert captured["create_pr"] is False
    assert captured["pr_dry_run"] is False


def test_cli_threads_pr_flags(monkeypatch):
    captured = {}
    monkeypatch.setattr(evolve_skill, "evolve", lambda **kw: captured.update(kw))

    result = CliRunner().invoke(
        evolve_skill.main, ["--skill", "arxiv", "--create-pr", "--pr-dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert captured["create_pr"] is True
    assert captured["pr_dry_run"] is True


def test_handle_pr_refuses_non_deployable_run(monkeypatch, tmp_path):
    import evolution.core.pr_builder as pr_builder_mod

    _forbid_subprocess(monkeypatch)
    monkeypatch.setattr(pr_builder_mod, "PRBuilder", FakePRBuilder)

    info = evolve_skill._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=tmp_path,
        skill_relpath="skills/arxiv/SKILL.md",
        baseline_text="old",
        evolved_text="new",
        run_metrics=_run_metrics(deployable=False),
    )

    assert info["created"] is False
    assert info["branch_pushed"] is False
    assert "not deployable" in info["skipped_reason"]
    assert FakePRBuilder.instances == []


def test_handle_pr_refuses_non_improving_run(monkeypatch, tmp_path):
    import evolution.core.pr_builder as pr_builder_mod

    _forbid_subprocess(monkeypatch)
    monkeypatch.setattr(pr_builder_mod, "PRBuilder", FakePRBuilder)

    info = evolve_skill._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=tmp_path,
        skill_relpath="skills/arxiv/SKILL.md",
        baseline_text="old",
        evolved_text="new",
        run_metrics=_run_metrics(improvement=0.0),
    )

    assert info["created"] is False
    assert "no positive holdout proxy improvement" in info["skipped_reason"]
    assert FakePRBuilder.instances == []


def test_handle_pr_optin_calls_prbuilder_with_evolved_change(monkeypatch, tmp_path):
    import evolution.core.pr_builder as pr_builder_mod

    monkeypatch.setattr(pr_builder_mod, "PRBuilder", FakePRBuilder)

    info = evolve_skill._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=tmp_path,
        skill_relpath="skills/arxiv/SKILL.md",
        baseline_text="old skill text",
        evolved_text="new skill text",
        run_metrics=_run_metrics(),
    )

    assert len(FakePRBuilder.instances) == 1
    builder = FakePRBuilder.instances[0]
    assert builder.hermes_agent_path == tmp_path
    changes, pr_metrics, _prefix = builder.create_pr_calls[0]
    assert len(changes) == 1
    assert changes[0].file_path == "skills/arxiv/SKILL.md"
    assert changes[0].original_content == "old skill text"
    assert changes[0].evolved_content == "new skill text"
    assert pr_metrics.baseline_score == pytest.approx(0.40)
    assert pr_metrics.evolved_score == pytest.approx(0.45)
    assert pr_metrics.eval_dataset_size == 10

    assert info["created"] is True
    assert info["branch_pushed"] is True
    assert info["url"] == "https://github.com/x/pull/1"
    assert info["error"] is None


def test_handle_pr_branch_pushed_without_pr_is_not_created(monkeypatch, tmp_path):
    import evolution.core.pr_builder as pr_builder_mod

    monkeypatch.setattr(pr_builder_mod, "PRBuilder", FakePRBuilder)
    FakePRBuilder.result = PRResult(
        success=False,
        branch_name="evolve/arxiv-x",
        error="gh CLI not found",
        branch_pushed=True,
        pr_created=False,
    )

    info = evolve_skill._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=tmp_path,
        skill_relpath="skills/arxiv/SKILL.md",
        baseline_text="old",
        evolved_text="new",
        run_metrics=_run_metrics(),
    )

    assert info["created"] is False
    assert info["branch_pushed"] is True
    assert info["error"] == "gh CLI not found"


def test_pr_dry_run_redacts_preview_and_touches_no_git(monkeypatch, tmp_path):
    # Real PRBuilder, but any subprocess use fails the test: the dry run
    # must be pure string rendering.
    _forbid_subprocess(monkeypatch)

    secret = "sk-or-v1-abcdef1234567890abcdef"
    info = evolve_skill._handle_pr_request(
        create_pr=True,
        pr_dry_run=True,
        hermes_agent_path=tmp_path,
        skill_relpath="skills/arxiv/SKILL.md",
        baseline_text="old skill text",
        evolved_text=f"new skill text with leaked key {secret}",
        run_metrics=_run_metrics(),
    )

    assert info["dry_run"] is True
    assert info["created"] is False
    assert info["branch_pushed"] is False
    preview = info["preview"]
    assert secret not in preview
    assert "[REDACTED]" in preview
    assert "skills/arxiv/SKILL.md" in preview
    assert "requires human review" in preview
