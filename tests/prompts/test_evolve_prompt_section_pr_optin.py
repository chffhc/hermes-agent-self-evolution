"""Regression tests for the opt-in Phase 3 PR step.

Same invariants as Phases 1/2: no PR-related git operation by default, gates
refuse failed/non-improving runs, --pr-dry-run is pure redacted rendering,
and sections whose baseline content can't be located verbatim in the source
file refuse the PR instead of guessing.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from click.testing import CliRunner

from evolution.core.pr_builder import PRResult
from evolution.prompts import evolve_prompt_section as phase3
from evolution.prompts.evolve_prompt_section import PromptSection

BASELINE_CONTENT = "Use memory to persist important user facts across sessions."
EVOLVED_CONTENT = "Use memory to persist important user facts; review before saving."


def _section(content: str) -> PromptSection:
    return PromptSection(
        name="MEMORY_GUIDANCE",
        content=content,
        file_path="agent/prompt_builder.py",
        description="How and when to use persistent memory",
        max_growth_pct=20,
        risk_level="medium",
    )


def _run_metrics(**overrides) -> dict:
    metrics = {
        "deployable": True,
        "improvement": 0.05,
        "baseline_score": 0.50,
        "evolved_score": 0.55,
        "iterations": 10,
        "optimizer_model": "qwen3.6-plus",
        "sections": ["MEMORY_GUIDANCE"],
        "train_examples": 6,
        "val_examples": 2,
        "holdout_examples": 2,
        "elapsed_seconds": 100.0,
    }
    metrics.update(overrides)
    return metrics


def _forbid_subprocess(monkeypatch):
    import evolution.core.pr_builder as pr_builder_mod

    def _boom(*args, **kwargs):
        raise AssertionError(f"unexpected subprocess call: {args} {kwargs}")

    monkeypatch.setattr(pr_builder_mod.subprocess, "run", _boom)


def _hermes_repo(tmp_path: Path, body: str = BASELINE_CONTENT) -> Path:
    (tmp_path / "agent").mkdir(parents=True, exist_ok=True)
    (tmp_path / "agent" / "prompt_builder.py").write_text(
        f'MEMORY_GUIDANCE = """{body}"""\n', encoding="utf-8"
    )
    return tmp_path


def test_evolve_defaults_never_request_pr():
    sig = inspect.signature(phase3.evolve_prompt_section)
    assert sig.parameters["create_pr"].default is False
    assert sig.parameters["pr_dry_run"].default is False


def test_cli_defaults_pr_flags_off(monkeypatch):
    captured = {}
    monkeypatch.setattr(phase3, "evolve_prompt_section", lambda **kw: captured.update(kw))

    result = CliRunner().invoke(phase3.main, [])

    assert result.exit_code == 0, result.output
    assert captured["create_pr"] is False
    assert captured["pr_dry_run"] is False


def test_cli_threads_pr_flags(monkeypatch):
    captured = {}
    monkeypatch.setattr(phase3, "evolve_prompt_section", lambda **kw: captured.update(kw))

    result = CliRunner().invoke(phase3.main, ["--create-pr", "--pr-dry-run"])

    assert result.exit_code == 0, result.output
    assert captured["create_pr"] is True
    assert captured["pr_dry_run"] is True


def test_handle_pr_refuses_non_deployable_run(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    info = phase3._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=_hermes_repo(tmp_path),
        evolved_sections=[_section(EVOLVED_CONTENT)],
        baseline_sections=[_section(BASELINE_CONTENT)],
        run_metrics=_run_metrics(deployable=False),
    )

    assert info["created"] is False
    assert info["branch_pushed"] is False
    assert "not deployable" in info["skipped_reason"]


def test_handle_pr_refuses_non_improving_run(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    info = phase3._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=_hermes_repo(tmp_path),
        evolved_sections=[_section(EVOLVED_CONTENT)],
        baseline_sections=[_section(BASELINE_CONTENT)],
        run_metrics=_run_metrics(improvement=-0.01),
    )

    assert info["created"] is False
    assert "no positive proxy improvement" in info["skipped_reason"]


def test_handle_pr_refuses_when_baseline_not_in_source(monkeypatch, tmp_path):
    # Simulates the common Phase 3 reality: extracted constants (joined string
    # parts) don't appear verbatim in the source — the PR must be refused.
    _forbid_subprocess(monkeypatch)

    info = phase3._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=_hermes_repo(tmp_path, body="different source text"),
        evolved_sections=[_section(EVOLVED_CONTENT)],
        baseline_sections=[_section(BASELINE_CONTENT)],
        run_metrics=_run_metrics(),
    )

    assert info["created"] is False
    assert info["branch_pushed"] is False
    assert "not found verbatim" in info["error"]


def test_handle_pr_refuses_when_nothing_changed(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    info = phase3._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=_hermes_repo(tmp_path),
        evolved_sections=[_section(BASELINE_CONTENT)],
        baseline_sections=[_section(BASELINE_CONTENT)],
        run_metrics=_run_metrics(),
    )

    assert info["created"] is False
    assert "no changed prompt sections" in info["error"]


def test_pr_dry_run_renders_redacted_preview_and_touches_no_git(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    secret = "sk-or-v1-abcdef1234567890abcdef"
    info = phase3._handle_pr_request(
        create_pr=True,
        pr_dry_run=True,
        hermes_agent_path=_hermes_repo(tmp_path),
        evolved_sections=[_section(f"{EVOLVED_CONTENT} leaked {secret}")],
        baseline_sections=[_section(BASELINE_CONTENT)],
        run_metrics=_run_metrics(),
    )

    assert info["dry_run"] is True
    assert info["created"] is False
    assert info["branch_pushed"] is False
    preview = info["preview"]
    assert secret not in preview
    assert "[REDACTED]" in preview
    assert "agent/prompt_builder.py" in preview
    assert "requires human review" in preview


def test_handle_pr_optin_calls_prbuilder_with_patched_source(monkeypatch, tmp_path):
    import evolution.core.pr_builder as pr_builder_mod

    class FakePRBuilder:
        instances = []

        def __init__(self, hermes_agent_path, target_repo="NousResearch/hermes-agent"):
            self.hermes_agent_path = hermes_agent_path
            self.create_pr_calls = []
            FakePRBuilder.instances.append(self)

        def create_pr(self, changes, metrics, title_prefix="evolve"):
            self.create_pr_calls.append((changes, metrics, title_prefix))
            return PRResult(
                success=True,
                branch_name="evolve/prompt-x",
                pr_url="https://github.com/x/pull/4",
                branch_pushed=True,
                pr_created=True,
            )

    monkeypatch.setattr(pr_builder_mod, "PRBuilder", FakePRBuilder)

    repo = _hermes_repo(tmp_path)
    original_source = (repo / "agent" / "prompt_builder.py").read_text(encoding="utf-8")

    info = phase3._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=repo,
        evolved_sections=[_section(EVOLVED_CONTENT)],
        baseline_sections=[_section(BASELINE_CONTENT)],
        run_metrics=_run_metrics(),
    )

    builder = FakePRBuilder.instances[0]
    changes, pr_metrics, _prefix = builder.create_pr_calls[0]
    assert changes[0].file_path == "agent/prompt_builder.py"
    assert BASELINE_CONTENT in changes[0].original_content
    assert EVOLVED_CONTENT in changes[0].evolved_content
    assert BASELINE_CONTENT not in changes[0].evolved_content
    assert pr_metrics.baseline_score == pytest.approx(0.50)
    assert pr_metrics.eval_dataset_size == 10

    assert info["created"] is True
    assert info["url"] == "https://github.com/x/pull/4"
    # The hermes-agent source file must not be modified in place.
    assert (repo / "agent" / "prompt_builder.py").read_text(encoding="utf-8") == original_source
