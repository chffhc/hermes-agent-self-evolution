"""Regression tests for the opt-in Phase 2 PR step.

Same invariants as Phase 1: no PR-related git operation by default, gates
refuse failed/non-improving runs, --pr-dry-run is pure redacted rendering,
and descriptions that cannot be located verbatim in their source file refuse
the PR instead of guessing.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from click.testing import CliRunner

from evolution.core.pr_builder import PRResult
from evolution.tools import evolve_tool_descriptions as phase2
from evolution.tools.evolve_tool_descriptions import ToolDescription


def _tool(description: str = "Improved desc") -> ToolDescription:
    return ToolDescription(
        name="search",
        toolset="core",
        description=description,
        param_descriptions={},
        schema={},
        file_path="tools/search.py",
    )


def _run_metrics(**overrides) -> dict:
    metrics = {
        "deployable": True,
        "improvement": 0.10,
        "baseline_accuracy": 0.60,
        "evolved_accuracy": 0.70,
        "iterations": 10,
        "optimizer_model": "qwen3.6-plus",
        "num_tools": 1,
        "train_examples": 6,
        "val_examples": 2,
        "holdout_examples": 2,
        "elapsed_seconds": 90.0,
    }
    metrics.update(overrides)
    return metrics


def _forbid_subprocess(monkeypatch):
    import evolution.core.pr_builder as pr_builder_mod

    def _boom(*args, **kwargs):
        raise AssertionError(f"unexpected subprocess call: {args} {kwargs}")

    monkeypatch.setattr(pr_builder_mod.subprocess, "run", _boom)


def _hermes_repo(tmp_path: Path, source: str = 'DESCRIPTION = "Original desc"\n') -> Path:
    (tmp_path / "tools").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tools" / "search.py").write_text(source, encoding="utf-8")
    return tmp_path


def test_evolve_defaults_never_request_pr():
    sig = inspect.signature(phase2.evolve_tool_descriptions)
    assert sig.parameters["create_pr"].default is False
    assert sig.parameters["pr_dry_run"].default is False


def test_cli_defaults_pr_flags_off(monkeypatch):
    captured = {}
    monkeypatch.setattr(phase2, "evolve_tool_descriptions", lambda **kw: captured.update(kw))

    result = CliRunner().invoke(phase2.main, [])

    assert result.exit_code == 0, result.output
    assert captured["create_pr"] is False
    assert captured["pr_dry_run"] is False


def test_cli_threads_pr_flags(monkeypatch):
    captured = {}
    monkeypatch.setattr(phase2, "evolve_tool_descriptions", lambda **kw: captured.update(kw))

    result = CliRunner().invoke(phase2.main, ["--create-pr", "--pr-dry-run"])

    assert result.exit_code == 0, result.output
    assert captured["create_pr"] is True
    assert captured["pr_dry_run"] is True


def test_handle_pr_refuses_non_deployable_run(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    info = phase2._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=_hermes_repo(tmp_path),
        evolved_tools=[_tool()],
        baseline_descriptions={"search": "Original desc"},
        run_metrics=_run_metrics(deployable=False),
    )

    assert info["created"] is False
    assert info["branch_pushed"] is False
    assert "not deployable" in info["skipped_reason"]


def test_handle_pr_refuses_non_improving_run(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    info = phase2._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=_hermes_repo(tmp_path),
        evolved_tools=[_tool()],
        baseline_descriptions={"search": "Original desc"},
        run_metrics=_run_metrics(improvement=0.0),
    )

    assert info["created"] is False
    assert "no positive proxy improvement" in info["skipped_reason"]


def test_handle_pr_refuses_when_baseline_not_in_source(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    info = phase2._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=_hermes_repo(tmp_path, source="# no description string here\n"),
        evolved_tools=[_tool()],
        baseline_descriptions={"search": "Original desc"},
        run_metrics=_run_metrics(),
    )

    assert info["created"] is False
    assert info["branch_pushed"] is False
    assert "not found verbatim" in info["error"]


def test_handle_pr_refuses_when_nothing_changed(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    info = phase2._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=_hermes_repo(tmp_path),
        evolved_tools=[_tool(description="Original desc")],
        baseline_descriptions={"search": "Original desc"},
        run_metrics=_run_metrics(),
    )

    assert info["created"] is False
    assert "no changed tool descriptions" in info["error"]


def test_pr_dry_run_renders_redacted_preview_and_touches_no_git(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    secret = "sk-or-v1-abcdef1234567890abcdef"
    info = phase2._handle_pr_request(
        create_pr=True,
        pr_dry_run=True,
        hermes_agent_path=_hermes_repo(tmp_path),
        evolved_tools=[_tool(description=f"Improved desc with leaked key {secret}")],
        baseline_descriptions={"search": "Original desc"},
        run_metrics=_run_metrics(),
    )

    assert info["dry_run"] is True
    assert info["created"] is False
    assert info["branch_pushed"] is False
    preview = info["preview"]
    assert secret not in preview
    assert "[REDACTED]" in preview
    assert "tools/search.py" in preview
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
                branch_name="evolve/search-x",
                pr_url="https://github.com/x/pull/3",
                branch_pushed=True,
                pr_created=True,
            )

    monkeypatch.setattr(pr_builder_mod, "PRBuilder", FakePRBuilder)

    repo = _hermes_repo(tmp_path)
    info = phase2._handle_pr_request(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=repo,
        evolved_tools=[_tool()],
        baseline_descriptions={"search": "Original desc"},
        run_metrics=_run_metrics(),
    )

    builder = FakePRBuilder.instances[0]
    changes, pr_metrics, _prefix = builder.create_pr_calls[0]
    assert changes[0].file_path == "tools/search.py"
    assert changes[0].original_content == 'DESCRIPTION = "Original desc"\n'
    assert changes[0].evolved_content == 'DESCRIPTION = "Improved desc"\n'
    assert pr_metrics.baseline_score == pytest.approx(0.60)
    assert pr_metrics.evolved_score == pytest.approx(0.70)
    assert pr_metrics.eval_dataset_size == 10

    assert info["created"] is True
    assert info["url"] == "https://github.com/x/pull/3"
    # The hermes-agent source file must not be modified in place.
    assert (repo / "tools" / "search.py").read_text(encoding="utf-8") == (
        'DESCRIPTION = "Original desc"\n'
    )
