"""Regression tests for the shared opt-in PR helper (evolution.core.pr_optin).

Invariants: gate-failing or non-improving runs are refused before any change
is even built, change-build failures never reach git, --pr-dry-run is pure
string rendering with redaction, and the create path goes through PRBuilder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evolution.core.pr_builder import PRChange, PRMetrics, PRResult
from evolution.core.pr_optin import build_source_replacement_changes, handle_opt_in_pr


def _pr_metrics() -> PRMetrics:
    return PRMetrics(
        baseline_score=0.40,
        evolved_score=0.45,
        holdout_score=0.45,
        improvement=0.05,
        improvement_pct=12.5,
        iterations=10,
        optimizer="GEPA (qwen3.6-plus)",
        eval_dataset_size=10,
        train_examples=6,
        val_examples=2,
        holdout_examples=2,
        elapsed_seconds=120.0,
        cost_estimate="~$0.10 (estimated)",
    )


def _run_metrics(**overrides) -> dict:
    metrics = {"deployable": True, "improvement": 0.05}
    metrics.update(overrides)
    return metrics


def _change(evolved: str = "new text") -> PRChange:
    return PRChange(
        file_path="tools/search.py",
        original_content="old text",
        evolved_content=evolved,
        change_type="tool_description",
    )


def _forbid_subprocess(monkeypatch):
    """Any subprocess call from pr_builder means a git/gh side effect leaked."""
    import evolution.core.pr_builder as pr_builder_mod

    def _boom(*args, **kwargs):
        raise AssertionError(f"unexpected subprocess call: {args} {kwargs}")

    monkeypatch.setattr(pr_builder_mod.subprocess, "run", _boom)


class FakePRBuilder:
    instances: list[FakePRBuilder] = []
    result = PRResult(
        success=True,
        branch_name="evolve/search-x",
        pr_url="https://github.com/x/pull/2",
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
        branch_name="evolve/search-x",
        pr_url="https://github.com/x/pull/2",
        branch_pushed=True,
        pr_created=True,
    )
    yield
    FakePRBuilder.instances = []


# ── handle_opt_in_pr gates ──────────────────────────────────────────────


def test_refuses_non_deployable_run_without_building_changes(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    def _must_not_build():
        raise AssertionError("build_changes called for a refused run")

    info = handle_opt_in_pr(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=tmp_path,
        run_metrics=_run_metrics(deployable=False),
        build_changes=_must_not_build,
        pr_metrics=_pr_metrics(),
    )

    assert info["created"] is False
    assert info["branch_pushed"] is False
    assert "not deployable" in info["skipped_reason"]


def test_refuses_non_improving_run(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    def _must_not_build():
        raise AssertionError("build_changes called for a refused run")

    info = handle_opt_in_pr(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=tmp_path,
        run_metrics=_run_metrics(improvement=0.0),
        build_changes=_must_not_build,
        pr_metrics=_pr_metrics(),
    )

    assert info["created"] is False
    assert "no positive proxy improvement" in info["skipped_reason"]


def test_change_build_error_never_reaches_git(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    info = handle_opt_in_pr(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=tmp_path,
        run_metrics=_run_metrics(),
        build_changes=lambda: ([], "baseline text not found verbatim in tools/search.py"),
        pr_metrics=_pr_metrics(),
    )

    assert info["created"] is False
    assert info["branch_pushed"] is False
    assert "not found verbatim" in info["error"]
    assert FakePRBuilder.instances == []


def test_dry_run_renders_redacted_preview_with_no_git_calls(monkeypatch, tmp_path):
    # Real PRBuilder: the dry run must be pure string rendering.
    _forbid_subprocess(monkeypatch)

    secret = "sk-or-v1-abcdef1234567890abcdef"
    info = handle_opt_in_pr(
        create_pr=True,
        pr_dry_run=True,
        hermes_agent_path=tmp_path,
        run_metrics=_run_metrics(),
        build_changes=lambda: ([_change(f"new text with leaked key {secret}")], None),
        pr_metrics=_pr_metrics(),
    )

    assert info["dry_run"] is True
    assert info["created"] is False
    assert info["branch_pushed"] is False
    preview = info["preview"]
    assert secret not in preview
    assert "[REDACTED]" in preview
    assert "tools/search.py" in preview
    assert "requires human review" in preview


def test_create_path_delegates_to_prbuilder(monkeypatch, tmp_path):
    import evolution.core.pr_builder as pr_builder_mod

    monkeypatch.setattr(pr_builder_mod, "PRBuilder", FakePRBuilder)

    info = handle_opt_in_pr(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=tmp_path,
        run_metrics=_run_metrics(),
        build_changes=lambda: ([_change()], None),
        pr_metrics=_pr_metrics(),
    )

    assert len(FakePRBuilder.instances) == 1
    builder = FakePRBuilder.instances[0]
    assert builder.hermes_agent_path == tmp_path
    changes, pr_metrics, _prefix = builder.create_pr_calls[0]
    assert changes[0].file_path == "tools/search.py"
    assert pr_metrics.baseline_score == pytest.approx(0.40)

    assert info["created"] is True
    assert info["branch_pushed"] is True
    assert info["url"] == "https://github.com/x/pull/2"
    assert info["error"] is None


def test_branch_pushed_without_pr_is_not_created(monkeypatch, tmp_path):
    import evolution.core.pr_builder as pr_builder_mod

    monkeypatch.setattr(pr_builder_mod, "PRBuilder", FakePRBuilder)
    FakePRBuilder.result = PRResult(
        success=False,
        branch_name="evolve/search-x",
        error="gh CLI not found",
        branch_pushed=True,
        pr_created=False,
    )

    info = handle_opt_in_pr(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=tmp_path,
        run_metrics=_run_metrics(),
        build_changes=lambda: ([_change()], None),
        pr_metrics=_pr_metrics(),
    )

    assert info["created"] is False
    assert info["branch_pushed"] is True
    assert info["error"] == "gh CLI not found"


def test_lazy_pr_metrics_never_built_for_refused_run(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    def _must_not_build_metrics():
        raise AssertionError("pr_metrics callable invoked for a refused run")

    info = handle_opt_in_pr(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=tmp_path,
        run_metrics=_run_metrics(deployable=False),
        build_changes=lambda: ([_change()], None),
        pr_metrics=_must_not_build_metrics,
    )

    assert info["created"] is False
    assert "not deployable" in info["skipped_reason"]


def test_lazy_pr_metrics_never_built_on_change_build_error(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    def _must_not_build_metrics():
        raise AssertionError("pr_metrics callable invoked despite change-build failure")

    info = handle_opt_in_pr(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=tmp_path,
        run_metrics=_run_metrics(),
        build_changes=lambda: ([], "baseline text not found verbatim in tools/search.py"),
        pr_metrics=_must_not_build_metrics,
    )

    assert info["created"] is False
    assert "not found verbatim" in info["error"]


def test_lazy_pr_metrics_resolved_on_create_path(monkeypatch, tmp_path):
    import evolution.core.pr_builder as pr_builder_mod

    monkeypatch.setattr(pr_builder_mod, "PRBuilder", FakePRBuilder)

    info = handle_opt_in_pr(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=tmp_path,
        run_metrics=_run_metrics(),
        build_changes=lambda: ([_change()], None),
        pr_metrics=_pr_metrics,
    )

    assert info["created"] is True
    _changes, pr_metrics, _prefix = FakePRBuilder.instances[0].create_pr_calls[0]
    assert pr_metrics.baseline_score == pytest.approx(0.40)


def test_custom_no_improvement_reason(monkeypatch, tmp_path):
    _forbid_subprocess(monkeypatch)

    info = handle_opt_in_pr(
        create_pr=True,
        pr_dry_run=False,
        hermes_agent_path=tmp_path,
        run_metrics=_run_metrics(improvement=0.0),
        build_changes=lambda: ([_change()], None),
        pr_metrics=_pr_metrics(),
        no_improvement_reason="no positive holdout proxy improvement",
    )

    assert info["created"] is False
    assert info["skipped_reason"] == "no positive holdout proxy improvement"


# ── build_source_replacement_changes ────────────────────────────────────


def test_replaces_unique_snippet(tmp_path):
    source = tmp_path / "tools" / "search.py"
    source.parent.mkdir(parents=True)
    source.write_text('DESCRIPTION = "Original desc"\n', encoding="utf-8")

    changes, error = build_source_replacement_changes(
        tmp_path,
        {"tools/search.py": [("Original desc", "Improved desc")]},
        "tool_description",
    )

    assert error is None
    assert len(changes) == 1
    assert changes[0].file_path == "tools/search.py"
    assert changes[0].original_content == 'DESCRIPTION = "Original desc"\n'
    assert changes[0].evolved_content == 'DESCRIPTION = "Improved desc"\n'
    assert changes[0].change_type == "tool_description"
    # The source file itself must not be modified — only the PRChange carries it.
    assert source.read_text(encoding="utf-8") == 'DESCRIPTION = "Original desc"\n'


def test_multiple_replacements_in_same_file(tmp_path):
    source = tmp_path / "tools.py"
    source.write_text('A = "alpha one"\nB = "beta two"\n', encoding="utf-8")

    changes, error = build_source_replacement_changes(
        tmp_path,
        {"tools.py": [("alpha one", "alpha NEW"), ("beta two", "beta NEW")]},
        "tool_description",
    )

    assert error is None
    assert changes[0].evolved_content == 'A = "alpha NEW"\nB = "beta NEW"\n'


def test_missing_file_fails_closed(tmp_path):
    changes, error = build_source_replacement_changes(
        tmp_path,
        {"tools/gone.py": [("old", "new")]},
        "tool_description",
    )
    assert changes == []
    assert "not found" in error


def test_snippet_not_found_fails_closed(tmp_path):
    (tmp_path / "tools.py").write_text("something else entirely\n", encoding="utf-8")

    changes, error = build_source_replacement_changes(
        tmp_path,
        {"tools.py": [("Original desc", "Improved desc")]},
        "tool_description",
    )
    assert changes == []
    assert "not found verbatim" in error


def test_ambiguous_snippet_fails_closed(tmp_path):
    (tmp_path / "tools.py").write_text('A = "dup"\nB = "dup"\n', encoding="utf-8")

    changes, error = build_source_replacement_changes(
        tmp_path,
        {"tools.py": [("dup", "unique")]},
        "tool_description",
    )
    assert changes == []
    assert "ambiguous" in error


def test_empty_snippets_fail_closed(tmp_path):
    (tmp_path / "tools.py").write_text('A = "x"\n', encoding="utf-8")

    changes, error = build_source_replacement_changes(
        tmp_path, {"tools.py": [("", "new")]}, "tool_description"
    )
    assert changes == [] and "empty baseline" in error

    changes, error = build_source_replacement_changes(
        tmp_path, {"tools.py": [("x", "   ")]}, "tool_description"
    )
    assert changes == [] and "empty evolved" in error


def test_escaped_literal_falls_back_to_ast_patcher(tmp_path):
    # The constant's decoded value has a real newline that the source spells
    # as "\n" — exact-snippet matching cannot find it, the AST patcher can.
    source = tmp_path / "prompts.py"
    source.write_text('GUIDANCE = "line one\\nline two"\n', encoding="utf-8")

    changes, error = build_source_replacement_changes(
        tmp_path,
        {"prompts.py": [("line one\nline two", "line one\nline two improved")]},
        "prompt_section",
    )

    assert error is None
    assert len(changes) == 1
    evolved_module = {}
    exec(changes[0].evolved_content, evolved_module)
    assert evolved_module["GUIDANCE"] == "line one\nline two improved"
    # Source file untouched; only the PRChange carries the patch.
    assert source.read_text(encoding="utf-8") == 'GUIDANCE = "line one\\nline two"\n'


def test_ast_fallback_ambiguity_fails_closed(tmp_path):
    (tmp_path / "prompts.py").write_text('A = "dup\\ntext"\nB = "dup\\ntext"\n', encoding="utf-8")

    changes, error = build_source_replacement_changes(
        tmp_path,
        {"prompts.py": [("dup\ntext", "new\ntext")]},
        "prompt_section",
    )

    assert changes == []
    assert "not found verbatim" in error
    assert "ambiguous" in error


def test_ast_fallback_not_attempted_for_non_python_files(tmp_path):
    (tmp_path / "SKILL.md").write_text("something else entirely\n", encoding="utf-8")

    changes, error = build_source_replacement_changes(
        tmp_path,
        {"SKILL.md": [("line one\nline two", "new")]},
        "skill",
    )

    assert changes == []
    assert "not found verbatim" in error
    assert "refusing to guess" in error


def test_no_effective_change_fails_closed(tmp_path):
    (tmp_path / "tools.py").write_text('A = "same"\n', encoding="utf-8")

    changes, error = build_source_replacement_changes(
        tmp_path, {"tools.py": [("same", "same")]}, "tool_description"
    )
    assert changes == []
    assert "no effective source changes" in error
