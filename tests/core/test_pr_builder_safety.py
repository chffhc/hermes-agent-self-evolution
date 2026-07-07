"""Safety tests for PRBuilder."""

import subprocess
from pathlib import Path

from evolution.core.pr_builder import PRBuilder, PRChange, PRMetrics, _safe_branch_name


def _metrics() -> PRMetrics:
    return PRMetrics(
        baseline_score=0.1,
        evolved_score=0.2,
        holdout_score=0.2,
        improvement=0.1,
        improvement_pct=100.0,
        iterations=1,
        optimizer="test",
        eval_dataset_size=1,
        train_examples=1,
        val_examples=0,
        holdout_examples=0,
        elapsed_seconds=1.0,
        cost_estimate="$0.00",
    )


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)


def test_pr_builder_refuses_dirty_worktree(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "dirty.txt").write_text("untracked\n", encoding="utf-8")

    result = PRBuilder(tmp_path).create_pr(
        [
            PRChange(
                file_path="README.md",
                original_content="baseline\n",
                evolved_content="changed\n",
                change_type="code",
            )
        ],
        _metrics(),
    )

    assert not result.success
    assert "dirty worktree" in (result.error or "")


def test_pr_builder_blocks_path_traversal_and_rolls_back(tmp_path):
    _init_repo(tmp_path)

    result = PRBuilder(tmp_path).create_pr(
        [
            PRChange(
                file_path="../escape.txt",
                original_content="",
                evolved_content="owned\n",
                change_type="code",
            )
        ],
        _metrics(),
    )

    assert not result.success
    assert "Path traversal" in (result.error or "")
    assert not (tmp_path.parent / "escape.txt").exists()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "main"


def test_safe_branch_name_strips_shell_metacharacters():
    branch = _safe_branch_name("evolve;rm -rf", ["../bad name", "tool$"], "20260707")
    assert ";" not in branch
    assert " " not in branch
    assert "$" not in branch
    assert ".." not in branch
