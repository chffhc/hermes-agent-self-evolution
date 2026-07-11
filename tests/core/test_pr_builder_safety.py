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


def _init_repo_with_origin(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(origin)], check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=repo, check=True)
    return repo


def _intercept_gh(monkeypatch, gh_result):
    """Route `gh` invocations to a stub while running real git commands."""
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "gh":
            if isinstance(gh_result, Exception):
                raise gh_result
            return gh_result
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)


def _change() -> list[PRChange]:
    return [
        PRChange(
            file_path="README.md",
            original_content="baseline\n",
            evolved_content="changed\n",
            change_type="code",
        )
    ]


def test_pr_create_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    repo = _init_repo_with_origin(tmp_path)
    _intercept_gh(
        monkeypatch,
        subprocess.CompletedProcess(["gh"], 1, stdout="", stderr="gh: not authenticated"),
    )

    result = PRBuilder(repo).create_pr(_change(), _metrics())

    assert not result.success
    assert result.branch_pushed
    assert not result.pr_created
    assert result.pr_url is None
    assert "gh pr create failed" in (result.error or "")


def test_missing_gh_cli_is_not_reported_as_success(tmp_path, monkeypatch):
    repo = _init_repo_with_origin(tmp_path)
    _intercept_gh(monkeypatch, FileNotFoundError("gh"))

    result = PRBuilder(repo).create_pr(_change(), _metrics())

    assert not result.success
    assert result.branch_pushed
    assert not result.pr_created
    assert "gh CLI not found" in (result.error or "")


def test_pr_body_and_diff_redact_secret_content(tmp_path, monkeypatch):
    repo = _init_repo_with_origin(tmp_path)
    captured = {}
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "gh":
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(
                cmd, 0, stdout="https://github.com/example/repo/pull/1\n", stderr=""
            )
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    secret = "sk-ant-api03-supersecretvalue1234567890"
    result = PRBuilder(repo).create_pr(
        [
            PRChange(
                file_path="README.md",
                original_content="baseline\n",
                evolved_content=f"changed with {secret}\n",
                change_type="code",
            )
        ],
        _metrics(),
    )

    assert result.success
    pr_body = captured["cmd"][captured["cmd"].index("--body") + 1]
    assert secret not in pr_body
    assert "[REDACTED]" in pr_body
    assert secret not in result.diff_summary
    commit_msg = subprocess.run(
        ["git", "log", "-1", "--format=%B", result.branch_name],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "evolve" in commit_msg  # read the evolved branch, not the restored main
    assert secret not in commit_msg


def test_pr_body_frames_scores_as_local_proxy_eval(tmp_path, monkeypatch):
    repo = _init_repo_with_origin(tmp_path)
    captured = {}
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "gh":
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(
                cmd, 0, stdout="https://github.com/example/repo/pull/1\n", stderr=""
            )
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = PRBuilder(repo).create_pr(_change(), _metrics())

    assert result.success
    pr_title = captured["cmd"][captured["cmd"].index("--title") + 1]
    pr_body = captured["cmd"][captured["cmd"].index("--body") + 1]
    # A tiny local eval must never be presented as a validated improvement.
    assert "proxy score" in pr_title
    assert "local proxy evaluation" in pr_body
    assert "not a production benchmark" in pr_body
    assert "human review" in pr_body
    assert "validated improvement" not in pr_body.replace("not a validated improvement", "")
    commit_msg = subprocess.run(
        ["git", "log", "-1", "--format=%B", result.branch_name],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "local proxy eval score" in commit_msg


def _current_branch(repo: Path) -> str:
    return subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_worktree_restored_and_second_pr_does_not_stack_commits(tmp_path, monkeypatch):
    """After create_pr the checkout must be back on the original branch, so a
    later run branches from it instead of stacking the previous evolved commit."""
    repo = _init_repo_with_origin(tmp_path)
    _intercept_gh(
        monkeypatch,
        subprocess.CompletedProcess(
            ["gh"], 0, stdout="https://github.com/example/repo/pull/1\n", stderr=""
        ),
    )
    builder = PRBuilder(repo)

    first = builder.create_pr(_change(), _metrics())
    assert first.success
    assert _current_branch(repo) == "main"

    second = builder.create_pr(
        [
            PRChange(
                file_path="docs/notes.md",
                original_content="",
                evolved_content="evolved notes\n",
                change_type="code",
            )
        ],
        _metrics(),
    )
    assert second.success
    assert _current_branch(repo) == "main"

    # The second branch must contain exactly its own commit, not the first's.
    count = subprocess.run(
        ["git", "rev-list", "--count", f"main..{second.branch_name}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert count == "1"


def test_worktree_restored_after_push_failure(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)  # no origin remote — push will fail

    result = PRBuilder(repo).create_pr(_change(), _metrics())

    assert not result.success
    assert "Failed to push" in (result.error or "")
    assert _current_branch(repo) == "main"


def test_worktree_restored_after_gh_failure(tmp_path, monkeypatch):
    repo = _init_repo_with_origin(tmp_path)
    _intercept_gh(
        monkeypatch,
        subprocess.CompletedProcess(["gh"], 1, stdout="", stderr="gh: not authenticated"),
    )

    result = PRBuilder(repo).create_pr(_change(), _metrics())

    assert not result.success
    assert result.branch_pushed
    assert _current_branch(repo) == "main"


def test_pr_create_success_reports_pr_created(tmp_path, monkeypatch):
    repo = _init_repo_with_origin(tmp_path)
    _intercept_gh(
        monkeypatch,
        subprocess.CompletedProcess(
            ["gh"], 0, stdout="https://github.com/example/repo/pull/1\n", stderr=""
        ),
    )

    result = PRBuilder(repo).create_pr(_change(), _metrics())

    assert result.success
    assert result.branch_pushed
    assert result.pr_created
    assert result.pr_url == "https://github.com/example/repo/pull/1"
