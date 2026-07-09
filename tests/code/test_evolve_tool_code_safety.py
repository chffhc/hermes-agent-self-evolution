"""Safety regression tests for the darwinian tool-code evolution path."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evolution.code.evolve_tool_code import ensure_clean_git_checkout, evolve_tool_code
from evolution.core.errors import EvolutionError


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    return repo


class TestEnsureCleanGitCheckout:
    def test_rejects_non_git_directory(self, tmp_path: Path):
        ok, reason = ensure_clean_git_checkout(tmp_path)

        assert not ok
        assert "not a git repository" in reason

    def test_rejects_dirty_tree(self, git_repo: Path):
        (git_repo / "wip.py").write_text("x = 1\n")

        ok, reason = ensure_clean_git_checkout(git_repo)

        assert not ok
        assert "uncommitted change" in reason

    def test_accepts_clean_tree(self, git_repo: Path):
        ok, reason = ensure_clean_git_checkout(git_repo)

        assert ok
        assert reason == "clean"


class TestDarwinianCleanCheckoutGuard:
    def test_refuses_dirty_checkout_before_touching_files(self, git_repo: Path):
        (git_repo / "wip.py").write_text("uncommitted\n")

        with pytest.raises(EvolutionError, match="Refusing to run darwinian engine"):
            evolve_tool_code(
                tool_name="terminal",
                hermes_repo=str(git_repo),
                engine="darwinian",
            )

    def test_refuses_non_git_checkout(self, tmp_path: Path):
        with pytest.raises(EvolutionError, match="Refusing to run darwinian engine"):
            evolve_tool_code(
                tool_name="terminal",
                hermes_repo=str(tmp_path),
                engine="darwinian",
            )
