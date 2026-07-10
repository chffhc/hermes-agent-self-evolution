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


ORIGINAL_TOOL_SOURCE = '''\
"""Stub terminal tool for revert regression tests."""


def run_terminal(command):
    return command


def register():
    return dict(name="terminal")
'''


@pytest.fixture
def tool_repo(git_repo: Path) -> Path:
    """A clean committed checkout containing a discoverable stub tool."""
    _git(git_repo, "config", "user.email", "test@example.com")
    _git(git_repo, "config", "user.name", "Test User")
    tools_dir = git_repo / "tools"
    tools_dir.mkdir()
    (tools_dir / "terminal.py").write_text(ORIGINAL_TOOL_SOURCE)
    _git(git_repo, "add", ".")
    _git(git_repo, "commit", "-m", "init")
    return git_repo


class TestDarwinianRevertOnFailure:
    """The evolver mutates the checkout in place; any post-evolution gate
    failure must restore the original file and save the variant for review."""

    def _run_darwinian(self, tool_repo: Path, tmp_path: Path, monkeypatch, *, post_tests):
        import evolution.code.evolve_tool_code as etc

        monkeypatch.chdir(tmp_path)  # output/code_evolution/ lands in tmp
        mutated = ORIGINAL_TOOL_SOURCE + "\n# mutated by evolver\n"

        def fake_evolver(organism_path, iterations):
            organism_path.write_text(mutated)
            return True, "evolver log"

        pytest_calls = []

        def fake_pytest(tool_name, hermes_agent_path, test_files=None):
            pytest_calls.append(test_files)
            if len(pytest_calls) == 1:  # baseline run
                return True, "baseline ok"
            return post_tests

        monkeypatch.setattr(etc, "run_darwinian_evolver", fake_evolver)
        monkeypatch.setattr(etc, "run_pytest_for_tool", fake_pytest)
        monkeypatch.setattr(etc, "evaluate_code_fitness", lambda *a, **k: (0.5, {}))
        monkeypatch.setattr(etc, "validate_code_constraints", lambda *a, **k: [])

        etc.evolve_tool_code(tool_name="terminal", hermes_repo=str(tool_repo), engine="darwinian")
        return mutated

    def test_reverts_tool_file_when_post_evolution_tests_fail(
        self, tool_repo: Path, tmp_path: Path, monkeypatch
    ):
        mutated = self._run_darwinian(
            tool_repo, tmp_path, monkeypatch, post_tests=(False, "1 failed")
        )

        # Original code restored — failing evolved code never stays live.
        assert (tool_repo / "tools" / "terminal.py").read_text() == ORIGINAL_TOOL_SOURCE

        failed_dirs = list((tmp_path / "output" / "code_evolution").glob("terminal_*_FAILED"))
        assert len(failed_dirs) == 1
        assert (failed_dirs[0] / "evolved_code_FAILED.py").read_text() == mutated
        reason = (failed_dirs[0] / "failure_reason.txt").read_text()
        assert "FAILED the test suite" in reason
        assert "1 failed" in reason

    def test_keeps_evolved_code_when_post_evolution_tests_pass(
        self, tool_repo: Path, tmp_path: Path, monkeypatch
    ):
        mutated = self._run_darwinian(
            tool_repo, tmp_path, monkeypatch, post_tests=(True, "all passed")
        )

        assert (tool_repo / "tools" / "terminal.py").read_text() == mutated
        assert not list((tmp_path / "output" / "code_evolution").glob("terminal_*_FAILED"))

    def test_reverts_tool_file_when_validation_raises(
        self, tool_repo: Path, tmp_path: Path, monkeypatch
    ):
        import evolution.code.evolve_tool_code as etc

        monkeypatch.chdir(tmp_path)
        mutated = ORIGINAL_TOOL_SOURCE + "\n# mutated by evolver\n"

        def fake_evolver(organism_path, iterations):
            organism_path.write_text(mutated)
            return True, "evolver log"

        def boom(*a, **k):
            raise RuntimeError("validator crashed")

        monkeypatch.setattr(etc, "run_darwinian_evolver", fake_evolver)
        monkeypatch.setattr(etc, "run_pytest_for_tool", lambda *a, **k: (True, "ok"))
        monkeypatch.setattr(etc, "evaluate_code_fitness", lambda *a, **k: (0.5, {}))
        monkeypatch.setattr(etc, "validate_code_constraints", boom)

        with pytest.raises(RuntimeError, match="validator crashed"):
            etc.evolve_tool_code(
                tool_name="terminal", hermes_repo=str(tool_repo), engine="darwinian"
            )

        assert (tool_repo / "tools" / "terminal.py").read_text() == ORIGINAL_TOOL_SOURCE
