"""Constraint validators for evolved artifacts.

Every candidate variant must pass ALL constraints before it can be
considered valid. Failed constraints = immediate rejection.
"""

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from evolution.core.config import EvolutionConfig

# Directories that are never needed to run the test suite and can be huge.
_WORKSPACE_COPY_IGNORE = shutil.ignore_patterns(
    ".git",
    "venv",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
)


@dataclass
class ConstraintResult:
    """Result of constraint validation."""

    passed: bool
    constraint_name: str
    message: str
    details: str | None = None


class ConstraintValidator:
    """Validates evolved artifacts against hard constraints."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def validate_all(
        self,
        artifact_text: str,
        artifact_type: str,
        baseline_text: str | None = None,
    ) -> list[ConstraintResult]:
        """Run all applicable constraints. Returns list of results."""
        results = []

        # 1. Size limits
        results.append(self._check_size(artifact_text, artifact_type))

        # 2. Growth limit (if baseline provided)
        if baseline_text:
            results.append(self._check_growth(artifact_text, baseline_text, artifact_type))

        # 3. Non-empty
        results.append(self._check_non_empty(artifact_text))

        # 4. Structural integrity
        if artifact_type == "skill":
            results.append(self._check_skill_structure(artifact_text))

        return results

    def run_test_suite(
        self,
        hermes_repo: Path,
        artifact_relpath: str | Path | None = None,
        artifact_text: str | None = None,
    ) -> ConstraintResult:
        """Run the hermes-agent test suite, optionally with an evolved artifact applied.

        When ``artifact_relpath`` and ``artifact_text`` are both given, the repo is
        copied to a temporary workspace, the evolved artifact is written over the
        file at ``artifact_relpath`` inside that copy, and pytest runs against the
        copy. This is a real artifact-then-test gate: the evolved content is what
        gets tested, and the live checkout is never touched.

        Without an artifact, this is only a *repo sanity check* — it proves the
        pristine checkout's tests pass, and says nothing about the evolved artifact.
        The constraint name reflects which mode ran.
        """
        if (artifact_relpath is None) != (artifact_text is None):
            return ConstraintResult(
                passed=False,
                constraint_name="artifact_test_suite",
                message=(
                    "Both artifact_relpath and artifact_text are required to test "
                    "an evolved artifact — refusing to run a misleading gate"
                ),
            )

        if artifact_relpath is None:
            result = self._run_pytest(hermes_repo, "repo_sanity_test_suite")
            if result.passed:
                result.message = (
                    "Repo tests passed (sanity check only — evolved artifact was NOT applied)"
                )
            return result

        try:
            with tempfile.TemporaryDirectory(prefix="hermes_artifact_gate_") as tmp:
                workspace = Path(tmp) / "workspace"
                shutil.copytree(
                    hermes_repo,
                    workspace,
                    ignore=_WORKSPACE_COPY_IGNORE,
                    symlinks=True,
                )
                target = (workspace / artifact_relpath).resolve()
                if not target.is_relative_to(workspace.resolve()):
                    return ConstraintResult(
                        passed=False,
                        constraint_name="artifact_test_suite",
                        message=f"Artifact path escapes workspace: {artifact_relpath}",
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(artifact_text, encoding="utf-8")

                result = self._run_pytest(workspace, "artifact_test_suite")
                if result.passed:
                    result.message = (
                        "All tests passed with evolved artifact applied in temp workspace"
                    )
                return result
        except Exception as e:
            return ConstraintResult(
                passed=False,
                constraint_name="artifact_test_suite",
                message=f"Failed to prepare artifact test workspace: {e}",
            )

    def _run_pytest(self, repo: Path, constraint_name: str) -> ConstraintResult:
        """Run pytest in ``repo``. Must pass 100%."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(repo),
            )

            if result.returncode == 0:
                return ConstraintResult(
                    passed=True,
                    constraint_name=constraint_name,
                    message="All tests passed",
                    details=result.stdout.strip().split("\n")[-1] if result.stdout else "",
                )
            else:
                # Extract failure summary
                last_lines = result.stdout.strip().split("\n")[-5:] if result.stdout else []
                return ConstraintResult(
                    passed=False,
                    constraint_name=constraint_name,
                    message="Test suite failed",
                    details="\n".join(last_lines),
                )
        except subprocess.TimeoutExpired:
            return ConstraintResult(
                passed=False,
                constraint_name=constraint_name,
                message="Test suite timed out (300s)",
            )
        except Exception as e:
            return ConstraintResult(
                passed=False,
                constraint_name=constraint_name,
                message=f"Failed to run tests: {e}",
            )

    def _check_size(self, text: str, artifact_type: str) -> ConstraintResult:
        # Use byte count for accurate size measurement (multi-byte UTF-8)
        size = len(text.encode("utf-8"))
        if artifact_type == "skill":
            limit = self.config.max_skill_size
        elif artifact_type == "tool_description":
            limit = self.config.max_tool_desc_size
        elif artifact_type == "param_description":
            limit = self.config.max_param_desc_size
        else:
            limit = self.config.max_skill_size  # Default

        if size <= limit:
            return ConstraintResult(
                passed=True,
                constraint_name="size_limit",
                message=f"Size OK: {size}/{limit} chars",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="size_limit",
                message=f"Size exceeded: {size}/{limit} chars ({size - limit} over)",
            )

    def _check_growth(self, text: str, baseline: str, artifact_type: str) -> ConstraintResult:
        growth = (len(text) - len(baseline)) / max(1, len(baseline))
        max_growth = self.config.max_prompt_growth

        if growth <= max_growth:
            return ConstraintResult(
                passed=True,
                constraint_name="growth_limit",
                message=f"Growth OK: {growth:+.1%} (max {max_growth:+.1%})",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="growth_limit",
                message=f"Growth exceeded: {growth:+.1%} (max {max_growth:+.1%})",
            )

    def _check_non_empty(self, text: str) -> ConstraintResult:
        if text.strip():
            return ConstraintResult(
                passed=True,
                constraint_name="non_empty",
                message="Artifact is non-empty",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="non_empty",
                message="Artifact is empty",
            )

    def _check_skill_structure(self, text: str) -> ConstraintResult:
        """Check that a skill file has valid YAML frontmatter and markdown body."""
        has_frontmatter = text.strip().startswith("---")
        has_name = "name:" in text[:500] if has_frontmatter else False
        has_description = "description:" in text[:500] if has_frontmatter else False

        if has_frontmatter and has_name and has_description:
            return ConstraintResult(
                passed=True,
                constraint_name="skill_structure",
                message="Skill has valid frontmatter (name + description)",
            )
        else:
            missing = []
            if not has_frontmatter:
                missing.append("YAML frontmatter (---)")
            if not has_name:
                missing.append("name field")
            if not has_description:
                missing.append("description field")
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message=f"Skill missing: {', '.join(missing)}",
            )
