"""Auto-generate PRs against hermes-agent with evolved artifacts.

Creates a git branch, applies evolved changes, and generates a PR
with full metrics, diffs, and before/after comparisons.

All evolved changes go through PR — never direct commit.
"""

import difflib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class PRChange:
    """A single file change in the PR."""
    file_path: str  # Relative path in hermes-agent repo
    original_content: str
    evolved_content: str
    change_type: str  # "skill", "tool_description", "prompt_section", "code"


@dataclass
class PRMetrics:
    """Metrics to include in the PR body."""
    baseline_score: float
    evolved_score: float
    holdout_score: float
    improvement: float
    improvement_pct: float
    iterations: int
    optimizer: str
    eval_dataset_size: int
    train_examples: int
    val_examples: int
    holdout_examples: int
    elapsed_seconds: float
    cost_estimate: str
    constraint_violations: list[str] = field(default_factory=list)
    benchmark_regressions: list[str] = field(default_factory=list)


@dataclass
class PRResult:
    """Result of creating a PR."""
    success: bool
    branch_name: str
    pr_url: Optional[str] = None
    error: Optional[str] = None
    diff_summary: str = ""


class PRBuilder:
    """Builds and creates PRs for evolved artifacts."""

    def __init__(
        self,
        hermes_agent_path: Path,
        target_repo: str = "NousResearch/hermes-agent",
    ):
        self.hermes_agent_path = hermes_agent_path
        self.target_repo = target_repo

    def create_pr(
        self,
        changes: list[PRChange],
        metrics: PRMetrics,
        title_prefix: str = "evolve",
    ) -> PRResult:
        """Create a PR with evolved changes.

        Steps:
        1. Create a new branch in hermes-agent repo
        2. Apply evolved file changes
        3. Commit with detailed message
        4. Push and create PR via gh CLI
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        change_names = _extract_change_names(changes)
        branch_name = f"{title_prefix}/{'-'.join(change_names)}-{timestamp}"

        # Ensure we're in a clean state
        try:
            self._run_git(["checkout", "main"], cwd=self.hermes_agent_path)
            self._run_git(["pull", "origin", "main"], cwd=self.hermes_agent_path)
        except subprocess.CalledProcessError as e:
            return PRResult(
                success=False,
                branch_name=branch_name,
                error=f"Failed to checkout main: {e.stderr}",
            )

        # Create new branch
        try:
            self._run_git(
                ["checkout", "-b", branch_name],
                cwd=self.hermes_agent_path,
            )
        except subprocess.CalledProcessError as e:
            return PRResult(
                success=False,
                branch_name=branch_name,
                error=f"Failed to create branch: {e.stderr}",
            )

        # Apply changes
        files_changed = []
        for change in changes:
            target_path = (self.hermes_agent_path / change.file_path).resolve()
            # Path traversal guard: ensure target stays within hermes-agent repo
            repo_resolved = self.hermes_agent_path.resolve()
            if not target_path.is_relative_to(repo_resolved):
                return PRResult(
                    success=False,
                    branch_name=branch_name,
                    error=f"Path traversal detected: {change.file_path} escapes repo directory",
                )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(change.evolved_content, encoding="utf-8")
            files_changed.append(change.file_path)

        # Generate diff summary
        diff_summary = self._generate_diff_summary(changes)

        # Commit
        commit_msg = self._build_commit_message(changes, metrics)
        try:
            self._run_git(["add"] + files_changed, cwd=self.hermes_agent_path)
            self._run_git(["commit", "-m", commit_msg], cwd=self.hermes_agent_path)
        except subprocess.CalledProcessError as e:
            return PRResult(
                success=False,
                branch_name=branch_name,
                error=f"Failed to commit: {e.stderr}",
            )

        # Push
        try:
            self._run_git(
                ["push", "-u", "origin", branch_name],
                cwd=self.hermes_agent_path,
            )
        except subprocess.CalledProcessError as e:
            return PRResult(
                success=False,
                branch_name=branch_name,
                error=f"Failed to push: {e.stderr}",
            )

        # Create PR via gh CLI
        pr_body = self._build_pr_body(changes, metrics, diff_summary)
        pr_title = f"{title_prefix}: {' & '.join(change_names)} (score {metrics.baseline_score:.3f} → {metrics.evolved_score:.3f})"

        try:
            pr_output = self._run_git(
                [
                    "pr", "create",
                    "--title", pr_title,
                    "--body", pr_body,
                    "--base", "main",
                ],
                cwd=self.hermes_agent_path,
                capture_output=True,
            )
            if pr_output.returncode == 0:
                pr_url = pr_output.stdout.strip()
            else:
                # gh CLI failed — branch is still created, log the error
                import logging
                logging.getLogger(__name__).warning(
                    "gh pr create failed: %s", pr_output.stderr.strip()
                )
                pr_url = None
        except subprocess.CalledProcessError:
            # gh CLI not available or not authenticated — branch is still created
            pr_url = None

        return PRResult(
            success=True,
            branch_name=branch_name,
            pr_url=pr_url,
            diff_summary=diff_summary,
        )

    def _generate_diff_summary(self, changes: list[PRChange]) -> str:
        """Generate a unified diff summary for all changes."""
        lines = []
        for change in changes:
            original_lines = change.original_content.splitlines(keepends=True)
            evolved_lines = change.evolved_content.splitlines(keepends=True)

            diff = difflib.unified_diff(
                original_lines,
                evolved_lines,
                fromfile=f"baseline/{change.file_path}",
                tofile=f"evolved/{change.file_path}",
                n=3,
            )
            lines.append(f"\n### {change.file_path}\n\n```diff")
            lines.extend(diff)
            lines.append("```")

        return "\n".join(lines)

    def _build_commit_message(
        self, changes: list[PRChange], metrics: PRMetrics
    ) -> str:
        """Build a detailed git commit message."""
        change_names = _extract_change_names(changes)
        names_str = ", ".join(change_names)

        msg = (
            f"evolve: {names_str} — score improved "
            f"{metrics.baseline_score:.3f} → {metrics.evolved_score:.3f} "
            f"({metrics.improvement_pct:+.1f}%)\n\n"
            f"Optimizer: {metrics.optimizer} ({metrics.iterations} iterations)\n"
            f"Eval dataset: {metrics.eval_dataset_size} examples "
            f"({metrics.train_examples} train / {metrics.val_examples} val / "
            f"{metrics.holdout_examples} holdout)\n"
            f"Before: {metrics.baseline_score:.3f}\n"
            f"After: {metrics.evolved_score:.3f}\n"
            f"Holdout: {metrics.holdout_score:.3f}\n"
            f"Time: {metrics.elapsed_seconds:.0f}s\n"
            f"Cost estimate: {metrics.cost_estimate}"
        )

        if metrics.constraint_violations:
            msg += "\n\nConstraint violations caught during evolution:\n"
            for v in metrics.constraint_violations:
                msg += f"  - {v}\n"

        return msg

    def _build_pr_body(
        self,
        changes: list[PRChange],
        metrics: PRMetrics,
        diff_summary: str,
    ) -> str:
        """Build the PR body with full metrics and comparison."""
        change_names = _extract_change_names(changes)

        body = (
            f"## 🧬 Self-Evolution: {' & '.join(change_names)}\n\n"
            f"Automatically evolved via DSPy + GEPA optimization pipeline.\n\n"
            f"### Score Comparison\n\n"
            f"| Metric | Baseline | Evolved | Change |\n"
            f"|--------|----------|---------|--------|\n"
            f"| **Score** | {metrics.baseline_score:.3f} | {metrics.evolved_score:.3f} | "
            f"**{metrics.improvement:+.3f} ({metrics.improvement_pct:+.1f}%)** |\n"
            f"| Holdout | — | {metrics.holdout_score:.3f} | — |\n\n"
            f"### Optimization Details\n\n"
            f"- **Optimizer:** {metrics.optimizer}\n"
            f"- **Iterations:** {metrics.iterations}\n"
            f"- **Eval Dataset:** {metrics.eval_dataset_size} examples\n"
            f"- **Train/Val/Holdout:** {metrics.train_examples} / {metrics.val_examples} / {metrics.holdout_examples}\n"
            f"- **Time:** {metrics.elapsed_seconds:.0f}s\n"
            f"- **Estimated Cost:** {metrics.cost_estimate}\n\n"
        )

        if metrics.constraint_violations:
            body += (
                f"### Constraint Violations (filtered during evolution)\n\n"
            )
            for v in metrics.constraint_violations:
                body += f"- {v}\n"
            body += "\n"

        if metrics.benchmark_regressions:
            body += (
                f"### ⚠️ Benchmark Regressions\n\n"
                f"These variants were rejected due to benchmark regression:\n\n"
            )
            for r in metrics.benchmark_regressions:
                body += f"- {r}\n"
            body += "\n"

        body += f"### Changes\n\n{diff_summary}\n\n"
        body += (
            "---\n\n"
            f"*Generated by Hermes Agent Self-Evolution on {datetime.now().strftime('%Y-%m-%d %H:%M')}.*\n"
            f"*This PR requires human review before merge.*"
        )

        return body

    def _run_git(
        self,
        args: list[str],
        cwd: Path,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = ["git"] + args
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=capture_output,
            text=True,
        )
        if result.returncode != 0 and not capture_output:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        return result


def _extract_change_names(changes: list[PRChange]) -> list[str]:
    """Extract human-readable names from changes."""
    names = []
    for change in changes:
        # Try to get the name from the file path
        path = Path(change.file_path)
        if "skills" in change.file_path:
            # skills/category/name/SKILL.md -> name (second dir after "skills")
            # or skills/name/SKILL.md -> name (first dir after "skills")
            parts = path.parts
            for i, part in enumerate(parts):
                if part == "skills":
                    # If the path has SKILL.md as the filename, get the parent dir
                    if path.name == "SKILL.md" and path.parent.name:
                        names.append(path.parent.name)
                    elif i + 1 < len(parts):
                        names.append(parts[i + 1])
                    break
        elif "tools" in change.file_path:
            names.append(path.stem)
        else:
            names.append(path.stem)
    return names if names else ["artifact"]


def create_pr_from_output(
    output_dir: Path,
    hermes_agent_path: Path,
    metrics: PRMetrics,
    change_type: str = "skill",
) -> PRResult:
    """Convenience function to create a PR from evolution output directory.

    Reads the evolved and baseline files from the output directory
    and creates a PR automatically.
    """
    builder = PRBuilder(hermes_agent_path=hermes_agent_path)

    # Read files
    evolved_file = output_dir / "evolved_skill.md"
    baseline_file = output_dir / "baseline_skill.md"

    if not evolved_file.exists():
        return PRResult(
            success=False,
            branch_name="",
            error=f"Evolved file not found: {evolved_file}",
        )

    evolved_content = evolved_file.read_text()
    baseline_content = ""
    if baseline_file.exists():
        baseline_content = baseline_file.read_text()

    # Determine the file path in hermes-agent
    # Inferred from output directory structure: output/{skill_name}/timestamp/
    skill_name = output_dir.parent.name
    file_path = f"skills/{skill_name}/SKILL.md"

    changes = [
        PRChange(
            file_path=file_path,
            original_content=baseline_content,
            evolved_content=evolved_content,
            change_type=change_type,
        )
    ]

    return builder.create_pr(changes, metrics)
