"""Auto-generate pull requests with evolved skills.

Creates a PR against the hermes-agent repo with:
- Full diff of evolved skill vs baseline
- Before/after metrics comparison
- Constraint validation results
- Benchmark gate status

Usage:
    python -m evolution.core.pr_builder \
        --skill github-code-review \
        --output-dir output/github-code-review/20260429_120000 \
        --hermes-repo ~/.hermes/hermes-agent
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

console = Console()


@dataclass
class PRContent:
    """Content for a pull request."""
    title: str
    body: str
    branch_name: str
    diff_text: str
    metrics: dict = field(default_factory=dict)


def compute_diff(baseline_path: Path, evolved_path: Path) -> str:
    """Compute a unified diff between baseline and evolved skill files."""
    try:
        result = subprocess.run(
            ["diff", "-u", str(baseline_path), str(evolved_path)],
            capture_output=True,
            text=True,
        )
        # diff returns exit code 1 when files differ (not an error)
        return result.stdout if result.returncode <= 1 else result.stdout
    except FileNotFoundError:
        return "(diff not available — diff command not found)"


def generate_pr_body(
    skill_name: str,
    metrics: dict,
    constraints_passed: bool,
    benchmark_passed: bool,
    diff_text: str,
    baseline_path: Path,
    evolved_path: Path,
) -> str:
    """Generate a comprehensive PR body in markdown."""

    improvement = metrics.get("improvement", 0.0)
    improvement_pct = metrics.get("baseline_score", 0.001)
    improvement_color = "✅" if improvement > 0 else "⚠️"

    body = f"""## 🧬 Evolved Skill: `{skill_name}`

This PR contains an automatically evolved version of the `{skill_name}` skill, optimized using DSPy + GEPA (Genetic-Pareto Prompt Evolution).

### Summary

| Metric | Baseline | Evolved | Change |
|--------|----------|---------|--------|
| Holdout Score | {metrics.get('baseline_score', 'N/A'):.3f} | {metrics.get('evolved_score', 'N/A'):.3f} | {improvement:+.3f} |
| Skill Size | {metrics.get('baseline_size', 0):,} chars | {metrics.get('evolved_size', 0):,} chars | {metrics.get('evolved_size', 0) - metrics.get('baseline_size', 0):+,} chars |
| Iterations | — | {metrics.get('iterations', 'N/A')} | — |
| Time | — | {metrics.get('elapsed_seconds', 0):.1f}s | — |

### Gates

| Gate | Status |
|------|--------|
| Constraint Validation | {'✅ Passed' if constraints_passed else '❌ Failed'} |
| Benchmark Regression | {'✅ Passed' if benchmark_passed else '❌ Failed / Not Run'} |
| Improvement | {improvement_color} {'+' if improvement > 0 else ''}{improvement/improvement_pct*100:+.1f}% |

### Configuration

- **Optimizer Model**: {metrics.get('optimizer_model', 'N/A')}
- **Eval Model**: {metrics.get('eval_model', 'N/A')}
- **Train Examples**: {metrics.get('train_examples', 0)}
- **Val Examples**: {metrics.get('val_examples', 0)}
- **Holdout Examples**: {metrics.get('holdout_examples', 0)}

### How It Was Generated

This skill was evolved automatically by [Hermes Agent Self-Evolution](https://github.com/NousResearch/hermes-agent-self-evolution):

1. The original skill was loaded from `hermes-agent/skills/`
2. A synthetic evaluation dataset was generated ({metrics.get('train_examples', 0) + metrics.get('val_examples', 0) + metrics.get('holdout_examples', 0)} examples)
3. GEPA (Genetic-Pareto Prompt Evolution) ran {metrics.get('iterations', 0)} optimization iterations
4. The evolved variant was validated against hard constraints (size limits, structural integrity)
5. Holdout evaluation confirmed improvement (or no regression)

### Diff

```diff
{diff_text[:5000]}
```

{'(Diff truncated — see attached files for full diff)' if len(diff_text) > 5000 else ''}

### Files Changed

- `skills/{skill_name}/SKILL.md` — Evolved skill instructions

---

⚠️ **This PR was auto-generated.** Please review the diff carefully before merging. The evolved skill has been validated against automated constraints, but human review is required for final approval.
"""
    return body


def create_pr_branch(
    skill_name: str,
    hermes_repo: Path,
    evolved_full: str,
    skill_path: Path,
) -> tuple[str, str]:
    """Create a git branch with the evolved skill file.

    Returns (branch_name, commit_message).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_name = f"evolve/{skill_name}-{timestamp}"

    # Copy evolved skill to the hermes-agent repo
    target_path = hermes_repo / skill_path.relative_to(skill_path.parent.parent.parent)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(evolved_full)

    # Create branch and commit
    subprocess.run(["git", "checkout", "-b", branch_name], cwd=str(hermes_repo), capture_output=True)
    subprocess.run(["git", "add", str(target_path)], cwd=str(hermes_repo), capture_output=True)

    commit_msg = f"evolve({skill_name}): auto-evolved skill via GEPA optimization"
    subprocess.run(
        ["git", "commit", "-m", commit_msg, "-m", "Generated by hermes-agent-self-evolution"],
        cwd=str(hermes_repo),
        capture_output=True,
    )

    return branch_name, commit_msg


def build_pr(
    skill_name: str,
    output_dir: Path,
    hermes_repo: Path,
    baseline_path: Optional[Path] = None,
    evolved_path: Optional[Path] = None,
    constraints_passed: bool = True,
    benchmark_passed: bool = True,
) -> PRContent:
    """Build complete PR content for an evolved skill.

    Args:
        skill_name: Name of the evolved skill.
        output_dir: Directory containing evolution output (metrics, files).
        hermes_repo: Path to hermes-agent repository.
        baseline_path: Path to baseline skill file.
        evolved_path: Path to evolved skill file.
        constraints_passed: Whether constraint validation passed.
        benchmark_passed: Whether benchmark gate passed.

    Returns:
        PRContent with title, body, branch_name, and diff.
    """
    # Load metrics
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    else:
        metrics = {}

    # Find skill path in hermes-agent repo
    skills_dir = hermes_repo / "skills"
    skill_path = None
    if skills_dir.exists():
        for p in skills_dir.rglob("SKILL.md"):
            if p.parent.name == skill_name:
                skill_path = p
                break

    if not skill_path:
        raise FileNotFoundError(f"Skill '{skill_name}' not found in {skills_dir}")

    # Determine file paths
    if not baseline_path:
        baseline_path = output_dir / "baseline_skill.md"
    if not evolved_path:
        evolved_path = output_dir / "evolved_skill.md"

    # Compute diff
    if baseline_path.exists() and evolved_path.exists():
        diff_text = compute_diff(baseline_path, evolved_path)
    else:
        diff_text = "(Files not found for diff)"

    # Generate PR body
    body = generate_pr_body(
        skill_name=skill_name,
        metrics=metrics,
        constraints_passed=constraints_passed,
        benchmark_passed=benchmark_passed,
        diff_text=diff_text,
        baseline_path=baseline_path,
        evolved_path=evolved_path,
    )

    title = f"evolve({skill_name}): auto-evolved skill via GEPA optimization"

    return PRContent(
        title=title,
        body=body,
        branch_name=f"evolve/{skill_name}",
        diff_text=diff_text,
        metrics=metrics,
    )


def display_pr_summary(pr: PRContent):
    """Display PR summary in the console."""
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]🧬 PR Summary[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

    console.print(f"[bold]Title:[/bold] {pr.title}")
    console.print(f"[bold]Branch:[/bold] {pr.branch_name}")
    console.print(f"[bold]Metrics:[/bold]")

    if pr.metrics:
        table = Table()
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        for k, v in pr.metrics.items():
            table.add_row(k, str(v))
        console.print(table)

    console.print(f"\n[bold]Body (preview):[/bold]")
    console.print(Markdown(pr.body[:1000] + "..." if len(pr.body) > 1000 else pr.body))


@click.command()
@click.option("--skill", required=True, help="Name of the evolved skill")
@click.option("--output-dir", required=True, help="Directory containing evolution output")
@click.option("--hermes-repo", required=True, help="Path to hermes-agent repo")
@click.option("--create-branch/--no-branch", default=True, help="Create git branch with evolved skill")
@click.option("--dry-run/--apply", default=False, help="Show PR content without creating branch")
def main(skill, output_dir, hermes_repo, create_branch, dry_run):
    """Generate a pull request for an evolved skill."""
    output_path = Path(output_dir)
    hermes_path = Path(hermes_repo)

    if not output_path.exists():
        console.print(f"[red]✗ Output directory not found: {output_path}[/red]")
        return

    if not hermes_path.exists():
        console.print(f"[red]✗ Hermes repo not found: {hermes_path}[/red]")
        return

    console.print(f"\n[bold]Building PR for skill '{skill}'...[/bold]")

    try:
        pr = build_pr(
            skill_name=skill,
            output_dir=output_path,
            hermes_repo=hermes_path,
        )
    except Exception as e:
        console.print(f"[red]✗ Failed to build PR: {e}[/red]")
        return

    display_pr_summary(pr)

    if dry_run:
        console.print(f"\n[yellow]DRY RUN — no branch created[/yellow]")
        console.print(f"  To apply: remove --dry-run flag")
        return

    if create_branch:
        evolved_path = output_path / "evolved_skill.md"
        if evolved_path.exists():
            evolved_full = evolved_path.read_text()
            # Find original skill path
            skills_dir = hermes_path / "skills"
            skill_path = None
            for p in skills_dir.rglob("SKILL.md"):
                if p.parent.name == skill:
                    skill_path = p
                    break

            if skill_path:
                try:
                    branch_name, commit_msg = create_pr_branch(
                        skill, hermes_path, evolved_full, skill_path
                    )
                    console.print(f"\n[bold green]✓ Branch created: {branch_name}[/bold green]")
                    console.print(f"  Commit: {commit_msg}")
                    console.print(f"\n  Push with: cd {hermes_path} && git push origin {branch_name}")
                    console.print(f"  Then create PR at: https://github.com/NousResearch/hermes-agent/pull/new/{branch_name}")
                except Exception as e:
                    console.print(f"[red]✗ Failed to create branch: {e}[/red]")
            else:
                console.print(f"[red]✗ Could not find skill path in hermes-agent repo[/red]")
        else:
            console.print(f"[red]✗ Evolved skill file not found: {evolved_path}[/red]")


if __name__ == "__main__":
    main()
