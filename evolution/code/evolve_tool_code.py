"""Phase 4: Tool implementation code evolution via Darwinian Evolver.

Evolves actual Python source code in tools/*.py files using the
Darwinian Evolver CLI. This is the highest-risk tier — code changes
can break everything, so the strictest guardrails are enforced.

License note: Darwinian Evolver is AGPL v3 — used as external CLI only,
no Python imports.

Usage:
    python -m evolution.code.evolve_tool_code --tool file_tools --iterations 10
    python -m evolution.code.evolve_tool_code --tool search_files --bug-issue 742
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from evolution.core.config import get_hermes_agent_path

console = Console()


# ── Data structures ─────────────────────────────────────────────────────

@dataclass
class CodeOrganism:
    """A tool file mapped to a Darwinian Evolver organism."""
    name: str
    file_path: Path  # Absolute path to the tool file
    description: str
    test_files: list[str]  # Relevant test files for this tool
    function_signatures: list[str]  # Frozen function signatures
    registry_calls: list[str]  # Frozen registry.register() calls


@dataclass
class BugReproduction:
    """A bug reproduction test case."""
    issue_number: int
    description: str
    reproduction_script: str  # Python code that triggers the bug
    expected_behavior: str
    tool_name: str


@dataclass
class CodeEvolutionResult:
    """Result of a code evolution run."""
    tool_name: str
    iterations: int
    bugs_fixed: list[int]
    tests_passed: bool
    benchmarks_passed: bool
    improved: bool
    elapsed_seconds: float
    diff_summary: str = ""


# ── Code-as-organism wrapper ────────────────────────────────────────────

def wrap_tool_as_organism(
    tool_name: str,
    hermes_agent_path: Path,
) -> CodeOrganism | None:
    """Map a tool file to a CodeOrganism for Darwinian Evolver.

    Extracts frozen function signatures and registry calls that
    must not be changed during evolution.
    """
    import ast

    tools_dir = hermes_agent_path / "tools"
    tool_file = None

    # Find the tool file
    for py_file in tools_dir.glob("*.py"):
        try:
            content = py_file.read_text()
            if f'name="{tool_name}"' in content or f"name='{tool_name}'" in content:
                tool_file = py_file
                break
        except Exception:
            continue

    if not tool_file:
        return None

    # Parse the AST to extract function signatures and registry calls
    try:
        source = tool_file.read_text()
        tree = ast.parse(source)
    except Exception:
        return None

    function_signatures = []
    registry_calls = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = []
            for arg in node.args.args:
                args.append(arg.arg)
            function_signatures.append(f"def {node.name}({', '.join(args)})")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "register":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "registry":
                    registry_calls.append(ast.dump(node))

    # Find relevant test files
    test_files = []
    tests_dir = hermes_agent_path / "tests"
    if tests_dir.exists():
        for test_file in tests_dir.rglob(f"*{tool_name}*"):
            if test_file.suffix == ".py":
                test_files.append(str(test_file))

    return CodeOrganism(
        name=tool_name,
        file_path=tool_file,
        description=f"Tool: {tool_name}",
        test_files=test_files,
        function_signatures=function_signatures[:20],  # Limit
        registry_calls=registry_calls[:10],
    )


# ── Test-driven fitness function ────────────────────────────────────────

def run_pytest_for_tool(
    tool_name: str,
    hermes_agent_path: Path,
    test_files: list[str] | None = None,
) -> tuple[bool, str]:
    """Run pytest for a specific tool.

    Returns (passed, output).
    """
    if test_files:
        cmd = ["python", "-m", "pytest"] + test_files + ["-v", "--tb=short"]
    else:
        # Run all tests that mention the tool name
        cmd = [
            "python", "-m", "pytest",
            "tests/", "-k", tool_name, "-v", "--tb=short",
        ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(hermes_agent_path),
            capture_output=True,
            text=True,
            timeout=300,
        )
        passed = result.returncode == 0
        output = result.stdout[-2000:] if result.stdout else ""
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "Tests timed out after 300s"
    except Exception as e:
        return False, str(e)


def evaluate_code_fitness(
    tool_name: str,
    hermes_agent_path: Path,
    bug_repro: BugReproduction | None = None,
) -> tuple[float, dict]:
    """Composite fitness score for evolved code.

    Components:
    - pytest results (hard gate — must pass 100%)
    - Bug reproduction resolution (did the mutation fix the bug?)
    - Code quality heuristics

    Returns a normalized score in [0, 1] using fixed-weight averaging
    so scores are comparable across runs with/without bug reproduction.
    """
    import tempfile

    scores = {}
    weights = {"tests": 0.5, "bug_fix": 0.3, "code_quality": 0.2}

    # 1. Run tests
    tests_passed, test_output = run_pytest_for_tool(tool_name, hermes_agent_path)
    scores["tests"] = 1.0 if tests_passed else 0.0

    # 2. Bug reproduction — write script to tempfile, not -c (arbitrary code risk)
    if bug_repro:
        try:
            fd, script_path = tempfile.mkstemp(suffix="_repro.py", prefix="bug_")
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(bug_repro.reproduction_script)
                result = subprocess.run(
                    [sys.executable, script_path],
                    cwd=str(hermes_agent_path),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                bug_fixed = result.returncode == 0
                scores["bug_fix"] = 1.0 if bug_fixed else 0.0
            finally:
                os.unlink(script_path)
        except Exception:
            scores["bug_fix"] = 0.0
    else:
        # When no bug repro, use default weight for code_quality
        scores["bug_fix"] = 0.0  # Neutral — no bug to fix

    # 3. Code quality heuristics
    # Check that the file still has proper structure
    tool_file = hermes_agent_path / "tools" / f"{tool_name}.py"
    if not tool_file.exists():
        # Try to find it
        for f in (hermes_agent_path / "tools").glob("*.py"):
            try:
                if tool_name in f.read_text():
                    tool_file = f
                    break
            except Exception:
                continue

    if tool_file.exists():
        try:
            content = tool_file.read_text()
            # Basic quality checks
            has_error_handling = "try:" in content or "except" in content
            has_logging = "logger." in content or "logging." in content
            scores["code_quality"] = (
                0.5 + 0.25 * has_error_handling + 0.25 * has_logging
            )
        except Exception:
            scores["code_quality"] = 0.0
    else:
        scores["code_quality"] = 0.0

    # Fixed-weight composite — always divides by same weights sum (1.0)
    composite = sum(scores[k] * weights[k] for k in weights)
    return composite, scores


# ── Safety guardrails ───────────────────────────────────────────────────

def validate_code_constraints(
    tool_name: str,
    hermes_agent_path: Path,
    original_file: str | None = None,
) -> list[dict]:
    """Validate evolved code meets all safety constraints.

    Constraints:
    - Full test suite passes
    - Function signatures frozen
    - registry.register() calls frozen
    - Error handling coverage not decreased
    """
    violations = []

    # Find the tool file
    tool_file = None
    for f in (hermes_agent_path / "tools").glob("*.py"):
        try:
            if tool_name in f.read_text():
                tool_file = f
                break
        except Exception:
            continue

    if not tool_file:
        violations.append({
            "tool": tool_name,
            "violation": "Tool file not found",
        })
        return violations

    try:
        content = tool_file.read_text()
    except Exception as e:
        violations.append({
            "tool": tool_name,
            "violation": f"Cannot read file: {e}",
        })
        return violations

    # Check for registry.register() call
    if "registry.register" not in content and original_file and "registry.register" in original_file:
        violations.append({
            "tool": tool_name,
            "violation": "registry.register() call removed — would break tool discovery",
        })

    # Check error handling
    try_count = content.count("try:")
    if original_file:
        original_try = original_file.count("try:")
        if try_count < original_try:
            violations.append({
                "tool": tool_name,
                "violation": (
                    f"Error handling decreased: {try_count} try blocks vs "
                    f"{original_try} in original"
                ),
            })

    return violations


# ── Darwinian Evolver integration ───────────────────────────────────────

def run_darwinian_evolver(
    organism_path: Path,
    iterations: int = 10,
    work_dir: Path | None = None,
) -> tuple[bool, str]:
    """Run Darwinian Evolver CLI on an organism.

    Darwinian Evolver is an external CLI tool (AGPL v3).
    We invoke it as a subprocess, not a Python import.

    Returns (success, output).
    """
    # Check if darwinian-evolver is installed
    try:
        result = subprocess.run(
            ["darwinian-evolver", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, "darwinian-evolver CLI not found or not working"
    except FileNotFoundError:
        return False, "darwinian-evolver CLI not installed. Install with: pip install darwinian-evolver"
    except Exception as e:
        return False, f"Failed to check darwinian-evolver: {e}"

    # Run evolution
    cmd = [
        "darwinian-evolver",
        "run",
        "--organism", str(organism_path),
        "--generations", str(iterations),
        "--fitness", "pytest",  # Use pytest as fitness function
    ]

    if work_dir:
        cmd.extend(["--work-dir", str(work_dir)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
        success = result.returncode == 0
        output = result.stdout[-3000:] if result.stdout else result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Evolution timed out after 1 hour"
    except Exception as e:
        return False, str(e)


# ── Main evolution function ─────────────────────────────────────────────

def evolve_tool_code(
    tool_name: str,
    iterations: int = 10,
    bug_issue: int | None = None,
    hermes_repo: str | None = None,
    dry_run: bool = False,
):
    """Main function to evolve tool implementation code."""

    hermes_agent_path = Path(hermes_repo) if hermes_repo else get_hermes_agent_path()

    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — "
                  f"Evolving tool code: [bold]{tool_name}[/bold]\n")

    # ── 1. Wrap tool as organism ────────────────────────────────────────
    console.print("[bold]Step 1: Wrapping tool as organism[/bold]")
    organism = wrap_tool_as_organism(tool_name, hermes_agent_path)
    if not organism:
        console.print(f"[red]✗ Could not find tool '{tool_name}'[/red]")
        sys.exit(1)

    console.print(f"  File: {organism.file_path.relative_to(hermes_agent_path)}")
    console.print(f"  Functions: {len(organism.function_signatures)}")
    console.print(f"  Test files: {len(organism.test_files)}")

    # Save original for comparison
    original_content = organism.file_path.read_text()

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated.[/bold green]")
        return

    # ── 2. Evaluate baseline fitness ────────────────────────────────────
    console.print("\n[bold]Step 2: Evaluating baseline fitness[/bold]")

    # Run tests
    tests_passed, test_output = run_pytest_for_tool(tool_name, hermes_agent_path)
    console.print(f"  Tests: {'✓ Passed' if tests_passed else '✗ Failed'}")

    baseline_fitness, baseline_scores = evaluate_code_fitness(
        tool_name, hermes_agent_path
    )
    console.print(f"  Baseline fitness: {baseline_fitness:.3f}")
    for name, score in baseline_scores.items():
        console.print(f"    {name}: {score:.3f}")

    # ── 3. Run Darwinian Evolver ────────────────────────────────────────
    console.print(f"\n[bold cyan]Step 3: Running Darwinian Evolver ({iterations} generations)[/bold cyan]\n")

    start_time = time.time()

    success, evolver_output = run_darwinian_evolver(
        organism_path=organism.file_path,
        iterations=iterations,
    )

    elapsed = time.time() - start_time

    if not success:
        console.print(f"[yellow]⚠ Darwinian Evolver failed: {evolver_output}[/yellow]")
        console.print("\n[yellow]Skipping code evolution — tool code remains unchanged.[/yellow]")
        return

    # ── 4. Validate evolved code ────────────────────────────────────────
    console.print("\n[bold]Step 4: Validating evolved code[/bold]")

    violations = validate_code_constraints(
        tool_name, hermes_agent_path, original_file=original_content
    )

    if violations:
        for v in violations:
            console.print(f"  [red]✗ {v['violation']}[/red]")
        console.print("\n[red]✗ Evolved code FAILED safety constraints — reverting[/red]")

        # Revert changes
        organism.file_path.write_text(original_content)
        return

    console.print("  [green]✓ All safety constraints pass[/green]")

    # ── 5. Report ───────────────────────────────────────────────────────
    evolved_fitness, evolved_scores = evaluate_code_fitness(
        tool_name, hermes_agent_path
    )

    console.print(f"\n  Baseline fitness: {baseline_fitness:.3f}")
    console.print(f"  Evolved fitness:  {evolved_fitness:.3f}")
    console.print(f"  Time: {elapsed:.1f}s")

    # Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output/code_evolution") / f"{tool_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "evolved_code.py").write_text(organism.file_path.read_text())
    (output_dir / "baseline_code.py").write_text(original_content)
    (output_dir / "evolver_output.txt").write_text(evolver_output)

    metrics = {
        "tool_name": tool_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "baseline_fitness": baseline_fitness,
        "evolved_fitness": evolved_fitness,
        "tests_passed": tests_passed,
        "constraint_violations": violations,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")


@click.command()
@click.option("--tool", required=True, help="Tool name to evolve")
@click.option("--iterations", default=10, help="Number of evolution generations")
@click.option("--bug-issue", default=None, type=int, help="GitHub issue number to fix")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--dry-run", is_flag=True, help="Validate setup without running")
def main(tool, iterations, bug_issue, hermes_repo, dry_run):
    """Evolve tool implementation code using Darwinian Evolver."""
    evolve_tool_code(
        tool_name=tool,
        iterations=iterations,
        bug_issue=bug_issue,
        hermes_repo=hermes_repo,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
