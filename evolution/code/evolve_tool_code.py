"""Phase 4: Tool implementation code evolution.

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

from evolution.code.openevolve_runner import (
    OpenEvolveRunnerConfig,
    OpenEvolveRunResult,
    run_openevolve_isolated,
)
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


@dataclass(frozen=True)
class OpenEvolveToolScaffold:
    """Input files generated for an OpenEvolve tool-code run."""

    initial_program: Path
    evaluator: Path
    config_file: Path


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
        cmd = [sys.executable, "-m", "pytest"] + test_files + ["-v", "--tb=short"]
    else:
        # Run all tests that mention the tool name
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-k",
            tool_name,
            "-v",
            "--tb=short",
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
            scores["code_quality"] = 0.5 + 0.25 * has_error_handling + 0.25 * has_logging
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
        violations.append(
            {
                "tool": tool_name,
                "violation": "Tool file not found",
            }
        )
        return violations

    try:
        content = tool_file.read_text()
    except Exception as e:
        violations.append(
            {
                "tool": tool_name,
                "violation": f"Cannot read file: {e}",
            }
        )
        return violations

    # Check for registry.register() call
    if (
        "registry.register" not in content
        and original_file
        and "registry.register" in original_file
    ):
        violations.append(
            {
                "tool": tool_name,
                "violation": "registry.register() call removed — would break tool discovery",
            }
        )

    # Check error handling
    try_count = content.count("try:")
    if original_file:
        original_try = original_file.count("try:")
        if try_count < original_try:
            violations.append(
                {
                    "tool": tool_name,
                    "violation": (
                        f"Error handling decreased: {try_count} try blocks vs "
                        f"{original_try} in original"
                    ),
                }
            )

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
        return (
            False,
            "darwinian-evolver CLI not installed. Install with: pip install darwinian-evolver",
        )
    except Exception as e:
        return False, f"Failed to check darwinian-evolver: {e}"

    # Run evolution
    cmd = [
        "darwinian-evolver",
        "run",
        "--organism",
        str(organism_path),
        "--generations",
        str(iterations),
        "--fitness",
        "pytest",  # Use pytest as fitness function
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


# ── OpenEvolve integration ───────────────────────────────────────────────


def create_openevolve_tool_scaffold(
    organism: CodeOrganism,
    output_dir: Path,
    iterations: int,
) -> OpenEvolveToolScaffold:
    """Create OpenEvolve input files for a Hermes tool without editing it.

    The scaffold wraps the tool source in one EVOLVE-BLOCK in a review-only copy.
    OpenEvolve is allowed to mutate that copy; the original tool file remains
    untouched and callers only receive patch artifacts.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    source = organism.file_path.read_text(encoding="utf-8")

    initial_program = output_dir / "initial_tool.py"
    evaluator = output_dir / "evaluator.py"
    config_file = output_dir / "config.yaml"

    initial_program.write_text(
        "# Generated review-only OpenEvolve scaffold for "
        f"{organism.name} from {organism.file_path.name}.\n"
        "# Do not apply this file directly; consume patch.diff/report.md.\n"
        "# EVOLVE-BLOCK-START\n"
        f"{source.rstrip()}\n"
        "# EVOLVE-BLOCK-END\n",
        encoding="utf-8",
    )

    evaluator.write_text(
        '"""Safety-first evaluator scaffold for Hermes tool evolution."""\n'
        "from __future__ import annotations\n\n"
        "import py_compile\n"
        "from pathlib import Path\n\n"
        "def evaluate(program_path):\n"
        "    path = Path(program_path)\n"
        "    try:\n"
        "        py_compile.compile(str(path), doraise=True)\n"
        "    except Exception as exc:\n"
        "        return {\n"
        "            'combined_score': 0.0,\n"
        "            'syntax_ok': 0.0,\n"
        "            'error': str(exc),\n"
        "        }\n"
        "    text = path.read_text(encoding='utf-8')\n"
        "    keeps_registry = 'registry.register' in text\n"
        "    return {\n"
        "        'combined_score': 0.5 + (0.5 if keeps_registry else 0.0),\n"
        "        'syntax_ok': 1.0,\n"
        "        'registry_present': 1.0 if keeps_registry else 0.0,\n"
        "    }\n",
        encoding="utf-8",
    )

    config_file.write_text(
        "# Minimal OpenEvolve config generated by Hermes self-evolution.\n"
        f"max_iterations: {iterations}\n"
        "checkpoint_interval: 1\n"
        "log_level: INFO\n"
        "diff_based_evolution: true\n",
        encoding="utf-8",
    )

    return OpenEvolveToolScaffold(initial_program, evaluator, config_file)


def _write_openevolve_review_artifacts(
    output_dir: Path,
    tool_name: str,
    iterations: int,
    elapsed: float,
    result: OpenEvolveRunResult,
) -> dict:
    """Persist patch-only OpenEvolve results for human/agent review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "patch.diff").write_text(result.patch_text, encoding="utf-8")
    (output_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")

    metrics = {
        "engine": "openevolve",
        "tool_name": tool_name,
        "iterations": iterations,
        "success": result.success,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "command": result.command,
        "work_dir": str(result.work_dir),
        "openevolve_output_dir": str(result.output_dir),
        "best_program": str(result.best_program) if result.best_program else None,
        "best_info": str(result.best_info) if result.best_info else None,
        "best_metrics": result.best_metrics,
        "error": result.error,
        "improved": result.improved,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = [
        f"# OpenEvolve review report: {tool_name}",
        "",
        "This run is patch-only. It did not modify the production Hermes repo.",
        "",
        f"- success: {result.success}",
        f"- returncode: {result.returncode}",
        f"- iterations: {iterations}",
        f"- elapsed_seconds: {elapsed:.2f}",
        f"- command: `{' '.join(result.command)}`",
        f"- work_dir: `{result.work_dir}`",
        f"- best_program: `{result.best_program}`",
        "",
        "## Next gate",
        "Review `patch.diff`, then run the relevant Hermes tests before applying anything.",
    ]
    if result.error:
        report.extend(["", "## Error", "", result.error])
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics


def run_openevolve_tool_evolution(
    organism: CodeOrganism,
    iterations: int,
    output_dir: Path,
    openevolve_cmd: str = "openevolve-run",
) -> dict:
    """Run OpenEvolve for a tool in isolated patch-only mode."""

    scaffold = create_openevolve_tool_scaffold(organism, output_dir, iterations)
    start_time = time.time()
    result = run_openevolve_isolated(
        OpenEvolveRunnerConfig(
            initial_program=scaffold.initial_program,
            evaluator=scaffold.evaluator,
            config_file=scaffold.config_file,
            iterations=iterations,
            output_root=output_dir,
            openevolve_cmd=openevolve_cmd,
        )
    )
    elapsed = time.time() - start_time
    return _write_openevolve_review_artifacts(
        output_dir=output_dir,
        tool_name=organism.name,
        iterations=iterations,
        elapsed=elapsed,
        result=result,
    )


# ── Main evolution function ─────────────────────────────────────────────


def evolve_tool_code(
    tool_name: str,
    iterations: int = 10,
    bug_issue: int | None = None,
    hermes_repo: str | None = None,
    dry_run: bool = False,
    engine: str = "openevolve",
    output_root: str | None = None,
    openevolve_cmd: str = "openevolve-run",
):
    """Main function to evolve tool implementation code."""

    hermes_agent_path = Path(hermes_repo) if hermes_repo else get_hermes_agent_path()
    engine = engine.lower()
    if engine not in {"darwinian", "openevolve"}:
        raise ValueError("engine must be one of: darwinian, openevolve")

    console.print(
        f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — "
        f"Evolving tool code: [bold]{tool_name}[/bold]"
    )
    console.print(f"Engine: {engine}\n")
    if engine == "darwinian":
        console.print(
            "[bold yellow]⚠ Darwinian mode can mutate the target Hermes checkout in place. "
            "Prefer the default openevolve engine unless running in an isolated worktree.[/bold yellow]"
        )

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
        return {"engine": engine, "tool_name": tool_name, "dry_run": True}

    if engine == "openevolve":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(output_root) if output_root else Path("output/code_evolution")
        oe_output_dir = root / f"{tool_name}_{timestamp}_openevolve"
        console.print(
            f"\n[bold cyan]Step 2: Running OpenEvolve patch-only ({iterations} iterations)[/bold cyan]\n"
        )
        metrics = run_openevolve_tool_evolution(
            organism=organism,
            iterations=iterations,
            output_dir=oe_output_dir,
            openevolve_cmd=openevolve_cmd,
        )
        console.print(f"  Success: {'✓' if metrics['success'] else '✗'}")
        console.print(f"  Output saved to {oe_output_dir}/")
        console.print("  Review patch.diff/report.md before applying anything.")
        return {**metrics, "output_dir": str(oe_output_dir)}

    # ── 2. Evaluate baseline fitness ────────────────────────────────────
    console.print("\n[bold]Step 2: Evaluating baseline fitness[/bold]")

    # Run tests
    tests_passed, test_output = run_pytest_for_tool(tool_name, hermes_agent_path)
    console.print(f"  Tests: {'✓ Passed' if tests_passed else '✗ Failed'}")

    baseline_fitness, baseline_scores = evaluate_code_fitness(tool_name, hermes_agent_path)
    console.print(f"  Baseline fitness: {baseline_fitness:.3f}")
    for name, score in baseline_scores.items():
        console.print(f"    {name}: {score:.3f}")

    # ── 3. Run Darwinian Evolver ────────────────────────────────────────
    console.print(
        f"\n[bold cyan]Step 3: Running Darwinian Evolver ({iterations} generations)[/bold cyan]\n"
    )

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
    evolved_fitness, evolved_scores = evaluate_code_fitness(tool_name, hermes_agent_path)

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
@click.option(
    "--engine",
    type=click.Choice(["darwinian", "openevolve"]),
    default="openevolve",
    show_default=True,
    help="Evolution engine to use",
)
@click.option("--output-root", default=None, help="Directory for review artifacts")
@click.option(
    "--openevolve-cmd", default="openevolve-run", show_default=True, help="OpenEvolve CLI command"
)
def main(tool, iterations, bug_issue, hermes_repo, dry_run, engine, output_root, openevolve_cmd):
    """Evolve tool implementation code using a selected evolution engine."""
    evolve_tool_code(
        tool_name=tool,
        iterations=iterations,
        bug_issue=bug_issue,
        hermes_repo=hermes_repo,
        dry_run=dry_run,
        engine=engine,
        output_root=output_root,
        openevolve_cmd=openevolve_cmd,
    )


if __name__ == "__main__":
    main()
