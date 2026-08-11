"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import json
import time
from datetime import datetime
from pathlib import Path

import click
import dspy
from rich.console import Console
from rich.table import Table

from benchmarks.capability.suite import load_suite as load_capability_suite
from evolution.core.capability_feedback import (
    CapabilityFeedbackError,
    CapabilityFeedbackPolicy,
    load_optimizer_feedback,
)
from evolution.core.config import (
    EvolutionConfig,
    make_dashscope_lm,
    make_lm,
    resolve_hermes_agent_path,
)
from evolution.core.constraints import ConstraintValidator
from evolution.core.cost_tracker import set_budget_from_option
from evolution.core.dataset_builder import EvalDataset, GoldenDatasetLoader, SyntheticDatasetBuilder
from evolution.core.errors import BudgetExceededError, EvolutionError
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import (
    LLMJudge,
    make_llm_judge_metric,
)
from evolution.skills.skill_module import (
    SkillModule,
    extract_evolved_skill_text,
    find_skill,
    load_skill,
    reassemble_skill,
)

console = Console()


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: str | None = None,
    optimizer_model: str = "qwen3.6-plus",
    eval_model: str = "qwen3.6-plus",
    hermes_repo: str | None = None,
    run_tests: bool = False,
    dry_run: bool = False,
    max_cost_usd: float | None = None,
    create_pr: bool = False,
    pr_dry_run: bool = False,
    capability_feedback: str | Path | None = None,
    capability_suite: str | Path | None = None,
) -> dict | None:
    """Main evolution function — orchestrates the full optimization loop.

    ``max_cost_usd`` sets a hard USD budget on the global cost tracker for
    this process; ``None`` keeps the EVOLUTION_MAX_COST_USD env default.

    ``capability_feedback`` optionally points at a development-only optimizer
    feedback JSON produced by ``python -m benchmarks.capability compare
    --optimizer-feedback``. ``capability_suite`` is required with it and binds
    the feedback to the trusted suite ID, hash, development task set, and
    critical-task policy before any billable work. A full holdout-aware
    comparison document is refused; only the validated development section is
    printed and recorded in ``metrics.json``.

    ``create_pr``/``pr_dry_run`` are strictly opt-in: by default no PR-related
    git operations happen at all. ``pr_dry_run`` renders the redacted PR
    preview without touching git; ``create_pr`` invokes PRBuilder (which
    itself enforces clean-worktree and redaction rules). Both are refused for
    runs that failed gates or showed no positive holdout improvement.

    Returns the run's metrics dict (including ``improvement``, ``deployable``
    and ``output_dir``) so programmatic callers such as Phase 5 can consume
    results directly instead of scavenging output directories. Failed runs
    return a metrics dict with ``deployable: False`` and an ``error``;
    ``dry_run`` returns None.
    """

    # ── 0. Optional capability harness feedback ─────────────────────────
    # Validated fail-closed before any billable work. Only the development-
    # only optimizer_feedback document is accepted; a full Comparison
    # payload (which carries holdout outcomes) raises CapabilityFeedbackError.
    capability_context = None
    if (capability_feedback is None) != (capability_suite is None):
        raise CapabilityFeedbackError(
            "optimizer feedback rejected: --capability-feedback and "
            "--capability-suite must be supplied together (fail closed)"
        )
    if capability_feedback is not None and capability_suite is not None:
        trusted_suite = None
        try:
            trusted_suite = load_capability_suite(capability_suite)
        except Exception:  # noqa: BLE001 - convert suite backend failures to typed errors
            raise CapabilityFeedbackError(
                "optimizer feedback rejected: trusted capability suite could not be "
                "loaded and validated (fail closed)"
            ) from None
        try:
            development_ids = frozenset(trusted_suite.development_task_ids)
            critical_development_ids = frozenset(
                task.task_id
                for task in trusted_suite.tasks
                if task.split == "development" and task.critical
            )
            capability_context = load_optimizer_feedback(
                capability_feedback,
                policy=CapabilityFeedbackPolicy(
                    suite_id=trusted_suite.suite_id,
                    suite_hash=trusted_suite.suite_hash,
                    development_task_ids=development_ids,
                    critical_development_task_ids=critical_development_ids,
                ),
            )
        finally:
            trusted_suite.close()
    capability_document = (
        capability_context.to_document() if capability_context is not None else None
    )

    hermes_agent_path = resolve_hermes_agent_path(hermes_repo)
    config = EvolutionConfig(
        hermes_agent_path=hermes_agent_path,
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
        max_cost_usd=max_cost_usd,
    )

    # Apply the budget before any billable work — dataset generation already
    # makes LLM calls, so this must precede everything, not just the optimizer.
    set_budget_from_option(config.max_cost_usd)

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(
        f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n"
    )

    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        raise EvolutionError(
            f"Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}"
        )

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path.relative_to(config.hermes_agent_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    # Configure DSPy EARLY — must be before any DSPy modules are used.
    # DashScope requires ChatAdapter (not JSONAdapter) because it needs 'json'
    # in the prompt to use response_format=json_object.
    from dspy.adapters import ChatAdapter

    lm = make_lm(eval_model, num_retries=8)
    dspy.configure(lm=lm, adapter=ChatAdapter())
    console.print(f"  DSPy configured: {eval_model} (ChatAdapter)")

    if capability_context is not None:
        console.print("\n[bold]Capability harness feedback (development slice only)[/bold]")
        for line in capability_context.prompt_section().splitlines():
            console.print(f"  {line}")

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        if capability_context is not None:
            console.print("  Capability feedback validated (development-only document)")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print("  Would validate constraints and create PR")
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "sessiondb":
        save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=["claude-code", "copilot", "hermes"],
            output_path=save_path,
            model=eval_model,
        )
        if not dataset.all_examples:
            raise EvolutionError("No relevant examples found from session history")
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(
            artifact_text=skill["raw"],
            artifact_type="skill",
        )
        # Save for reuse
        save_path = Path("datasets") / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        raise EvolutionError("Specify --dataset-path or use --eval-source synthetic")

    console.print(
        f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout"
    )

    # ── 3. Validate constraints on baseline ─────────────────────────────
    console.print("\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(skill["raw"], "skill")
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print(
            "[yellow]⚠ Baseline skill has constraint violations — proceeding anyway[/yellow]"
        )

    # ── 4. Set up DSPy + GEPA optimizer ─────────────────────────────────
    console.print("\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")
    if optimizer_model == eval_model:
        console.print(
            "[yellow]⚠ Optimizer and evaluator use the same model; treat scores as proxy "
            "signals until validated by a real Hermes batch_runner/judge split.[/yellow]"
        )

    # DSPy was already configured at step 1 — no need to re-configure
    console.print("  Using existing DSPy config from Step 1")

    # Create the baseline skill module
    baseline_module = SkillModule(skill["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(
        f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n"
    )

    start_time = time.time()

    try:
        # Create reflection LM for GEPA
        reflection_lm = make_dashscope_lm(optimizer_model, num_retries=8, temperature=1.0)

        # Use LLM-as-judge metric for meaningful fitness signal
        judge_metric = make_llm_judge_metric(config, skill["body"])

        optimizer = dspy.GEPA(
            metric=judge_metric,
            max_metric_calls=iterations * 5,
            reflection_lm=reflection_lm,
        )

        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )
    except BudgetExceededError:
        # A budget abort is not "GEPA unavailable" — falling back to MIPROv2
        # would start a second optimizer run past the hard budget.
        raise
    except Exception as e:
        # Fall back to MIPROv2 if GEPA isn't available in this DSPy version
        console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
        # MIPROv2 uses 'auto' to control budget: light(~10), medium(~50), heavy(~200)
        auto_budget = "light" if iterations <= 10 else ("medium" if iterations <= 50 else "heavy")

        # Use LLM-as-judge metric for meaningful fitness signal
        judge_metric = make_llm_judge_metric(config, skill["body"])

        optimizer = dspy.MIPROv2(
            metric=judge_metric,
            auto=auto_budget,
            num_threads=1,
        )
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved skill text ───────────────────────────────────
    # Use sentinel-based extraction to avoid the --- separator bug
    try:
        evolved_body = extract_evolved_skill_text(optimized_module)
    except ValueError as e:
        output_path = Path("output") / skill_name / "extraction_FAILED.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(e), encoding="utf-8")
        console.print(f"[red]✗ Could not extract evolved skill text: {e}[/red]")
        console.print(f"  Saved extraction error to {output_path}")
        return _failed_run_metrics(
            skill_name,
            str(output_path.parent),
            f"extraction failed: {e}",
            capability_feedback=capability_document,
        )
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 7. Validate evolved skill ───────────────────────────────────────
    console.print("\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validator.validate_all(evolved_full, "skill", baseline_text=skill["raw"])
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED constraints — not deploying[/red]")
        # Still save for inspection
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {output_path}")
        return _failed_run_metrics(
            skill_name,
            str(output_path.parent),
            "evolved skill failed constraint gates",
            capability_feedback=capability_document,
        )

    if config.run_pytest:
        console.print(
            "\n[bold]Running hermes-agent pytest gate "
            "(evolved skill applied in temp workspace)[/bold]"
        )
        test_result = validator.run_test_suite(
            hermes_agent_path,
            artifact_relpath=skill_path.relative_to(config.hermes_agent_path),
            artifact_text=evolved_full,
        )
        icon = "✓" if test_result.passed else "✗"
        color = "green" if test_result.passed else "red"
        console.print(
            f"  [{color}]{icon} {test_result.constraint_name}[/{color}]: {test_result.message}"
        )
        if test_result.details:
            console.print(f"    {test_result.details}")
        if not test_result.passed:
            console.print("[red]✗ Pytest gate failed — not deploying[/red]")
            output_path = Path("output") / skill_name / "evolved_TESTS_FAILED.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(evolved_full, encoding="utf-8")
            console.print(f"  Saved failed variant to {output_path}")
            return _failed_run_metrics(
                skill_name,
                str(output_path.parent),
                "evolved skill failed pytest gate",
                capability_feedback=capability_document,
            )

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")
    console.print("  Using LLM-as-judge (correctness + procedure + conciseness)")

    holdout_examples = dataset.to_dspy_examples("holdout")
    holdout_judge = LLMJudge(config)

    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        # Score baseline and evolved using LLM-as-judge
        baseline_pred = baseline_module(task_input=ex.task_input)
        baseline_score = holdout_judge.score(
            task_input=ex.task_input,
            expected_behavior=ex.expected_behavior,
            agent_output=getattr(baseline_pred, "output", ""),
            skill_text=skill["body"],
        )
        baseline_scores.append(baseline_score.composite)

        evolved_pred = optimized_module(task_input=ex.task_input)
        evolved_score = holdout_judge.score(
            task_input=ex.task_input,
            expected_behavior=ex.expected_behavior,
            agent_output=getattr(evolved_pred, "output", ""),
            skill_text=evolved_body,  # Use evolved skill text for evolved scoring
        )
        evolved_scores.append(evolved_score.composite)

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 9. Report results ───────────────────────────────────────────────
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row(
        "Skill Size",
        f"{len(skill['body']):,} chars",
        f"{len(evolved_body):,} chars",
        f"{len(evolved_body) - len(skill['body']):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    # ── 9b. Cost summary ────────────────────────────────────────────────
    from evolution.core.cost_tracker import tracker

    tracker.print_summary(console)

    # ── 10. Save output ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evolved skill
    (output_dir / "evolved_skill.md").write_text(evolved_full)

    # Save baseline for comparison
    (output_dir / "baseline_skill.md").write_text(skill["raw"])

    # Save metrics
    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
        "deployable": all_pass,
        "output_dir": str(output_dir),
    }
    if capability_document is not None:
        # Only the re-validated development-only view; never a full comparison.
        metrics["capability_feedback"] = capability_document
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if improvement > 0:
        console.print(
            f"\n[bold green]✓ Holdout proxy score improved by {improvement:+.3f} "
            f"({improvement/max(0.001, avg_baseline)*100:+.1f}%) on "
            f"{len(holdout_examples)} held-out examples[/bold green]"
        )
        console.print(
            "  Local proxy signal only — not validated production improvement; "
            "requires human review."
        )
        console.print(
            f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md"
        )
    else:
        console.print(
            f"\n[yellow]⚠ Evolution did not improve the holdout proxy score "
            f"(change: {improvement:+.3f})[/yellow]"
        )
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")

    # ── 11. Optional PR step (strictly opt-in; no-op by default) ────────
    if create_pr or pr_dry_run:
        pr_info = _handle_pr_request(
            create_pr=create_pr,
            pr_dry_run=pr_dry_run,
            hermes_agent_path=config.hermes_agent_path,
            skill_relpath=str(skill_path.relative_to(config.hermes_agent_path)),
            baseline_text=skill["raw"],
            evolved_text=evolved_full,
            run_metrics=metrics,
        )
        preview = pr_info.pop("preview", None)
        if preview:
            preview_path = output_dir / "pr_preview.md"
            preview_path.write_text(preview, encoding="utf-8")
            pr_info["preview_path"] = str(preview_path)
            console.print(f"  PR preview saved to {preview_path}")
        metrics["pr"] = pr_info
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def _handle_pr_request(
    *,
    create_pr: bool,
    pr_dry_run: bool,
    hermes_agent_path: Path,
    skill_relpath: str,
    baseline_text: str,
    evolved_text: str,
    run_metrics: dict,
) -> dict:
    """Opt-in PR step for Phase 1. Returns a JSON-serializable status dict.

    Refuses to build a PR for non-deployable runs or runs without a positive
    holdout proxy improvement — a PR must never be a way around the gates.
    ``pr_dry_run`` wins over ``create_pr``: it returns the redacted preview
    under the ``preview`` key without any git/GitHub side effects. Gate,
    preview, and PRBuilder semantics live in the shared opt-in helper; only
    the change/metrics construction is Phase-1-specific (the skill file is
    replaced wholesale rather than snippet-patched).
    """
    from evolution.core.pr_optin import handle_opt_in_pr

    def build_changes():
        from evolution.core.pr_builder import PRChange

        return [
            PRChange(
                file_path=skill_relpath,
                original_content=baseline_text,
                evolved_content=evolved_text,
                change_type="skill",
            )
        ], None

    def build_pr_metrics():
        from evolution.core.cost_tracker import tracker
        from evolution.core.pr_builder import PRMetrics

        baseline_score = run_metrics["baseline_score"]
        improvement = run_metrics["improvement"]
        return PRMetrics(
            baseline_score=baseline_score,
            evolved_score=run_metrics["evolved_score"],
            holdout_score=run_metrics["evolved_score"],
            improvement=improvement,
            improvement_pct=improvement / max(0.001, baseline_score) * 100,
            iterations=run_metrics["iterations"],
            optimizer=f"GEPA ({run_metrics['optimizer_model']})",
            eval_dataset_size=(
                run_metrics["train_examples"]
                + run_metrics["val_examples"]
                + run_metrics["holdout_examples"]
            ),
            train_examples=run_metrics["train_examples"],
            val_examples=run_metrics["val_examples"],
            holdout_examples=run_metrics["holdout_examples"],
            elapsed_seconds=run_metrics["elapsed_seconds"],
            cost_estimate=f"~${tracker.total_cost_usd:.2f} (estimated)",
        )

    return handle_opt_in_pr(
        create_pr=create_pr,
        pr_dry_run=pr_dry_run,
        hermes_agent_path=hermes_agent_path,
        run_metrics=run_metrics,
        build_changes=build_changes,
        pr_metrics=build_pr_metrics,
        no_improvement_reason="no positive holdout proxy improvement",
    )


def _failed_run_metrics(
    skill_name: str,
    output_dir: str,
    error: str,
    *,
    capability_feedback: dict | None = None,
) -> dict:
    """Persist metrics for a paid run that failed a gate, including provenance."""
    metrics = {
        "skill_name": skill_name,
        "improvement": 0.0,
        "deployable": False,
        "output_dir": output_dir,
        "error": error,
    }
    if capability_feedback is not None:
        metrics["capability_feedback"] = capability_feedback
    metrics_dir = Path(output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option(
    "--eval-source",
    default="synthetic",
    type=click.Choice(["synthetic", "golden", "sessiondb"]),
    help="Source for evaluation dataset",
)
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="qwen3.6-plus", help="Model for GEPA reflections")
@click.option("--eval-model", default="qwen3.6-plus", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option(
    "--max-cost-usd",
    default=None,
    type=click.FloatRange(min=0, min_open=True),
    help="Hard USD budget for LLM API cost; the run aborts once estimated spend "
    "exceeds it (overrides EVOLUTION_MAX_COST_USD)",
)
@click.option(
    "--create-pr",
    is_flag=True,
    help="After a deployable improvement, create a PR against hermes-agent via "
    "PRBuilder (requires clean worktree; never happens by default)",
)
@click.option(
    "--pr-dry-run",
    is_flag=True,
    help="Render the redacted PR title/body/diff without any git or GitHub "
    "side effects (takes precedence over --create-pr)",
)
@click.option(
    "--capability-feedback",
    default=None,
    help="Path to a development-only optimizer feedback JSON produced by "
    "'python -m benchmarks.capability compare --optimizer-feedback'; requires "
    "--capability-suite and is validated fail-closed",
)
@click.option(
    "--capability-suite",
    default=None,
    help="Trusted capability suite JSON used to bind feedback to its suite ID, "
    "hash, development task set, and critical-task policy",
)
def main(
    skill,
    iterations,
    eval_source,
    dataset_path,
    optimizer_model,
    eval_model,
    hermes_repo,
    run_tests,
    dry_run,
    max_cost_usd,
    create_pr,
    pr_dry_run,
    capability_feedback,
    capability_suite,
):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    try:
        evolve(
            skill_name=skill,
            iterations=iterations,
            eval_source=eval_source,
            dataset_path=dataset_path,
            optimizer_model=optimizer_model,
            eval_model=eval_model,
            hermes_repo=hermes_repo,
            run_tests=run_tests,
            dry_run=dry_run,
            max_cost_usd=max_cost_usd,
            create_pr=create_pr,
            pr_dry_run=pr_dry_run,
            capability_feedback=capability_feedback,
            capability_suite=capability_suite,
        )
    except EvolutionError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
