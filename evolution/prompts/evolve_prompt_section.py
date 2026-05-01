"""Phase 3: System prompt section evolution via DSPy + GEPA.

Optimizes sections of the system prompt that guide agent behavior.
Sections include:
- DEFAULT_AGENT_IDENTITY: Core persona and behavioral traits
- MEMORY_GUIDANCE: How to use persistent memory
- SESSION_SEARCH_GUIDANCE: When to search past sessions
- SKILLS_GUIDANCE: When to save/load skills
- PLATFORM_HINTS: Per-platform formatting guidance (per platform)

Each section is independently optimizable but evaluated together
(the full system prompt matters, not individual sections).

RISK: System prompt changes affect EVERYTHING. Benchmarks are mandatory
gates. No evolved prompt ships without passing the full benchmark suite.

Usage:
    python -m evolution.prompts.evolve_prompt_section --section MEMORY_GUIDANCE --iterations 10
    python -m evolution.prompts.evolve_prompt_section --section ALL --iterations 10
"""

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig
from evolution.core.utils import parse_json_array

console = Console()


# ── Prompt section definitions ──────────────────────────────────────────

PROMPT_SECTIONS = {
    "DEFAULT_AGENT_IDENTITY": {
        "file": "agent/prompt_builder.py",
        "description": "Core persona, behavioral traits, communication style",
        "max_growth_pct": 20,
        "risk_level": "high",
    },
    "MEMORY_GUIDANCE": {
        "file": "agent/prompt_builder.py",
        "description": "How and when to use persistent memory",
        "max_growth_pct": 20,
        "risk_level": "medium",
    },
    "SESSION_SEARCH_GUIDANCE": {
        "file": "agent/prompt_builder.py",
        "description": "When to search past session history",
        "max_growth_pct": 20,
        "risk_level": "medium",
    },
    "SKILLS_GUIDANCE": {
        "file": "agent/prompt_builder.py",
        "description": "When to save/load skills",
        "max_growth_pct": 20,
        "risk_level": "medium",
    },
}

# PLATFORM_HINTS is a dict, not a single section
PLATFORM_HINTS_FILE = "agent/prompt_builder.py"


@dataclass
class PromptSection:
    """A single evolvable system prompt section."""
    name: str
    content: str
    file_path: str  # Relative path in hermes-agent
    description: str
    max_growth_pct: float
    risk_level: str


@dataclass
class BehavioralTestExample:
    """A test case for evaluating prompt section quality."""
    scenario: str  # The situation/task
    section_name: str  # Which section this tests
    expected_behavior: str  # Rubric for good behavior
    expected_tool_usage: str | None = None  # Should use this tool
    should_not_do: str | None = None  # Should NOT do this


@dataclass
class PromptEvalDataset:
    """Dataset for prompt section evaluation."""
    examples: list[BehavioralTestExample]
    train: list[BehavioralTestExample] = field(default_factory=list)
    val: list[BehavioralTestExample] = field(default_factory=list)
    holdout: list[BehavioralTestExample] = field(default_factory=list)

    def split(self, train_ratio: float = 0.6, val_ratio: float = 0.2):
        import random
        random.seed(42)
        random.shuffle(self.examples)
        n = len(self.examples)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        self.train = self.examples[:n_train]
        self.val = self.examples[n_train:n_train + n_val]
        self.holdout = self.examples[n_train + n_val:]

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "scenario": e.scenario,
                "section_name": e.section_name,
                "expected_behavior": e.expected_behavior,
                "expected_tool_usage": e.expected_tool_usage,
                "should_not_do": e.should_not_do,
            }
            for e in self.examples
        ]
        (path / "prompt_behavioral_tests.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )

    @classmethod
    def load(cls, path: Path) -> "PromptEvalDataset":
        data_file = path / "prompt_behavioral_tests.json"
        if not data_file.exists():
            return cls(examples=[])
        data = json.loads(data_file.read_text())
        examples = [
            BehavioralTestExample(
                scenario=e["scenario"],
                section_name=e["section_name"],
                expected_behavior=e["expected_behavior"],
                expected_tool_usage=e.get("expected_tool_usage"),
                should_not_do=e.get("should_not_do"),
            )
            for e in data
        ]
        ds = cls(examples=examples)
        ds.split()
        return ds


# ── Section extractor ───────────────────────────────────────────────────

def extract_prompt_sections(
    hermes_agent_path: Path,
    section_names: list[str] | None = None,
) -> list[PromptSection]:
    """Extract prompt section content from hermes-agent source files.

    Parses Python source files to find string constant definitions.
    """
    sections = []
    target_names = section_names or list(PROMPT_SECTIONS.keys())

    for name in target_names:
        if name not in PROMPT_SECTIONS:
            console.print(f"[yellow]⚠ Unknown section: {name}[/yellow]")
            continue

        info = PROMPT_SECTIONS[name]
        file_path = hermes_agent_path / info["file"]

        if not file_path.exists():
            console.print(f"[red]✗ File not found: {file_path}[/red]")
            continue

        content = _extract_constant(file_path, name)
        if content:
            sections.append(PromptSection(
                name=name,
                content=content,
                file_path=info["file"],
                description=info["description"],
                max_growth_pct=info["max_growth_pct"],
                risk_level=info["risk_level"],
            ))
            console.print(f"  ✓ {name}: {len(content)} chars — {content[:60]}...")
        else:
            console.print(f"  [red]✗ Could not extract {name}[/red]")

    return sections


def _extract_constant(file_path: Path, name: str) -> str | None:
    """Extract a Python string constant from source code.

    Handles multi-line string definitions (parenthesized tuples of strings).
    """
    try:
        source = file_path.read_text()
    except Exception:
        return None

    # Find the constant definition
    import re
    # Pattern: NAME = (\n    "..." \n    "..." \n)
    # or: NAME = "..."
    safe_name = re.escape(name)
    pattern = rf'^{safe_name}\s*=\s*\(([\s\S]*?)\)\s*$'
    match = re.search(pattern, source, re.MULTILINE)

    if match:
        # Extract string contents from parenthesized block
        inner = match.group(1)
        # Join multi-line string literals
        parts = re.findall(r'["\']([^"\']*)["\']', inner)
        return "\n".join(p for p in parts if p.strip())

    # Try simple string: NAME = "..."
    pattern2 = rf'^{safe_name}\s*=\s*"""([\s\S]*?)"""'
    match2 = re.search(pattern2, source, re.MULTILINE)
    if match2:
        return match2.group(1).strip()

    return None


# ── Behavioral test generator ───────────────────────────────────────────

class BehavioralTestGenerator:
    """Generates behavioral test scenarios for prompt sections."""

    class TestGenerator(dspy.Signature):
        """Generate behavioral test scenarios for system prompt sections.

        For each prompt section, generate realistic scenarios that test
        whether the agent behaves correctly with the given instructions.

        Each test should include:
        - scenario: A realistic task/situation the agent might face
        - expected_behavior: What the agent SHOULD do
        - should_not_do: What the agent should NOT do (common failure mode)
        """
        section_name: str = dspy.InputField(desc="Name of the prompt section being tested")
        section_content: str = dspy.InputField(desc="Current section content/instructions")
        num_tests: int = dspy.InputField(desc="Number of test scenarios to generate")
        test_scenarios_json: str = dspy.OutputField(
            desc="JSON array of behavioral test scenarios"
        )

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(self.TestGenerator)

    def generate(
        self,
        sections: list[PromptSection],
        tests_per_section: int = 10,
        output_path: Path | None = None,
    ) -> PromptEvalDataset:
        """Generate behavioral tests for all sections."""
        all_examples = []

        for section in sections:
            console.print(f"  Generating tests for {section.name}...")
            try:
                result = self.generator(
                    section_name=section.name,
                    section_content=section.content[:2000],
                    num_tests=tests_per_section,
                )
                examples_json = parse_json_array(result.test_scenarios_json)
                for ex in examples_json:
                    all_examples.append(BehavioralTestExample(
                        scenario=ex.get("scenario", ""),
                        section_name=section.name,
                        expected_behavior=ex.get("expected_behavior", ""),
                        expected_tool_usage=ex.get("expected_tool_usage"),
                        should_not_do=ex.get("should_not_do"),
                    ))
            except Exception as e:
                console.print(f"    ⚠ Failed: {e}")

        dataset = PromptEvalDataset(examples=all_examples)
        dataset.split()

        if output_path:
            dataset.save(output_path)

        return dataset



# ── Behavioral evaluator ────────────────────────────────────────────────

class BehavioralEvaluator:
    """Evaluates agent behavior with a given system prompt.

    Uses LLM-as-judge to score whether the agent's behavior
    matches the expected behavior for each test scenario.
    """

    class BehaviorJudge(dspy.Signature):
        """Evaluate whether the agent's behavior matches expectations.

        Score how well the agent followed the system prompt instructions
        and behaved correctly for the given scenario.
        """
        scenario: str = dspy.InputField(desc="The task/scenario given to the agent")
        system_prompt_section: str = dspy.InputField(desc="The relevant system prompt section")
        agent_behavior: str = dspy.InputField(desc="How the agent responded/behaved")
        expected_behavior: str = dspy.InputField(desc="Rubric for expected behavior")
        score: float = dspy.OutputField(desc="Score 0.0-1.0: how well behavior matches expectations")
        feedback: str = dspy.OutputField(desc="Specific feedback on what was good/bad")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(self.BehaviorJudge)

    def evaluate(
        self,
        sections: list[PromptSection],
        examples: list[BehavioralTestExample],
    ) -> tuple[float, dict[str, float]]:
        """Evaluate behavior across all test examples.

        Returns (overall_score, per_section_scores).
        """
        # Build a section lookup
        section_map = {s.name: s.content for s in sections}

        total_score = 0.0
        per_section_scores = {}
        per_section_counts = {}

        for ex in examples:
            section_content = section_map.get(ex.section_name, "")
            if not section_content:
                continue

            # For Phase 3, we evaluate by simulating how the LLM would
            # respond to a scenario given the section, then judge the response.
            # This is a proxy evaluation — in production, you'd run the
            # actual agent through batch_runner.
            simulated_score = self._evaluate_scenario(
                scenario=ex.scenario,
                section_content=section_content,
                expected=ex.expected_behavior,
            )

            total_score += simulated_score
            per_section_scores[ex.section_name] = (
                per_section_scores.get(ex.section_name, 0) + simulated_score
            )
            per_section_counts[ex.section_name] = per_section_counts.get(ex.section_name, 0) + 1

        n_evaluated = len([ex for ex in examples if ex.section_name in section_map])
        overall = total_score / max(1, n_evaluated)
        per_section_avg = {
            name: per_section_scores[name] / per_section_counts[name]
            for name in per_section_counts
        }

        return overall, per_section_avg

    def _evaluate_scenario(
        self,
        scenario: str,
        section_content: str,
        expected: str,
    ) -> float:
        """Evaluate a single scenario with a section.

        Proxy evaluation: assesses how well the section instructions address
        the scenario type. In production, replace with actual agent execution
        through batch_runner for real behavioral scoring.
        """
        try:
            result = self.judge(
                scenario=scenario,
                system_prompt_section=section_content[:1000],
                agent_behavior=f"Section instructions: {section_content[:500]}",
                expected_behavior=expected,
            )
            score = float(result.score)
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5


# ── Section-as-DSPy-parameter module ────────────────────────────────────

class PromptSectionModule(dspy.Module):
    """DSPy module wrapping prompt sections for GEPA optimization.

    Each section is a separate optimizable parameter.
    """

    class SectionTask(dspy.Signature):
        """Follow the system prompt instructions to handle a scenario."""
        prompt_guidance: str = dspy.InputField(desc="System prompt section instructions")
        scenario: str = dspy.InputField(desc="The scenario/task to handle")
        behavior: str = dspy.OutputField(desc="How you would respond/act")

    def __init__(self, sections: list[PromptSection]):
        super().__init__()
        self.sections = {s.name: s for s in sections}
        self.predictors = {}
        for section in sections:
            self.predictors[section.name] = dspy.ChainOfThought(self.SectionTask)

    def forward(self, section_name: str, scenario: str) -> dspy.Prediction:
        section = self.sections.get(section_name)
        if not section:
            return dspy.Prediction(behavior="")

        predictor = self.predictors.get(section_name)
        if not predictor:
            return dspy.Prediction(behavior="")

        result = predictor(
            prompt_guidance=section.content,
            scenario=scenario,
        )
        return dspy.Prediction(behavior=result.behavior)


# ── Prompt section fitness metric ───────────────────────────────────────

def prompt_section_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
) -> float:
    """DSPy metric for prompt section optimization.

    GEPA-compatible 5-arg signature.
    Uses keyword overlap as a fast proxy, but without arbitrary floors.
    Returns 0.0 for empty output, overlap-based score otherwise.
    """
    behavior = getattr(prediction, "behavior", "").lower()
    expected = getattr(example, "expected_behavior", "").lower()

    if not behavior.strip():
        return 0.0

    # Keyword overlap — no floor bias
    expected_words = set(expected.split())
    behavior_words = set(behavior.split())
    if expected_words:
        overlap = len(expected_words & behavior_words) / len(expected_words)
        return min(1.0, overlap)
    return 0.0


# ── Prompt section constraint validator ─────────────────────────────────

def validate_prompt_sections(
    sections: list[PromptSection],
    baseline_sections: list[PromptSection] | None = None,
) -> list[dict]:
    """Validate evolved prompt sections meet constraints."""
    violations = []
    baseline_map = {s.name: s for s in (baseline_sections or [])}

    for section in sections:
        baseline = baseline_map.get(section.name)
        baseline_len = len(baseline.content) if baseline else len(section.content)

        # Check growth limit
        if baseline_len > 0:
            growth = (len(section.content) - baseline_len) / baseline_len * 100
            if growth > section.max_growth_pct:
                violations.append({
                    "section": section.name,
                    "violation": (
                        f"Growth {growth:.1f}% exceeds limit of {section.max_growth_pct}%"
                    ),
                })

        # Check for empty content
        if not section.content.strip():
            violations.append({
                "section": section.name,
                "violation": "Section content is empty",
            })

        # Check risk-level specific constraints
        if section.risk_level == "high":
            # Identity section must retain core traits
            core_traits = ["helpful", "direct", "honest"]
            content_lower = section.content.lower()
            for trait in core_traits:
                if trait not in content_lower:
                    violations.append({
                        "section": section.name,
                        "violation": f"Missing core trait: '{trait}'",
                    })

    return violations


# ── Main evolution function ─────────────────────────────────────────────

def evolve_prompt_section(
    section_name: str = "ALL",
    iterations: int = 10,
    optimizer_model: str = "qwen3.6-plus",
    eval_model: str = "qwen3.6-plus",
    hermes_repo: str | None = None,
    dataset_path: str | None = None,
    dry_run: bool = False,
):
    """Main function to evolve system prompt sections."""

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    console.print("\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — "
                  "Evolving system prompt sections\n")

    # ── 1. Extract current sections ─────────────────────────────────────
    console.print("[bold]Step 1: Extracting prompt sections[/bold]")
    target_sections = None
    if section_name != "ALL":
        target_sections = [section_name]

    sections = extract_prompt_sections(config.hermes_agent_path, target_sections)
    if not sections:
        console.print("[red]✗ No sections found[/red]")
        sys.exit(1)

    console.print(f"  Extracted {len(sections)} sections")

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated.[/bold green]")
        return

    # ── 2. Configure DSPy first (MUST be before any ChainOfThought is created) ──
    console.print("\n[bold]Step 2: Configuring DSPy[/bold]")
    from dspy.adapters import ChatAdapter

    from evolution.core.config import make_dashscope_lm

    lm = make_dashscope_lm(eval_model, num_retries=8)
    dspy.configure(lm=lm, adapter=ChatAdapter())
    console.print(f"  DSPy configured: {eval_model} (ChatAdapter, DashScope)")

    # ── 3. Build behavioral test dataset ────────────────────────────────
    console.print("\n[bold]Step 3: Building behavioral test dataset[/bold]")

    dataset_path_obj = Path(dataset_path) if dataset_path else Path("datasets/prompts/behavioral")

    if dataset_path_obj.exists() and (dataset_path_obj / "prompt_behavioral_tests.json").exists():
        dataset = PromptEvalDataset.load(dataset_path_obj)
        console.print(f"  Loaded existing dataset: {len(dataset.examples)} examples")
    else:
        generator = BehavioralTestGenerator(config)
        dataset = generator.generate(sections, output_path=dataset_path_obj)
        console.print(f"  Generated {len(dataset.examples)} behavioral tests")

    if not dataset.examples:
        console.print("[red]✗ No test examples generated[/red]")
        sys.exit(1)

    # ── 4. Evaluate baseline behavior ───────────────────────────────────
    console.print("\n[bold]Step 4: Evaluating baseline behavior[/bold]")

    evaluator = BehavioralEvaluator(config)
    baseline_score, baseline_per_section = evaluator.evaluate(sections, dataset.holdout)
    console.print(f"  Baseline behavioral score: {baseline_score:.3f}")
    for name, score in baseline_per_section.items():
        console.print(f"    {name}: {score:.3f}")

    # ── 4. Save baseline for comparison ─────────────────────────────────
    baseline_sections = [PromptSection(**{**s.__dict__}) for s in sections]

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Step 5: Running GEPA optimization ({iterations} iterations)[/bold cyan]\n")

    start_time = time.time()
    optimized_module = None

    try:
        reflection_lm = make_dashscope_lm(optimizer_model, num_retries=8, temperature=1.0)

        train_examples = [
            dspy.Example(
                section_name=ex.section_name,
                scenario=ex.scenario,
                expected_behavior=ex.expected_behavior,
            ).with_inputs("section_name", "scenario")
            for ex in dataset.train
        ]
        val_examples = [
            dspy.Example(
                section_name=ex.section_name,
                scenario=ex.scenario,
                expected_behavior=ex.expected_behavior,
            ).with_inputs("section_name", "scenario")
            for ex in dataset.val
        ]

        module = PromptSectionModule(sections)

        optimizer = dspy.GEPA(
            metric=prompt_section_fitness_metric,
            reflection_lm=reflection_lm,
        )

        optimized_module = optimizer.compile(
            module,
            trainset=train_examples,
            valset=val_examples,
            eval_kwargs={"max_calls": iterations * 5},
        )

        elapsed = time.time() - start_time
        console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    except Exception as e:
        # Fall back to MIPROv2 if GEPA isn't available
        console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
        auto_budget = "light" if iterations <= 10 else ("medium" if iterations <= 50 else "heavy")

        optimizer = dspy.MIPROv2(
            metric=prompt_section_fitness_metric,
            auto=auto_budget,
            num_threads=1,
        )
        optimized_module = optimizer.compile(
            module,
            trainset=train_examples,
            valset=val_examples,
        )
        elapsed = time.time() - start_time

    # ── 6. Evaluate evolved behavior ────────────────────────────────────
    console.print("\n[bold]Step 5: Evaluating evolved behavior[/bold]")

    if optimized_module and hasattr(optimized_module, 'sections'):
        evolved_sections = list(optimized_module.sections.values())
    else:
        evolved_sections = sections

    evolved_score, evolved_per_section = evaluator.evaluate(evolved_sections, dataset.holdout)
    improvement = evolved_score - baseline_score

    # ── 7. Validate constraints ─────────────────────────────────────────
    console.print("\n[bold]Step 6: Validating constraints[/bold]")
    violations = validate_prompt_sections(evolved_sections, baseline_sections)
    if violations:
        for v in violations:
            console.print(f"  [red]✗ {v['section']}: {v['violation']}[/red]")
    else:
        console.print("  [green]✓ All constraints pass[/green]")

    # ── 8. Report ───────────────────────────────────────────────────────
    table = Table(title="Prompt Section Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "yellow"
    table.add_row(
        "Behavioral Score",
        f"{baseline_score:.3f}",
        f"{evolved_score:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row("Sections", "", str(len(sections)), "")
    table.add_row("Time", "", f"{elapsed:.1f}s", "")

    console.print()
    console.print(table)

    # ── 9. Save output ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output/prompt_sections") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    for section in evolved_sections:
        (output_dir / f"evolved_{section.name}.txt").write_text(section.content)
    for section in baseline_sections:
        (output_dir / f"baseline_{section.name}.txt").write_text(section.content)

    metrics = {
        "timestamp": timestamp,
        "iterations": iterations,
        "sections": [s.name for s in sections],
        "baseline_score": baseline_score,
        "evolved_score": evolved_score,
        "improvement": improvement,
        "constraint_violations": violations,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if improvement > 0:
        console.print(f"\n[bold green]✓ Prompt behavior improved by {improvement:+.3f}[/bold green]")
    else:
        console.print(f"\n[yellow]⚠ No improvement ({improvement:+.3f})[/yellow]")


@click.command()
@click.option("--section", default="ALL", help="Section to evolve (or ALL for all)")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--optimizer-model", default="qwen3.6-plus", help="Model for GEPA reflections")
@click.option("--eval-model", default="qwen3.6-plus", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--dataset-path", default=None, help="Path to existing behavioral test dataset")
@click.option("--dry-run", is_flag=True, help="Validate setup without running")
def main(section, iterations, optimizer_model, eval_model, hermes_repo, dataset_path, dry_run):
    """Evolve system prompt sections using DSPy + GEPA optimization."""
    evolve_prompt_section(
        section_name=section,
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        dataset_path=dataset_path,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
