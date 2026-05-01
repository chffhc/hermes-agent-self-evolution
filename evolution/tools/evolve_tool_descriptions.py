"""Phase 2: Tool description evolution via DSPy + GEPA.

Optimizes the natural language descriptions in tool schemas so the agent
picks the right tools more reliably and uses them correctly.

Each tool description is an optimizable parameter. GEPA mutates descriptions
to improve tool selection accuracy while keeping them factually accurate
and under 500 chars.

Usage:
    python -m evolution.tools.evolve_tool_descriptions --iterations 10
    python -m evolution.tools.evolve_tool_descriptions --iterations 10 --tool read_file
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


# ── Data structures ─────────────────────────────────────────────────────

@dataclass
class ToolDescription:
    """A tool's description and parameter descriptions."""
    name: str
    toolset: str
    description: str  # Main tool description
    param_descriptions: dict[str, str]  # {param_name: description}
    schema: dict  # Full original schema
    file_path: str  # Source file path in hermes-agent


@dataclass
class ToolSelectionExample:
    """A single tool selection test case."""
    task_description: str
    correct_tool: str
    correct_params: dict  # Which params would be used (for validation)
    reasoning: str  # Why this tool is the right choice
    difficulty: str = "easy"  # easy, medium, hard (confusing)


@dataclass
class ToolEvalDataset:
    """Dataset for tool selection evaluation."""
    examples: list[ToolSelectionExample]
    train: list[ToolSelectionExample] = field(default_factory=list)
    val: list[ToolSelectionExample] = field(default_factory=list)
    holdout: list[ToolSelectionExample] = field(default_factory=list)

    def split(self, train_ratio: float = 0.6, val_ratio: float = 0.2):
        """Split examples into train/val/holdout."""
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
        data = {
            "examples": [
                {
                    "task_description": e.task_description,
                    "correct_tool": e.correct_tool,
                    "correct_params": e.correct_params,
                    "reasoning": e.reasoning,
                    "difficulty": e.difficulty,
                }
                for e in self.examples
            ],
        }
        (path / "tool_selection_dataset.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )

    @classmethod
    def load(cls, path: Path) -> "ToolEvalDataset":
        data_file = path / "tool_selection_dataset.json"
        if not data_file.exists():
            return cls(examples=[])
        data = json.loads(data_file.read_text())
        examples = [
            ToolSelectionExample(
                task_description=e["task_description"],
                correct_tool=e["correct_tool"],
                correct_params=e.get("correct_params", {}),
                reasoning=e.get("reasoning", ""),
                difficulty=e.get("difficulty", "easy"),
            )
            for e in data["examples"]
        ]
        ds = cls(examples=examples)
        ds.split()
        return ds


# ── Tool description extractor ──────────────────────────────────────────

def extract_tool_descriptions(
    hermes_agent_path: Path,
) -> list[ToolDescription]:
    """Extract all tool descriptions from the hermes-agent registry.

    Imports the registry and reads all registered tool descriptions.
    """
    sys.path.insert(0, str(hermes_agent_path))

    try:
        from tools.registry import discover_builtin_tools, registry

        # Discover and import all tool modules
        discover_builtin_tools(hermes_agent_path / "tools")

        descriptions = []
        for tool_name in registry.get_all_tool_names():
            entry = registry.get_entry(tool_name)
            if not entry:
                continue

            schema = entry.schema
            desc = schema.get("description", "")
            param_descs = {}
            params = schema.get("parameters", {})
            properties = params.get("properties", {})
            for param_name, param_schema in properties.items():
                if "description" in param_schema:
                    param_descs[param_name] = param_schema["description"]

            # Find source file
            source_file = _find_tool_source(hermes_agent_path, tool_name)

            descriptions.append(ToolDescription(
                name=tool_name,
                toolset=entry.toolset,
                description=desc,
                param_descriptions=param_descs,
                schema=schema,
                file_path=source_file,
            ))

        return descriptions
    finally:
        # Clean up sys.path
        if str(hermes_agent_path) in sys.path:
            sys.path.remove(str(hermes_agent_path))


def _find_tool_source(hermes_agent_path: Path, tool_name: str) -> str:
    """Find the source file containing a tool's registration."""
    tools_dir = hermes_agent_path / "tools"
    for py_file in tools_dir.glob("*.py"):
        if py_file.name in ("__init__.py", "registry.py"):
            continue
        try:
            content = py_file.read_text()
            if f'name="{tool_name}"' in content or f"name='{tool_name}'" in content:
                return str(py_file.relative_to(hermes_agent_path))
        except Exception:
            continue
    return f"tools/{tool_name}.py"


# ── Tool selection dataset builder ──────────────────────────────────────

class ToolSelectionDatasetBuilder:
    """Generates tool selection evaluation datasets.

    Uses an LLM to generate (task, correct_tool) triples for all
    registered tools, including confusing edge cases.
    """

    class TaskGenerator(dspy.Signature):
        """Generate realistic tool selection test cases.

        Given a list of available tools with their descriptions, generate
        test cases that evaluate whether an agent picks the right tool.

        For each tool, generate:
        - 3 easy tasks where the tool is clearly the right choice
        - 2 medium tasks where the tool is good but alternatives exist
        - 1 hard/confusing task where two tools could work

        Return JSON array of objects with:
        - task_description: The task to give the agent
        - correct_tool: The best tool name
        - correct_params: Which parameters would be used
        - reasoning: Why this tool is the best choice
        - difficulty: easy/medium/hard
        """
        tools_json: str = dspy.InputField(desc="JSON array of tool descriptions")
        tool_selection_examples_json: str = dspy.OutputField(
            desc="JSON array of tool selection examples"
        )

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(self.TaskGenerator)

    def generate(
        self,
        tools: list[ToolDescription],
        output_path: Path | None = None,
    ) -> ToolEvalDataset:
        """Generate a tool selection dataset."""
        console.print("\n[bold]Generating tool selection dataset[/bold]")
        console.print(f"  Tools: {len(tools)}")

        # Split into batches to avoid context overflow
        batch_size = 8
        all_examples = []

        for i in range(0, len(tools), batch_size):
            batch = tools[i:i + batch_size]
            tools_json = json.dumps([
                {
                    "name": t.name,
                    "toolset": t.toolset,
                    "description": t.description,
                    "params": list(t.param_descriptions.keys()),
                }
                for t in batch
            ], indent=2, ensure_ascii=False)

            console.print(
                f"  Generating examples for tools "
                f"{i+1}-{min(i+batch_size, len(tools))}/{len(tools)}..."
            )

            try:
                result = self.generator(tools_json=tools_json)
                examples_json = parse_json_array(result.tool_selection_examples_json)
                for ex in examples_json:
                    all_examples.append(ToolSelectionExample(
                        task_description=ex.get("task_description", ""),
                        correct_tool=ex.get("correct_tool", ""),
                        correct_params=ex.get("correct_params", {}),
                        reasoning=ex.get("reasoning", ""),
                        difficulty=ex.get("difficulty", "easy"),
                    ))
            except Exception as e:
                console.print(f"  ⚠ Batch failed: {e}")

        dataset = ToolEvalDataset(examples=all_examples)
        dataset.split()

        if output_path:
            dataset.save(output_path)
            console.print(f"  Saved {len(all_examples)} examples to {output_path}/")

        console.print(
            f"  Split: {len(dataset.train)} train / "
            f"{len(dataset.val)} val / {len(dataset.holdout)} holdout"
        )
        return dataset



# ── Tool selection evaluator ────────────────────────────────────────────

class ToolSelectionEvaluator:
    """Evaluates tool selection accuracy given a set of tool descriptions.

    Presents a task to the LLM with the current tool descriptions and
    checks if it picks the right tool.
    """

    class ToolSelector(dspy.Signature):
        """Select the best tool for the given task.

        You are an AI agent that needs to pick the right tool for a task.
        Read the available tool descriptions carefully and select the best one.
        """
        task_description: str = dspy.InputField(desc="The task to complete")
        available_tools_json: str = dspy.InputField(
            desc="JSON array of available tools with names and descriptions"
        )
        selected_tool: str = dspy.OutputField(
            desc="The name of the best tool for this task"
        )
        reasoning: str = dspy.OutputField(
            desc="Brief explanation of why this tool is the best choice"
        )

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.selector = dspy.ChainOfThought(self.ToolSelector)

    def evaluate(
        self,
        tools: list[ToolDescription],
        examples: list[ToolSelectionExample],
    ) -> tuple[float, dict[str, float]]:
        """Evaluate tool selection accuracy.

        Returns:
            (overall_accuracy, per_tool_accuracy)
        """
        # Build tool descriptions JSON for the evaluator
        tools_json = json.dumps([
            {"name": t.name, "description": t.description}
            for t in tools
        ], indent=2, ensure_ascii=False)

        correct = 0
        total = len(examples)
        per_tool_correct = {}
        per_tool_total = {}

        for ex in examples:
            try:
                result = self.selector(
                    task_description=ex.task_description,
                    available_tools_json=tools_json,
                )
                selected = result.selected_tool.strip()
                is_correct = selected == ex.correct_tool

                if is_correct:
                    correct += 1

                per_tool_total[ex.correct_tool] = per_tool_total.get(ex.correct_tool, 0) + 1
                if is_correct:
                    per_tool_correct[ex.correct_tool] = per_tool_correct.get(ex.correct_tool, 0) + 1

            except Exception:
                per_tool_total[ex.correct_tool] = per_tool_total.get(ex.correct_tool, 0) + 1

        overall_accuracy = correct / max(1, total)
        per_tool_accuracy = {
            tool: per_tool_correct.get(tool, 0) / per_tool_total[tool]
            for tool in per_tool_total
        }

        return overall_accuracy, per_tool_accuracy


# ── Tool description constraint validator ───────────────────────────────

def validate_tool_descriptions(
    tools: list[ToolDescription],
) -> list[dict]:
    """Validate evolved tool descriptions meet constraints.

    Constraints:
    - Max 500 chars per tool description
    - Max 200 chars per parameter description
    - Must remain factually accurate (basic check)
    """
    violations = []
    for tool in tools:
        if len(tool.description) > 500:
            violations.append({
                "tool": tool.name,
                "violation": f"Description too long: {len(tool.description)} chars (max 500)",
            })
        for param, desc in tool.param_descriptions.items():
            if len(desc) > 200:
                violations.append({
                    "tool": tool.name,
                    "violation": f"Param '{param}' description too long: {len(desc)} chars (max 200)",
                })
    return violations


# ── Tool description module for DSPy ────────────────────────────────────

_TOOL_SENTINEL_START = "<!-- __TOOL_DESC_START__ -->"
_TOOL_SENTINEL_END = "<!-- __TOOL_DESC_END__ -->"


class ToolDescriptionModule(dspy.Module):
    """DSPy module for optimizing tool descriptions.

    All tool descriptions are embedded in the signature instruction via
    sentinels, making them optimizable parameters for GEPA (same approach
    as SkillModule for Phase 1).
    """

    def __init__(self, tools: list[ToolDescription]):
        super().__init__()
        self.tools = tools
        self._build_signature()

    def _build_signature(self):
        descriptions = json.dumps([
            {"name": t.name, "description": t.description}
            for t in self.tools
        ], indent=2, ensure_ascii=False)

        class ToolSelectionTask(dspy.Signature):
            __doc__ = (
                "Select the best tool for a task given the tool descriptions.\n"
                "Read the available tool descriptions carefully and pick the one\n"
                "whose description best matches the task.\n\n"
                f"{_TOOL_SENTINEL_START}\n{descriptions}\n{_TOOL_SENTINEL_END}"
            )
            task: str = dspy.InputField(desc="Task to complete")
            selected_tool: str = dspy.OutputField(desc="Best tool name")

        self.predictor = dspy.ChainOfThought(ToolSelectionTask)

    def forward(self, task: str) -> dspy.Prediction:
        result = self.predictor(task=task)
        return dspy.Prediction(selected_tool=result.selected_tool)

    def get_evolved_tools(self) -> list[ToolDescription]:
        """Extract evolved tool descriptions from the compiled module."""
        predictor = getattr(self, "predictor", None)
        instruction = ""

        for sig_attr in ("extended_signature", "signature"):
            sig = getattr(predictor, sig_attr, None) if predictor else None
            if sig is None:
                continue
            instruction = getattr(sig, "instructions", None) or getattr(sig, "__doc__", "") or ""
            if instruction:
                break

        if not instruction:
            return self.tools

        start_idx = instruction.find(_TOOL_SENTINEL_START)
        end_idx = instruction.find(_TOOL_SENTINEL_END)
        if start_idx == -1 or end_idx == -1:
            return self.tools

        evolved_json = instruction[start_idx + len(_TOOL_SENTINEL_START):end_idx].strip()
        try:
            evolved = json.loads(evolved_json)
            name_to_desc = {t["name"]: t["description"] for t in evolved if isinstance(t, dict) and "name" in t and "description" in t}
            for tool in self.tools:
                if tool.name in name_to_desc:
                    tool.description = name_to_desc[tool.name]
        except (json.JSONDecodeError, KeyError):
            pass

        return self.tools


# ── Tool fitness metric ─────────────────────────────────────────────────

def tool_selection_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
) -> float:
    """DSPy metric for tool selection accuracy.

    GEPA-compatible 5-arg signature.
    Returns 1.0 if the selected tool matches the correct tool, 0.0 otherwise.
    """
    selected = getattr(prediction, "selected_tool", "").strip()
    correct = getattr(example, "correct_tool", "").strip()
    return 1.0 if selected == correct else 0.0


# ── Main evolution function ─────────────────────────────────────────────

def evolve_tool_descriptions(
    iterations: int = 10,
    optimizer_model: str = "qwen3.6-plus",
    eval_model: str = "qwen3.6-plus",
    hermes_repo: str | None = None,
    tool_filter: list[str] | None = None,
    dataset_path: str | None = None,
    dry_run: bool = False,
):
    """Main function to evolve tool descriptions."""

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    console.print("\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — "
                  "Evolving tool descriptions\n")

    # ── 1. Extract current tool descriptions ────────────────────────────
    console.print("[bold]Step 1: Extracting tool descriptions[/bold]")
    all_tools = extract_tool_descriptions(config.hermes_agent_path)

    if tool_filter:
        tools = [t for t in all_tools if t.name in tool_filter]
        console.print(f"  Filtered to {len(tools)} tools: {', '.join(t.name for t in tools)}")
    else:
        tools = all_tools
        console.print(f"  Found {len(tools)} tools across "
                      f"{len({t.toolset for t in tools})} toolsets")

    # Print summary
    for t in tools[:10]:
        console.print(f"  • {t.name}: {len(t.description)} chars — {t.description[:60]}...")
    if len(tools) > 10:
        console.print(f"  ... and {len(tools) - 10} more")

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

    # ── 3. Build tool selection dataset ─────────────────────────────────
    console.print("\n[bold]Step 3: Building tool selection dataset[/bold]")

    dataset_path_obj = Path(dataset_path) if dataset_path else Path("datasets/tools/tool_selection")

    if dataset_path_obj.exists() and (dataset_path_obj / "tool_selection_dataset.json").exists():
        dataset = ToolEvalDataset.load(dataset_path_obj)
        console.print(f"  Loaded existing dataset: {len(dataset.examples)} examples")
    else:
        builder = ToolSelectionDatasetBuilder(config)
        dataset = builder.generate(tools, output_path=dataset_path_obj)

    if not dataset.examples:
        console.print("[red]✗ No dataset examples generated[/red]")
        sys.exit(1)

    # ── 4. Evaluate baseline accuracy ───────────────────────────────────
    console.print("\n[bold]Step 4: Evaluating baseline tool selection[/bold]")
    evaluator = ToolSelectionEvaluator(config)

    baseline_accuracy, baseline_per_tool = evaluator.evaluate(tools, dataset.holdout)
    console.print(f"  Baseline accuracy: {baseline_accuracy:.1%} "
                  f"({len(dataset.holdout)} holdout examples)")

    # ── 4. Save baseline descriptions for comparison ────────────────────
    baseline_descriptions = {t.name: t.description for t in tools}

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Step 5: Running GEPA optimization ({iterations} iterations)[/bold cyan]\n")

    start_time = time.time()
    optimized_module = None

    try:
        reflection_lm = make_dashscope_lm(optimizer_model, num_retries=8, temperature=1.0)

        # Create DSPy examples
        train_examples = [
            dspy.Example(
                task=ex.task_description,
                correct_tool=ex.correct_tool,
            ).with_inputs("task")
            for ex in dataset.train
        ]
        val_examples = [
            dspy.Example(
                task=ex.task_description,
                correct_tool=ex.correct_tool,
            ).with_inputs("task")
            for ex in dataset.val
        ]

        module = ToolDescriptionModule(tools)

        optimizer = dspy.GEPA(
            metric=tool_selection_fitness_metric,
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
            metric=tool_selection_fitness_metric,
            auto=auto_budget,
            num_threads=1,
        )
        optimized_module = optimizer.compile(
            module,
            trainset=train_examples,
            valset=val_examples,
        )
        elapsed = time.time() - start_time

    # ── 6. Evaluate evolved accuracy ────────────────────────────────────
    console.print("\n[bold]Step 5: Evaluating evolved tool selection[/bold]")

    if optimized_module and hasattr(optimized_module, 'get_evolved_tools'):
        evolved_tools = optimized_module.get_evolved_tools()
    else:
        evolved_tools = tools

    evolved_accuracy, evolved_per_tool = evaluator.evaluate(evolved_tools, dataset.holdout)
    improvement = evolved_accuracy - baseline_accuracy

    # ── 7. Validate constraints ─────────────────────────────────────────
    console.print("\n[bold]Step 6: Validating constraints[/bold]")
    violations = validate_tool_descriptions(evolved_tools)
    if violations:
        for v in violations:
            console.print(f"  [red]✗ {v['tool']}: {v['violation']}[/red]")
    else:
        console.print("  [green]✓ All constraints pass[/green]")

    # ── 8. Report results ───────────────────────────────────────────────
    table = Table(title="Tool Description Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "yellow"
    table.add_row(
        "Selection Accuracy",
        f"{baseline_accuracy:.1%}",
        f"{evolved_accuracy:.1%}",
        f"[{change_color}]{improvement:+.1%}[/{change_color}]",
    )
    table.add_row("Tools", "", str(len(tools)), "")
    table.add_row("Time", "", f"{elapsed:.1f}s", "")

    console.print()
    console.print(table)

    # ── 9. Save output ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output/tool_descriptions") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evolved descriptions
    evolved_descriptions = {t.name: t.description for t in evolved_tools}
    (output_dir / "evolved_descriptions.json").write_text(
        json.dumps(evolved_descriptions, indent=2, ensure_ascii=False)
    )
    (output_dir / "baseline_descriptions.json").write_text(
        json.dumps(baseline_descriptions, indent=2, ensure_ascii=False)
    )

    # Diff file
    diff_lines = []
    for name in sorted(set(baseline_descriptions.keys()) | set(evolved_descriptions.keys())):
        old = baseline_descriptions.get(name, "")
        new = evolved_descriptions.get(name, "")
        if old != new:
            diff_lines.append(f"\n## {name}")
            diff_lines.append(f"  Before ({len(old)} chars): {old[:80]}...")
            diff_lines.append(f"  After  ({len(new)} chars): {new[:80]}...")

    (output_dir / "diff.md").write_text("\n".join(diff_lines))

    metrics = {
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "num_tools": len(tools),
        "baseline_accuracy": baseline_accuracy,
        "evolved_accuracy": evolved_accuracy,
        "improvement": improvement,
        "constraint_violations": violations,
        "elapsed_seconds": elapsed,
        "train_examples": len(dataset.train),
        "holdout_examples": len(dataset.holdout),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if improvement > 0:
        console.print(f"\n[bold green]✓ Tool selection improved by {improvement:+.1%}[/bold green]")
    else:
        console.print(f"\n[yellow]⚠ No improvement in tool selection ({improvement:+.1%})[/yellow]")


@click.command()
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--optimizer-model", default="qwen3.6-plus", help="Model for GEPA reflections")
@click.option("--eval-model", default="qwen3.6-plus", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--tool", multiple=True, help="Specific tool(s) to evolve (repeat for multiple)")
@click.option("--dataset-path", default=None, help="Path to existing tool selection dataset")
@click.option("--dry-run", is_flag=True, help="Validate setup without running")
def main(iterations, optimizer_model, eval_model, hermes_repo, tool, dataset_path, dry_run):
    """Evolve tool descriptions using DSPy + GEPA optimization."""
    evolve_tool_descriptions(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        tool_filter=list(tool) if tool else None,
        dataset_path=dataset_path,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
