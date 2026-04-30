"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a parameterized DSPy module
where the skill text is the optimizable parameter. GEPA can then
mutate the skill text and evaluate the results.

Fixes:
- Skill text embedded in instruction template (not as input field) so
  GEPA/MIPROv2 can actually mutate it.
- HTML comment sentinel for clean extraction of evolved skill body.
"""

import re
from pathlib import Path
from typing import Optional

import dspy

# Sentinel that cannot appear in markdown content
_SENTINEL_START = "<!-- __SKILL_EVOLVED_START__ -->"
_SENTINEL_END = "<!-- __SKILL_EVOLVED_END__ -->"


def load_skill(skill_path: Path) -> dict:
    """Load a skill file and parse its frontmatter + body.

    Returns:
        {
            "path": Path,
            "raw": str (full file content),
            "frontmatter": str (YAML between --- markers),
            "body": str (markdown after frontmatter),
            "name": str,
            "description": str,
        }
    """
    raw = skill_path.read_text()

    # Parse YAML frontmatter
    frontmatter = ""
    body = raw
    if raw.strip().startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()

    # Extract name and description from frontmatter
    name = ""
    description = ""
    for line in frontmatter.split("\n"):
        if line.strip().startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("'\"")
        elif line.strip().startswith("description:"):
            description = line.split(":", 1)[1].strip().strip("'\"")

    return {
        "path": skill_path,
        "raw": raw,
        "frontmatter": frontmatter,
        "body": body,
        "name": name,
        "description": description,
    }


def find_skill(skill_name: str, hermes_agent_path: Path) -> Optional[Path]:
    """Find a skill by name in the hermes-agent skills directory.

    Searches recursively for a SKILL.md in a directory matching the skill name.
    """
    skills_dir = hermes_agent_path / "skills"
    if not skills_dir.exists():
        return None

    # Direct match: skills/<category>/<skill_name>/SKILL.md
    for skill_md in skills_dir.rglob("SKILL.md"):
        if skill_md.parent.name == skill_name:
            return skill_md

    # Fuzzy match: check the name field in frontmatter
    for skill_md in skills_dir.rglob("SKILL.md"):
        try:
            content = skill_md.read_text()[:500]
            if f"name: {skill_name}" in content or f'name: "{skill_name}"' in content:
                return skill_md
        except Exception:
            continue

    return None


class SkillModule(dspy.Module):
    """A DSPy module that wraps a skill file for optimization.

    The skill text (body) is the parameter that GEPA/MIPROv2 optimizes.
    Unlike the original implementation which passed skill text as an input
    field (making it unoptimizable), this version embeds it in the
    instruction template of a dynamically-created Signature, allowing
    the optimizer to propose improved skill bodies.
    """

    def __init__(self, skill_text: str):
        super().__init__()
        # Dynamically create a Signature with the skill text as its instruction.
        # This makes the skill text an optimizable parameter rather than a static input.
        class TaskWithSkill(dspy.Signature):
            __doc__ = (
                "You are an AI agent following specific skill instructions to complete a task.\n"
                "Read the skill instructions carefully and follow the procedure described.\n\n"
                f"{_SENTINEL_START}\n{skill_text}\n{_SENTINEL_END}"
            )
            task_input: str = dspy.InputField(desc="The task to complete")
            output: str = dspy.OutputField(desc="Your response following the skill instructions")

        self.predictor = dspy.ChainOfThought(TaskWithSkill)

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        return dspy.Prediction(output=result.output)


def extract_evolved_skill_text(optimized_module: dspy.Module) -> str:
    """Extract the evolved skill body from a compiled DSPy module.

    Uses HTML comment sentinels to reliably locate the skill text within
    the optimizer's instruction template, supporting both GEPA (extended_signature)
    and MIPROv2/BootstrapFewShot (signature).
    """
    predictor = getattr(optimized_module, "predictor", None)
    instruction = ""

    # Try extended_signature first (GEPA), then signature (MIPROv2/Bootstrap)
    for sig_attr in ("extended_signature", "signature"):
        sig = getattr(predictor, sig_attr, None) if predictor else None
        if sig is None:
            continue
        instruction = getattr(sig, "instructions", None) or getattr(sig, "__doc__", "") or ""
        if instruction:
            break

    # Fallback: search sub-predictors
    if not instruction and predictor:
        for attr_name in dir(predictor):
            attr = getattr(predictor, attr_name, None)
            if isinstance(attr, dspy.Predict) and hasattr(attr, "signature"):
                instruction = getattr(attr.signature, "instructions", None) or getattr(attr.signature, "__doc__", "") or ""
                if instruction:
                    break

    if not instruction:
        raise ValueError(
            f"Cannot find instruction in optimized module predictor. "
            f"Available attrs: {[a for a in dir(predictor) if not a.startswith('_')]}"
        )

    # Extract between sentinels
    start_idx = instruction.find(_SENTINEL_START)
    end_idx = instruction.find(_SENTINEL_END)

    if start_idx == -1 or end_idx == -1:
        raise ValueError(
            f"Cannot find skill text sentinels in evolved instruction. "
            f"Instruction preview: {instruction[:300]}"
        )

    evolved_body = instruction[start_idx + len(_SENTINEL_START):end_idx].strip()
    return evolved_body


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body."""
    return f"---\n{frontmatter}\n---\n\n{evolved_body}\n"
