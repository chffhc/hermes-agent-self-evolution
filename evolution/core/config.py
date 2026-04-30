"""Configuration and hermes-agent repo discovery.

Automatically discovers DashScope API credentials from ~/.hermes/.env
to match the user's existing Hermes Agent configuration.
"""

import os
import functools
from pathlib import Path
from dataclasses import dataclass, field


@functools.lru_cache(maxsize=1)
def _load_hermes_env() -> None:
    """Load environment variables from ~/.hermes/.env if not already set.

    This ensures the evolution pipeline reuses the same API key and base URL
    as the user's existing Hermes Agent installation.
    """
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.exists():
        return

    # Parse .env file (simple KEY=VALUE format)
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Only set if not already in environment
            if key not in os.environ:
                os.environ[key] = value


def get_api_key() -> str:
    """Get the API key from environment or Hermes config."""
    _load_hermes_env()

    # Priority: DASHSCOPE_API_KEY > OPENAI_API_KEY
    key = os.getenv("DASHSCOPE_API_KEY")
    if key:
        return key

    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    raise OSError(
        "No API key found. Set DASHSCOPE_API_KEY or OPENAI_API_KEY in "
        "~/.hermes/.env or as environment variable."
    )


def get_api_base() -> str:
    """Get the API base URL from environment or Hermes config."""
    _load_hermes_env()

    # Priority: DASHSCOPE_BASE_URL > OPENAI_API_BASE
    base = os.getenv("DASHSCOPE_BASE_URL")
    if base:
        return base

    base = os.getenv("OPENAI_API_BASE")
    if base:
        return base

    # Default to DashScope compatible mode
    return "https://dashscope.aliyuncs.com/compatible-mode/v1"


def make_lm(model: str, **kwargs) -> "dspy.LM":
    """Create a DSPy LM configured for DashScope / OpenAI-compatible API.

    Args:
        model: Model name (e.g., 'qwen3.6-plus', 'qwen-max').
               DSPy's LM accepts the model name and uses the api_base/api_key
               for routing. Using the 'openai/' prefix triggers OpenAI-compatible mode.

    Returns:
        Configured dspy.LM instance.
    """
    import dspy

    # If model doesn't already have a provider prefix, use openai/ for compatibility
    if "/" not in model:
        model = f"openai/{model}"

    return dspy.LM(
        model=model,
        api_base=get_api_base(),
        api_key=get_api_key(),
        **kwargs,
    )


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    # hermes-agent repo path
    hermes_agent_path: Path = field(default_factory=lambda: get_hermes_agent_path())

    # Optimization parameters
    iterations: int = 10
    population_size: int = 5

    # LLM configuration — defaults to DashScope qwen3.6-plus
    optimizer_model: str = "qwen3.6-plus"  # Model for GEPA reflections
    eval_model: str = "qwen3.6-plus"       # Model for LLM-as-judge scoring
    judge_model: str = "qwen3.6-plus"      # Model for dataset generation

    # Constraints
    max_skill_size: int = 50_000  # 50KB default (evolved skills may include few-shot examples)
    max_tool_desc_size: int = 500  # chars
    max_param_desc_size: int = 200  # chars
    max_prompt_growth: float = 0.2  # 20% max growth over baseline

    # Eval dataset
    eval_dataset_size: int = 20  # Total examples to generate
    train_ratio: float = 0.5
    val_ratio: float = 0.25
    holdout_ratio: float = 0.25

    # Benchmark gating
    run_pytest: bool = True
    run_tblite: bool = False  # Expensive — opt-in
    tblite_regression_threshold: float = 0.02  # Max 2% regression allowed

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    create_pr: bool = True


def get_hermes_agent_path() -> Path:
    """Discover the hermes-agent repo path.

    Priority:
    1. HERMES_AGENT_REPO env var
    2. ~/.hermes/hermes-agent (standard install location)
    3. ../hermes-agent (sibling directory)
    """
    env_path = os.getenv("HERMES_AGENT_REPO")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    home_path = Path.home() / ".hermes" / "hermes-agent"
    if home_path.exists():
        return home_path

    sibling_path = Path(__file__).parent.parent.parent / "hermes-agent"
    if sibling_path.exists():
        return sibling_path

    raise FileNotFoundError(
        "Cannot find hermes-agent repo. Set HERMES_AGENT_REPO env var "
        "or ensure it exists at ~/.hermes/hermes-agent"
    )
