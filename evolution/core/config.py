"""Configuration and hermes-agent repo discovery.

Automatically discovers DashScope API credentials from ~/.hermes/.env
to match the user's existing Hermes Agent configuration.
"""

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


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


def _record_lm_usage(lm: Any) -> None:
    """Record any new DSPy LM history entries since the last tracked call."""
    from evolution.core.cost_tracker import tracker

    try:
        history = getattr(lm, "history", []) or []
        last_recorded = getattr(lm, "_usage_tracking_last_history_len", 0)
        raw_model = getattr(lm, "_usage_tracking_raw_model", "unknown")
        for entry in history[last_recorded:]:
            usage = entry.get("usage", {}) if isinstance(entry, dict) else {}
            if not isinstance(usage, dict):
                continue
            inp = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
            out = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
            tracker.record(raw_model, int(inp), int(out))
        lm._usage_tracking_last_history_len = len(history)
    except Exception:
        pass  # Usage tracking must never break inference.


def install_usage_tracking(lm: Any, raw_model: str) -> Any:
    """Install per-instance usage tracking while preserving ``isinstance(lm, dspy.LM)``.

    Assigning ``lm.__call__`` on the instance is ineffective for ``lm(...)``
    because Python resolves special methods on the type. Instead, replace the
    instance's class with a one-off subclass that wraps ``__call__``. This keeps
    the object a real DSPy LM for code that checks ``isinstance(lm, dspy.LM)``.
    """
    if getattr(lm, "_usage_tracking_installed", False):
        return lm

    original_cls = lm.__class__

    class UsageTrackedLM(original_cls):  # type: ignore[misc, valid-type]
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            _record_lm_usage(self)
            return result

    UsageTrackedLM.__name__ = f"UsageTracked{original_cls.__name__}"
    UsageTrackedLM.__qualname__ = UsageTrackedLM.__name__
    lm._usage_tracking_raw_model = raw_model
    lm._usage_tracking_last_history_len = 0
    lm._usage_tracking_installed = True
    lm.__class__ = UsageTrackedLM
    return lm


def make_lm(model: str, track_usage: bool = True, **kwargs) -> Any:
    """Create a DSPy LM configured for DashScope / OpenAI-compatible API.

    Args:
        model: Model name (e.g., 'qwen3.6-plus', 'qwen-max').
               DSPy's LM accepts the model name and uses the api_base/api_key
               for routing. Using the 'openai/' prefix triggers OpenAI-compatible mode.
        track_usage: If True, wrap LM to record token usage and estimated cost.

    Returns:
        Configured dspy.LM instance.
    """
    import dspy

    # If model doesn't already have a provider prefix, use openai/ for compatibility
    if "/" not in model:
        model = f"openai/{model}"

    lm = dspy.LM(
        model=model,
        api_base=get_api_base(),
        api_key=get_api_key(),
        **kwargs,
    )

    return install_usage_tracking(lm, model) if track_usage else lm


def make_dashscope_lm(model: str = "qwen3.6-plus", num_retries: int = 8, **kwargs) -> Any:
    """Create a DashScope LM with ChatAdapter-compatible settings.

    This is a convenience wrapper around make_lm that adds model_type='chat'
    for the DashScope coding endpoint, required for DSPy ChatAdapter usage.
    """
    return make_lm(model=model, num_retries=num_retries, model_type="chat", **kwargs)


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    # hermes-agent repo path. Discovered lazily and non-fatally so the config
    # can be constructed even when no repo is present (e.g. unit tests, or
    # callers that pass an explicit path). Use resolve_hermes_agent_path() when
    # an explicit override should win, or get_hermes_agent_path() to require one.
    hermes_agent_path: Path | None = field(default_factory=lambda: _discover_hermes_agent_path())

    # Optimization parameters
    iterations: int = 10
    population_size: int = 5

    # LLM configuration — defaults to DashScope qwen3.6-plus
    optimizer_model: str = "qwen3.6-plus"  # Model for GEPA reflections
    eval_model: str = "qwen3.6-plus"  # Model for LLM-as-judge scoring
    judge_model: str = "qwen3.6-plus"  # Model for dataset generation

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


def _discover_hermes_agent_path() -> Path | None:
    """Best-effort hermes-agent repo discovery that never raises.

    Returns the discovered path, or None when no repo can be found. Used as
    the EvolutionConfig default so construction never crashes; callers that
    truly require the repo should use get_hermes_agent_path().
    """
    try:
        return get_hermes_agent_path()
    except FileNotFoundError:
        return None


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


def resolve_hermes_agent_path(hermes_repo: str | None = None) -> Path:
    """Return the hermes-agent repo path, honoring an explicit override.

    An explicit path (for example from ``--hermes-repo``) is expanded and used
    as-is, taking precedence over auto-discovery. This lets callers point at a
    repo in a non-default location without the tool crashing just because
    ``~/.hermes/hermes-agent`` happens to be absent. When no override is given,
    falls back to :func:`get_hermes_agent_path`.
    """
    if hermes_repo:
        return Path(hermes_repo).expanduser()
    return get_hermes_agent_path()
