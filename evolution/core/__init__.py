"""Core infrastructure shared across all evolution phases."""

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.fitness import (
    FitnessScore,
    LLMJudge,
    skill_fitness_metric,
    make_llm_judge_metric,
)
from evolution.core.constraints import ConstraintValidator, ConstraintResult
from evolution.core.dataset_builder import (
    EvalDataset,
    SyntheticDatasetBuilder,
    GoldenDatasetLoader,
)
from evolution.core.benchmark_gate import BenchmarkGate, BenchmarkResult, GateResult
from evolution.core.pr_builder import PRBuilder, PRChange, PRMetrics, PRResult
