"""Core infrastructure shared across all evolution phases."""

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.dataset_builder import EvalExample, EvalDataset, SyntheticDatasetBuilder, GoldenDatasetLoader
from evolution.core.fitness import skill_fitness_metric, LLMJudge, FitnessScore
from evolution.core.constraints import ConstraintValidator
from evolution.core.benchmark_gate import run_benchmark_gate
from evolution.core.pr_builder import build_pr, create_pr_branch
