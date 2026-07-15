# Capability benchmark foundation

## Status and evidence boundary

`benchmarks/run_bench.py` remains a repository smoke/proxy check. The Level 1 package in `benchmarks/capability/` adds deterministic task contracts, verifiers, replay self-tests, paired comparison, and a dry-run Hermes integration seam. **Replay and dry-run output are always `capability_evidence=false`; they do not measure an agent.**

## Local archaeology

The current Hermes checkout still contains `batch_runner.py`, trajectory/statistics output, checkpointing, worker concurrency, and container-image fields. Historical benchmark environments were added in commits `0ea6c3432` / `ee7fde653` (OpenThoughts-TBLite) and `b4fbb6fe1` (YC-Bench). Commit `5af672c75` later removed the Atropos environment tree, including:

- `environments/benchmarks/tblite/`
- `environments/benchmarks/terminalbench_2/`
- `environments/benchmarks/yc_bench/`

That removal deleted more than 8,000 lines spanning agent loops, parser copies, sandbox integrations, and RL-specific environment code. Restoring it wholesale would couple self-evolution to removed infrastructure. Future adapters should recover only task loading, sandbox contracts, graders, and result normalization that still match current Hermes.

## Three levels

1. **Native deterministic suite** — cheap private regression tasks with disposable workspaces and programmatic verifiers. This repository now has a three-task harness-validation suite covering file editing, JSON transformation, and code repair.
2. **Historical/external adapters** — modern adapters for TBLite, YC-Bench, and Terminal-Bench 2, without restoring the Atropos stack.
3. **Heavy public benchmarks** — SWE-bench Verified, GAIA, OSWorld/WebArena, or similar suites run periodically, not on every optimizer iteration.

## Paired-run invariants

A baseline and candidate are comparable only when all of these match:

- exact task IDs and suite hash;
- model and non-artifact configuration digest;
- seed and environment fingerprint;
- execution mode and evidence classification;
- sandbox initial state, tool distribution, maximum turns, and budgets.

Only the evolved artifact may differ. Candidate injection must occur in an isolated workspace; the production Hermes checkout must not be mutated. Missing tasks, malformed counters, fingerprint mismatch, unknown verifiers, path traversal, or critical regressions fail closed.

## Anti-leakage rules

- Holdout task answers never enter optimizer feedback.
- Deterministic final-state verifiers take precedence over transcript keywords or an LLM judge.
- Tests and protected files can be marked immutable.
- Task IDs and expected outputs must not be embedded into candidate artifacts.
- Fixture/replay runs can validate the harness but can never be labeled as agent capability evidence.

## Current CLI

```bash
python -m benchmarks.capability validate --suite benchmarks/capability/suites/native_v1/suite.json
python -m benchmarks.capability replay ...
python -m benchmarks.capability compare ...
python -m benchmarks.capability plan-batch ...
```

`plan-batch` validates current `batch_runner.py` and emits a dataset plus command shape, but deliberately returns `executable=false`. Current `batch_runner.py` does not yet provide isolated per-task workspace mounting, candidate artifact injection, deterministic post-run verification, or complete cost attribution.

## Next live milestone

Implement a dedicated current-Hermes adapter that creates one isolated workspace per task, injects exactly one baseline or candidate artifact, invokes Hermes with explicit model/tool/budget settings, captures the final workspace and trajectory under a run ID, invokes deterministic verifiers, and emits schema-validated `execution_mode=live` results. A real baseline/candidate pair must then be run before any production-readiness claim.
