# Capability benchmark foundation

## Status and evidence boundary

`benchmarks/run_bench.py` remains a repository smoke/proxy check. The Level 1 package in `benchmarks/capability/` adds deterministic task contracts, verifiers, replay self-tests, paired comparison, a dry-run Hermes integration seam, and a local isolated-workspace executor driven by a bundled fake agent. **Replay, dry-run, and fake-agent output are always `capability_evidence=false`; they do not measure an agent.** Only a genuine live Hermes invocation — which is not yet implemented — could ever be labeled evidence.

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

## Local executor seam (`run-fake`)

`benchmarks/capability/executor.py` implements the live-adapter foundation with a local fake agent (`benchmarks/capability/fixtures/fake_agent.py`) instead of a model:

- **Isolation** — one disposable run root per run (`tempfile.mkdtemp`), one `tasks/<task_id>/{workspace,control}` tree per task; fixture workspaces are copied with `symlinks=False`; a symlink found in the final workspace fails the task closed.
- **Artifact injection with digest binding** — exactly one baseline/candidate artifact is copied into `workspace/hermes_artifact/`; the source digest is checked against an optional pinned `--artifact-digest`, the injected copy is re-digested, and any mismatch fails closed. This is the boundary a live adapter will reuse for real Hermes artifacts.
- **Attribution** — every run carries a `run_id`; each invocation records `control/invocation.json` (run ID, task ID, argv), and the agent process receives `HERMES_BENCH_RUN_ID`/`HERMES_BENCH_TASK_ID` plus argv placeholders.
- **Invocation seam** — agents run through an injectable argv/callable `AgentInvoker` (never `shell=True`), with per-task subprocess timeouts and guaranteed cleanup of the run root on success or failure; `--keep-workspaces` retains it for debugging (path recorded in run notes).
- **Budget gate** — each invocation must write a strict `usage.json` (`cost_usd`, `input_tokens`, `output_tokens`); missing, malformed, or over-budget usage fails the task closed, and exhausting the hard run budget (`--budget-usd`) blocks all remaining tasks.
- **Honesty** — only the `fake_agent` execution mode is implemented; the executor and schema both hard-reject `capability_evidence=true` for it.

## Current CLI

```bash
python -m benchmarks.capability validate --suite benchmarks/capability/suites/native_v1/suite.json
python -m benchmarks.capability replay ...
python -m benchmarks.capability run-fake --suite ... --role candidate --artifact <path> \
    --model test/model --environment local --solve --output candidate.json
python -m benchmarks.capability compare ...
python -m benchmarks.capability plan-batch ...
```

`run-fake` exercises the full executor seam end-to-end without any paid model call; its output is harness validation, never capability evidence. `plan-batch` validates current `batch_runner.py` and emits a dataset plus command shape, but deliberately returns `executable=false`. Current `batch_runner.py` does not yet provide isolated per-task workspace mounting, candidate artifact injection, deterministic post-run verification, or complete cost attribution.

## Next live milestone

Implement a real Hermes `AgentInvoker` behind the existing executor seam: invoke current Hermes with explicit model/tool/budget settings against the injected per-task workspace, extract genuine per-task usage/cost into the strict `usage.json` contract, capture the trajectory under the run ID, and only then permit schema-validated `execution_mode=live` results. A real baseline/candidate pair must be run — with prerequisites and the hard USD budget satisfied — before any production-readiness claim.
