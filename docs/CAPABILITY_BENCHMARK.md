# Capability benchmark foundation

## Status and evidence boundary

`benchmarks/run_bench.py` remains a repository smoke/proxy check. The Level 1 package in `benchmarks/capability/` adds deterministic task contracts, verifiers, replay self-tests, paired comparison, a local isolated-workspace executor driven by a bundled fake agent, a current-Hermes CLI adapter foundation (compatibility probe, skill-artifact injection contract, contract-emulating stub, and a fail-closed live design), and a structural live-readiness gate that turns the two live-execution blockers into typed fail-closed contracts with an informational `probe-live-readiness` command. **Replay, dry-run, fake-agent, and hermes-cli-stub output are always `capability_evidence=false`; they do not measure an agent. Schema v1 rejects `capability_evidence=true` for every execution mode—including externally supplied `live` JSON—and comparison rejects manually constructed evidence-bearing objects.** Paid execution is intentionally blocked: current Hermes exposes post-run cost attribution but no enforceable pre-spend USD ceiling, and `TERMINAL_CWD` does not confine absolute filesystem access.

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
- **Post-run accounting gate** — each invocation must write a strict `usage.json` (`cost_usd`, `input_tokens`, `output_tokens`); missing, malformed, or over-ceiling usage fails the task closed, and an observed cumulative overage blocks all remaining tasks. This is not a pre-spend hard limit: the subprocess may already have incurred the reported cost. Fake/stub runs are deterministic and free; live execution remains blocked until a real pre-spend enforcement mechanism exists.
- **Honesty** — fake-agent and hermes-cli-stub modes are implemented and permanently non-evidence; schema and executor reject `capability_evidence=true` for both.

## Current-Hermes integration seam (verified against 0.18.2)

`benchmarks/capability/hermes_adapter.py` implements the concrete adapter foundation. Every invariant below was verified read-only against the current checkout (`pyproject.toml` `name = "hermes-agent"`, `version = "0.18.2"`), and `probe-hermes` re-verifies all of them fail-closed before any live invocation:

- **Non-interactive invocation** — `cli.py:main` (`def main(query=..., quiet=..., skills=..., toolsets=..., model=...)`, ~line 15765) implements single-query mode: the final response goes to stdout, `session_id: <id>` goes to stderr (`print(f"\nsession_id: {cli.session_id}", file=sys.stderr)`), and the exit code is 0/1 from `result["failed"]`. `hermes_cli/main.py:cmd_chat` translates `ValueError` (e.g. all requested skills unknown) into `Error: …` + exit 1.
- **Skill preloading (the first live artifact type)** — `--skills <name>` routes through `agent/skill_commands.py:build_preloaded_skills_prompt` (line ~667) and `_build_skill_message`, which embed the SKILL.md body verbatim into the session system prompt. Skills resolve from `tools/skills_tool.py:SKILLS_DIR = HERMES_HOME / "skills"` (lines 142–143); layout is `skills/<name>/SKILL.md` with required `name`/`description` frontmatter. Caveat: `_substitute_template_vars` rewrites `{{…}}` tokens before embedding (on by default), so the artifact contract forbids them to keep the byte-exact consumption proof valid; `inline_shell` expansion is off by default.
- **State/cwd scoping, not a sandbox** — `hermes_constants.py:get_hermes_home` honors the `HERMES_HOME` env var (single source of truth for skills, config, `.env`, and `state.db`); `tools/terminal_tool.py` honors `TERMINAL_CWD` (~line 1304) for terminal-tool cwd; the subprocess cwd binds `AGENTS.md` resolution and the recorded session `cwd`. Neither mechanism prevents terminal commands from writing to absolute paths outside the task root, so it is insufficient for a paid adversarial benchmark.
- **Usage/cost/trajectory attribution** — `hermes_state.py` persists to `$HERMES_HOME/state.db`: the `sessions` row carries `system_prompt`, `input_tokens`, `output_tokens`, `estimated_cost_usd`, `actual_cost_usd`, `cost_status`, `cost_source`, `model`, and `cwd`; `messages` rows carry the trajectory; `session_model_usage` splits usage per model.
- **Rejected alternates** — `hermes -z/--oneshot` (`hermes_cli/oneshot.py:run_oneshot`) has a purpose-built `--usage-file` JSON report (`estimated_cost_usd`, `cost_status`, token counts, `session_id`) but no `--skills` support, and it sets `HERMES_YOLO_MODE=1`; `batch_runner.py` produces training trajectories without per-session cost attribution or isolated workspaces. `plan-batch` is therefore superseded and kept only as a non-executable record.

### Adapter behavior

In hermes-cli-stub mode, the `HermesCliInvoker` creates a fresh `HERMES_HOME` under the task's control dir, installs the digest-bound skill artifact at `skills/<name>/`, and invokes the emulated CLI contract via argv (no `shell=True`) in its own process group (timeout kills the whole tree) with a scrubbed environment. The same code contains the intended real argv/state attribution contract, but `build_live_hermes_invoker` refuses to construct a paid invoker while the probe reports structural safety blockers. After a zero-exit stub run it fails closed unless ALL of these hold:

1. a `session_id` was reported on stderr and its row exists in `state.db`;
2. the session `source` is `cli`, its recorded `cwd` resolves to the isolated task workspace, and its `model` equals the expected fingerprint model (catches silent fallback-model switches);
3. **consumption proof** — the SKILL.md body appears verbatim in the persisted session `system_prompt` (copying without loading is a task failure);
4. token counts are non-negative integers with nonzero total usage, `estimated_cost_usd` is finite/non-negative, and `cost_status`/`cost_source` prove the cost is attributable (`unknown`/`none` fails closed); strict `usage.json` then feeds the post-run accounting gate.

`control/` retains `invocation.json` (argv + env key names), `stdout.txt`/`stderr.txt`, `session.json`, `trajectory.json`, and `attestation.json` (all marked `capability_evidence=false`). Valid attributable usage is written before the remaining evidence checks, so a model/cwd/consumption failure, nonzero exit, post-session timeout, or later invoker exception still counts already-incurred spend; unknown usage halts every later task.

## Current CLI

```bash
python -m benchmarks.capability validate --suite benchmarks/capability/suites/native_v1/suite.json
python -m benchmarks.capability replay ...
python -m benchmarks.capability run-fake --suite ... --role candidate --artifact <path> \
    --model test/model --environment local --solve --output candidate.json
python -m benchmarks.capability probe-hermes --hermes-repo ~/.hermes/hermes-agent
python -m benchmarks.capability probe-live-readiness   # structural blocker status; exit 2 while blocked
python -m benchmarks.capability run-hermes-stub --suite ... --role candidate \
    --artifact <skill-dir> --model stub/model --environment stub-v1 --solve --output out.json
python -m benchmarks.capability run-hermes-live ...   # intentionally blocked; see below
python -m benchmarks.capability compare ...
python -m benchmarks.capability plan-batch ...        # superseded, non-executable record
```

`run-fake` and `run-hermes-stub` exercise the executor and adapter seams end-to-end without any paid model call; their output is harness validation, never capability evidence. The bundled `fixtures/hermes_cli_stub.py` emulates the verified CLI contract (skill resolution from `HERMES_HOME/skills`, `state.db` session/usage rows, `session_id:` on stderr, hard failure on unknown skills) so command construction, consumption proof, usage parsing, timeout/cleanup, and evidence labeling are all tested. Stub mode is pinned to the bundled script and live mode requires a passing probe with argv derived from the validated checkout — a user-supplied executable cannot claim either mode.

`run-hermes-live` is present as a stable CLI contract but intentionally returns exit 2 before launching Hermes. Even the exact confirmation phrase and positive accounting ceilings cannot bypass the structural blockers reported by `probe-hermes`. This prevents a post-hoc budget check or cwd convention from being mislabeled as hard enforcement.

## Live-readiness gate (`benchmarks/capability/live_gate.py`)

The two structural blockers are now typed, testable fail-closed contracts instead of prose. `LIVE_REQUIREMENTS` is the single source of truth: the Hermes adapter's `live_executable` gate reads its **static** blocker strings, so no runtime probe result, CLI flag, config file, or environment variable can shrink the blocker list. `probe-live-readiness` reports structured requirement status (always `live_ready=false`, exit 2, `capability_evidence=false` today) and is explicitly informational — unblocking live execution requires a reviewed code change after both requirements are genuinely satisfied.

- **Pre-spend enforcement seam (shape contract only; deliberately unsatisfiable)** — `PreSpendAttestation` validates a mechanism slug, `enforcement_point == "before-provider-call"`, a finite positive USD ceiling, verifier slug, absolute control-free evidence URI, and timezone-aware ISO-8601 timestamp. Those fields are not an authorization. `verify_pre_spend_attestation` has an explicit static fail-closed guard because provider/model, config and artifact fingerprints, approved run/task ceilings, freshness, and signatures are not yet bound. The frozen empty registry is keyed by `(verified_by, mechanism)` and has no runtime registration path. Both the context-binding guard and an independently verified implementation require reviewed code changes plus supervised validation.
- **Confinement backend seam (necessary canaries only; never readiness)** — `ConfinementBackend` is the protocol a real OS-level sandbox must implement. `probe_confinement` runs a versioned child result protocol that records actual `write+fsync` outcomes for an allowed path, a sibling escape, a symlink-resolved escape, and an independently created external temporary path; a backend cannot convert a successful escape into a pass merely by deleting the file afterward. Invalid protocol/argv/deadlines, launch failures, nonzero exits, and timeouts fail closed. A dedicated process group is killed and boundedly reaped on completion or timeout to contain same-group descendants; timeout cleanup closes parent pipe readers instead of performing an unbounded second `communicate()`. The generic stdlib probe cannot prove that a wrapper has no detached descendants or that every external path is denied, so `lifecycle_verified=false` and `confined=false` remain mandatory even when all canaries pass. The bundled `NoConfinementBackend` fails the canaries. A real backend needs a reviewed OS-specific lifecycle/deny-by-default attestation before this requirement can become satisfiable.
- **Defense in depth** — neither a structurally valid pre-spend claim nor a passing confinement canary can make `evaluate_live_requirements` ready. Even a future ready report cannot flip `live_executable`, which remains bound to the static blocker list until a reviewed code change removes it.

## Remaining blockers for a real paid paired run

The next live milestone is blocked, in order, by:

1. **Filesystem confinement** — run the Hermes process and all terminal descendants in a real sandbox/container that permits writes only to the task workspace/control paths while retaining the minimum provider network access. `cwd`/`TERMINAL_CWD` alone is not confinement. *Implemented so far:* the backend protocol and necessary multi-path write-result canaries. *Still missing:* a real backend plus OS-specific deny-by-default and detached-process lifecycle verification; the generic canary intentionally cannot satisfy readiness.
2. **Pre-spend enforcement** — add a provider/proxy/agent mechanism that rejects calls before the approved USD ceiling is crossed. A timeout, `max_turns`, and post-run `estimated_cost_usd` are useful secondary controls but are not a hard USD budget. *Implemented so far:* shape validation and explicit fail-closed guards. *Still missing:* the enforcement mechanism, execution-context/approval binding, freshness/signature validation, and an identity-bound independent verifier.
3. **Supervised contract validation** — only after 1–2 exist, perform the first bounded paid run to confirm state.db cost attribution, model pinning under fallback chains, stderr `session_id` stability, credential routing, and scrubbed-PATH tool behavior.
4. **Evidence transition** — introduce a new attested evidence schema and enable `capability_evidence=true` only for its verified adapter after those checks pass; schema v1 and every stub/replay path remain permanently false.
5. **Scale honesty** — the three-task native suite can validate the pipeline but can never support production-readiness or statistical-significance claims; add larger holdout tasks before optimizer gating.
