# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes Agent Self-Evolution uses DSPy + GEPA (Genetic-Pareto Prompt Evolution) to automatically evolve and optimize Hermes Agent's skills, tool descriptions, system prompts, and code — producing measurably better versions through reflective evolutionary search.

**No GPU training required.** Everything operates via API calls — mutating text, evaluating results, and selecting the best variants. ~$2-10 per optimization run.

## How It Works

```
Read current skill/prompt/tool ──► Generate eval dataset
                                        │
                                        ▼
                                   GEPA Optimizer ◄── Execution traces
                                        │                    ▲
                                        ▼                    │
                                   Candidate variants ──► Evaluate
                                        │
                                   Constraint gates (tests, size limits, benchmarks)
                                        │
                                        ▼
                                   Best variant ──► PR against hermes-agent
```

GEPA reads execution traces to understand *why* things fail (not just that they failed), then proposes targeted improvements. ICLR 2026 Oral, MIT licensed.

## Quick Start

```bash
# Install
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"

# Point at your hermes-agent repo
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# Evolve a skill (synthetic eval data)
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source synthetic

# Or use real session history from Claude Code, Copilot, and Hermes
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source sessiondb

# Phase 2: Evolve tool descriptions
python -m evolution.tools.evolve_tool_descriptions \
    --iterations 10

# Phase 3: Evolve a system prompt section
python -m evolution.prompts.evolve_prompt_section \
    --section MEMORY_GUIDANCE \
    --iterations 10

# Phase 4: Evolve tool code with OpenEvolve in patch-only mode
python -m evolution.code.evolve_tool_code \
    --tool file_tools \
    --engine openevolve \
    --iterations 3

# Build a qwen3.6-plus code-review harness prompt for a diff
python -m evolution.code.qwen_code_review_harness build-prompt \
    --diff-file patch.diff \
    --out review_prompt.md

# Validate the lower-model review JSON with fail-closed rules
python -m evolution.code.qwen_code_review_harness validate \
    --response-file review_response.json \
    --diff-file patch.diff

# Research fallback: Darwinian Evolver external CLI (explicit opt-in; use only in an isolated worktree)
python -m evolution.code.evolve_tool_code \
    --tool file_tools \
    --engine darwinian \
    --iterations 10

# Phase 5: fail-closed readiness report (exit 0 = ready for a live cycle)
python -m evolution.monitor.continuous_evolution --status

# Phase 5: Continuous improvement cycle (refuses to start when --status fails)
python -m evolution.monitor.continuous_evolution --cycle
```

## What It Optimizes

| Phase | Target | Engine | Status |
|-------|--------|--------|--------|
| **Phase 1** | Skill files (SKILL.md) | DSPy + GEPA | ⚠️ Prototype — proxy eval; pytest gate available with `--run-tests` |
| **Phase 2** | Tool descriptions | DSPy + GEPA | ⚠️ Prototype — proxy eval |
| **Phase 3** | System prompt sections | DSPy + GEPA | ⚠️ Prototype — proxy eval, no direct prompt_builder write-back |
| **Phase 4** | Tool implementation code | OpenEvolve (primary), Darwinian Evolver (research fallback) | ⚠️ Patch-only OpenEvolve prototype; Darwinian is explicit opt-in |
| **Phase 5** | Continuous improvement loop | Automated pipeline | 🚧 Experimental — benchmark gate is fail-closed; no autonomous PR wiring yet |
| **Review Harness** | qwen3.6-plus code review prompts + fail-closed JSON validation | Deterministic prompt/rubric/static-scan shell | ✅ Implemented |

## Code Review Harness

`evolution.code.qwen_code_review_harness` is an external skeleton for making `qwen3.6-plus` behave more like a stronger reviewer on narrow code-review tasks. It does not rely on qwen free-form judgment alone; it wraps the model with:

- deterministic added-line static scans for secrets, shell injection, eval/exec, pickle, SQL interpolation, and path traversal;
- mandatory `static_scan_dispositions` so qwen must confirm, clear as false positive, or escalate each scan hit instead of silently ignoring it;
- a severity calibration rubric that separates P0/P1/P2/P3-style impact by trust boundary, default reachability, and preconditions;
- a fixed review order: security → correctness → compatibility → error handling → concurrency/resources → tests → maintainability;
- mechanism self-check reminders for common low-model mistakes such as `except Exception` vs `SystemExit` and `git checkout` branch pollution vs silent data loss;
- injection hardening that treats the diff as data only;
- a strict JSON schema for blockers and suggestions, including `severity_rationale`, `trust_boundary`, `preconditions`, and `confidence` on high/critical blockers;
- fail-closed validation that flips `passed=false` whenever security or logic blockers exist, static-scan hits are confirmed/unreviewed, required fields are missing, evidence is not grounded in the reviewed diff, or output is not trustworthy JSON.

Typical flow:

```bash
git diff > patch.diff
python -m evolution.code.qwen_code_review_harness build-prompt \
    --diff-file patch.diff \
    --out review_prompt.md
# Send review_prompt.md to qwen3.6-plus, save its JSON as review_response.json
python -m evolution.code.qwen_code_review_harness validate \
    --response-file review_response.json \
    --diff-file patch.diff
```

## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| **[OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)** | Code evolution from an initial program + evaluator; used in isolated scratch/worktree mode and returns patch artifacts only | Apache-2.0 |
| **[Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver)** | Code evolution with Git-based organisms; research fallback only | AGPL v3 (external CLI only) |

## Guardrails and current limitations

Every evolved variant is intended to pass these gates before deployment, but the
project is still a prototype. Treat generated scores as proxy signals until a
real Hermes `batch_runner`/independent-judge evaluation is connected.

Current hardening status:
1. **Full test suite** — Phase 1 can run a pytest gate with `--run-tests`; CI now fails on lint/test failures.
2. **Size limits** — Skills and tool descriptions are checked before output is accepted.
3. **Benchmark gate** — benchmark errors/regressions are fail-closed by default. Runner discovery: an explicit `runner_path`/`EVOLUTION_BENCH_RUNNER` override wins (and fails closed if the configured path is missing), then hermes-agent's `environments/benchmarks/run_bench.py`, then this repo's `benchmarks/run_bench.py` smoke runner. The smoke runner executes real deterministic checks (Python syntax over the target repo, skill-override validation) and its results are labeled `[smoke]` so smoke scores never share a baseline namespace with real TBLite/YC-Bench scores — smoke evidence is a proxy, not a capability benchmark.
4. **Code evolution isolation** — the default Phase 4 engine is OpenEvolve patch-only; Darwinian Evolver is explicit opt-in and may mutate a checkout in place.
5. **PR review** — Phase 1 (`evolve_skill`) can opt into PR generation with `--create-pr` (or `--pr-dry-run` to render the redacted PR preview with zero git/GitHub side effects). No PR is ever created by default, gate-failing or non-improving runs are refused, and PRBuilder still enforces clean-worktree + secret-redaction rules; created PRs require human review. Other phases still emit artifacts only — review those manually.
6. **Phase 5 readiness gate** — a live continuous cycle refuses to start unless the fail-closed `--status` checks pass (resolvable hermes-agent repo, a discoverable benchmark runner — the report labels it when only the local smoke runner is available, hard USD budget configured, writable output dir); dry runs are exempt, and `--skip-readiness-check` is an explicit override.

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
