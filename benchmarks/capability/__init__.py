"""Level 1 native capability benchmark foundation.

Defines a fail-closed task/result schema, deterministic workspace verifiers,
a replay/fixture executor for harness tests, a local isolated-workspace
executor with an injectable agent-invocation seam and post-run accounting gate,
paired baseline-vs-candidate comparison gating, a current-Hermes CLI adapter
foundation (compatibility probe, skill-artifact injection contract,
contract-emulating stub, fail-closed live design), and typed informational
live-readiness prerequisites for pre-spend enforcement and OS confinement;
the older batch_runner dry-run seam is kept only as a superseded,
non-executable record.

Honesty invariant: schema v1 refuses ``capability_evidence=True`` for every
execution mode, including externally supplied ``live`` JSON. Replay, fixture,
fake-agent, hermes-cli-stub, and dry-run outputs measure the harness, not the
agent. A future attested live-evidence format requires an explicit schema
transition after the blockers in docs/CAPABILITY_BENCHMARK.md are cleared.
"""

SCHEMA_VERSION = 1
