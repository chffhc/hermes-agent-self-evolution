"""Level 1 native capability benchmark foundation.

Defines a fail-closed task/result schema, deterministic workspace verifiers,
a replay/fixture executor for harness tests, a local isolated-workspace
executor with an injectable agent-invocation seam and post-run accounting gate,
paired baseline-vs-candidate comparison gating, and a current-Hermes CLI
adapter foundation (compatibility probe, skill-artifact injection contract,
contract-emulating stub, fail-closed live design); the older batch_runner
dry-run seam is kept only as a superseded, non-executable record.

Honesty invariant: only ``execution_mode == "live"`` may ever carry
``capability_evidence=True``. Replay, fixture, fake-agent, hermes-cli-stub,
and dry-run outputs measure the harness, not the agent, and are permanently
labeled as such — and even live runs stay ``capability_evidence=False``
until the validation blockers in docs/CAPABILITY_BENCHMARK.md are cleared.
"""

SCHEMA_VERSION = 1
