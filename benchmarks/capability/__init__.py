"""Level 1 native capability benchmark foundation.

Defines a fail-closed task/result schema, deterministic workspace verifiers,
a replay/fixture executor for harness tests, paired baseline-vs-candidate
comparison gating, and a dry-run seam onto hermes-agent's batch_runner.

Honesty invariant: only ``execution_mode == "live"`` may ever carry
``capability_evidence=True``. Replay, fixture, and dry-run outputs measure
the harness, not the agent, and are permanently labeled as such.
"""

SCHEMA_VERSION = 1
