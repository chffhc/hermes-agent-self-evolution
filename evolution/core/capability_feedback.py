"""Fail-closed consumer boundary for capability optimizer-feedback documents.

The capability benchmark (``benchmarks/capability``) emits two artifacts from a
paired run: a holdout-aware ``Comparison`` for human review, and a
development-only ``optimizer_feedback`` document that withholds holdout
identities, outcomes, counts, the full-suite gate, and full-suite metric
deltas. Only the second document may ever reach the evolution optimizer.

This module is the strict consumer-side gate for that document. It never
trusts the producer: every field is re-validated fail-closed, so a full
``Comparison.to_dict()`` payload, a holdout/oracle-shaped extra field, an
evidence-bearing document, or an internally inconsistent development section
raises :class:`CapabilityFeedbackError` instead of leaking into optimization.

Wiring status (kept honest): ``evolve(..., capability_feedback=...,
capability_suite=...)`` loads the trusted suite and binds feedback schema v2 to
its ID, hash, development task set/count, and critical-task policy before any
billable work. It prints only the validated development section and records it
in ``metrics.json``. Injecting the rendered section into GEPA's reflection
prompt is intentionally NOT wired:
``dspy.GEPA`` exposes no prompt hook beyond the metric, and rewriting the
metric contract cannot be verified without a live optimizer run. The rendered
``prompt_section()`` exists so that future wiring can pass only validated,
fixed-format text.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from evolution.core.errors import EvolutionError

FEEDBACK_VERSION = 2

# Mirrors benchmarks.capability.schema._SLUG_RE without importing the producer
# package: the consumer boundary must not depend on producer code to stay
# strict when the producer changes.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_MAX_DOCUMENT_BYTES = 1_000_000

# The producer emits exactly this note; anything else is free text that could
# smuggle unvalidated content (including holdout hints or prompt injection)
# into an optimizer context, so the note must match byte-for-byte.
EXPECTED_NOTE = (
    "Development-only optimizer feedback. Holdout identities, outcomes, counts, "
    "full-suite gate, and full-suite metric deltas are withheld for human review. "
    "Harness comparison only, never live capability evidence."
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "feedback_version",
        "suite_id",
        "suite_hash",
        "capability_evidence",
        "development",
        "holdout_outcomes_withheld",
        "note",
    }
)

_DEVELOPMENT_KEYS = frozenset(
    {
        "task_count",
        "gate_passed",
        "pass_rate_delta",
        "regressions",
        "improvements",
        "critical_regressions",
    }
)

# Top-level keys that only appear in a full Comparison.to_dict() document.
# Their presence means the caller handed us human-review comparison data
# (which carries holdout outcomes) instead of the redacted feedback document.
_COMPARISON_ONLY_TOP_LEVEL_KEYS = frozenset(
    {
        "passed_gate",
        "baseline_pass_rate",
        "candidate_pass_rate",
        "pass_rate_delta",
        "baseline_mean_score",
        "candidate_mean_score",
        "score_delta",
        "regressions",
        "improvements",
        "critical_regressions",
        "duration_delta_seconds",
        "cost_delta_usd",
    }
)

# Key-name substrings that indicate holdout/oracle/full-comparison data at
# either nesting level, regardless of the exact spelling.
_FORBIDDEN_KEY_SUBSTRINGS = ("holdout", "baseline", "candidate", "oracle")


class CapabilityFeedbackError(EvolutionError):
    """An optimizer-feedback document failed fail-closed validation."""


def _fail(message: str) -> NoReturn:
    raise CapabilityFeedbackError(f"optimizer feedback rejected: {message} (fail closed)")


@dataclass(frozen=True)
class CapabilityFeedbackPolicy:
    """Trusted suite-side policy used to authenticate a feedback document."""

    suite_id: str
    suite_hash: str
    development_task_ids: frozenset[str]
    critical_development_task_ids: frozenset[str]


@dataclass(frozen=True)
class DevelopmentFeedback:
    """Validated development-only slice of a paired capability comparison."""

    task_count: int
    gate_passed: bool
    pass_rate_delta: float
    regressions: tuple[str, ...]
    improvements: tuple[str, ...]
    critical_regressions: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityFeedback:
    """A validated development-only optimizer-feedback document."""

    suite_id: str
    suite_hash: str
    development: DevelopmentFeedback

    def to_document(self) -> dict[str, object]:
        """Re-emit the exact validated document (dev-only, non-evidence)."""
        dev = self.development
        return {
            "feedback_version": FEEDBACK_VERSION,
            "suite_id": self.suite_id,
            "suite_hash": self.suite_hash,
            "capability_evidence": False,
            "development": {
                "task_count": dev.task_count,
                "gate_passed": dev.gate_passed,
                "pass_rate_delta": dev.pass_rate_delta,
                "regressions": list(dev.regressions),
                "improvements": list(dev.improvements),
                "critical_regressions": list(dev.critical_regressions),
            },
            "holdout_outcomes_withheld": True,
            "note": EXPECTED_NOTE,
        }

    def prompt_section(self) -> str:
        """Fixed-format text safe for an optimizer context.

        Interpolates only validated fields: slug task IDs, a finite delta,
        and a positive integer count. Contains no suite fixtures, prompts,
        holdout identities, or full-comparison metrics.
        """
        dev = self.development

        def fmt(ids: tuple[str, ...]) -> str:
            return ", ".join(ids) if ids else "none"

        return "\n".join(
            [
                f"Capability harness feedback for suite '{self.suite_id}' "
                "(development slice only):",
                f"- development tasks compared: {dev.task_count}",
                f"- development gate: {'passed' if dev.gate_passed else 'FAILED'}",
                f"- development pass-rate delta: {dev.pass_rate_delta:+.4f}",
                f"- regressions: {fmt(dev.regressions)}",
                f"- critical regressions: {fmt(dev.critical_regressions)}",
                f"- improvements: {fmt(dev.improvements)}",
                "Holdout task identities and outcomes are withheld from optimization.",
                "Harness comparison only — never live agent capability evidence.",
            ]
        )


def _reject_forbidden_key_names(keys: set[str], ctx: str) -> None:
    allowed = {"holdout_outcomes_withheld"}
    flagged = sorted(
        key
        for key in keys
        if key not in allowed and any(marker in key.lower() for marker in _FORBIDDEN_KEY_SUBSTRINGS)
    )
    if flagged:
        _fail(
            f"{ctx} contains holdout/comparison-shaped fields; "
            "holdout identities, outcomes, and counts must never reach the optimizer"
        )


def _require_exact_keys(obj: dict, expected: frozenset, ctx: str) -> None:
    keys = set(obj)
    _reject_forbidden_key_names(keys, ctx)
    missing = sorted(expected - keys)
    if missing:
        _fail(f"{ctx} is missing required fields")
    unknown = sorted(keys - expected)
    if unknown:
        _fail(f"{ctx} contains unknown fields")


def _require_slug(value: object, ctx: str) -> str:
    if not isinstance(value, str):
        _fail(f"{ctx} must be a string, got {type(value).__name__}")
    if len(value) > 128 or not _SLUG_RE.match(value):
        _fail(f"{ctx} must be a short slug matching {_SLUG_RE.pattern}")
    return value


def _require_sha256(value: object, ctx: str) -> str:
    if not isinstance(value, str):
        _fail(f"{ctx} must be a lowercase SHA-256 hex digest")
    if not _SHA256_RE.fullmatch(value):
        _fail(f"{ctx} must be a lowercase SHA-256 hex digest")
    return value


def _require_task_id_list(value: object, ctx: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(f"{ctx} must be a list, got {type(value).__name__}")
    ids = tuple(_require_slug(item, f"{ctx} entry") for item in value)
    if len(ids) != len(set(ids)):
        _fail(f"{ctx} contains duplicate task IDs")
    if any(a >= b for a, b in zip(ids, ids[1:], strict=False)):
        _fail(f"{ctx} must be sorted (producer emits sorted lists)")
    return ids


def _parse_development(section: object) -> DevelopmentFeedback:
    if not isinstance(section, dict):
        _fail(f"'development' must be an object, got {type(section).__name__}")
    _require_exact_keys(section, _DEVELOPMENT_KEYS, "development section")

    task_count = section["task_count"]
    if not isinstance(task_count, int) or isinstance(task_count, bool) or task_count < 1:
        _fail("development.task_count must be a positive integer")

    gate_passed = section["gate_passed"]
    if not isinstance(gate_passed, bool):
        _fail("development.gate_passed must be a boolean")

    delta = section["pass_rate_delta"]
    if not isinstance(delta, (int, float)) or isinstance(delta, bool) or not math.isfinite(delta):
        _fail("development.pass_rate_delta must be a finite number")

    regressions = _require_task_id_list(section["regressions"], "development.regressions")
    improvements = _require_task_id_list(section["improvements"], "development.improvements")
    critical = _require_task_id_list(
        section["critical_regressions"], "development.critical_regressions"
    )

    if set(regressions) & set(improvements):
        _fail("a task appears as both regression and improvement")
    if not set(critical).issubset(regressions):
        _fail("critical regressions must be a subset of regressions")
    if len(regressions) + len(improvements) > task_count:
        _fail("regressions plus improvements exceed the development task count")

    expected_delta = (len(improvements) - len(regressions)) / task_count
    if float(delta) != expected_delta:
        _fail(
            "pass_rate_delta is inconsistent with the regression/improvement counts "
            "(possible tampered or truncated document)"
        )
    expected_gate = not critical and expected_delta >= 0
    if gate_passed != expected_gate:
        _fail("gate_passed is inconsistent with the regression lists and delta")

    return DevelopmentFeedback(
        task_count=task_count,
        gate_passed=gate_passed,
        pass_rate_delta=float(delta),
        regressions=regressions,
        improvements=improvements,
        critical_regressions=critical,
    )


def parse_optimizer_feedback(
    document: object,
    *,
    policy: CapabilityFeedbackPolicy,
) -> CapabilityFeedback:
    """Validate feedback against a trusted suite-side policy, fail closed."""
    suite_id_policy = _require_slug(policy.suite_id, "policy.suite_id")
    suite_hash_policy = _require_sha256(policy.suite_hash, "policy.suite_hash")
    known_task_ids = set(policy.development_task_ids)
    known_critical_task_ids = set(policy.critical_development_task_ids)
    if not known_task_ids:
        _fail("trusted policy must contain at least one development task")
    for task_id in known_task_ids:
        _require_slug(task_id, "policy development task ID")
    for task_id in known_critical_task_ids:
        _require_slug(task_id, "policy critical development task ID")
    if not known_critical_task_ids.issubset(known_task_ids):
        _fail("trusted policy critical task IDs must be development task IDs")

    if not isinstance(document, dict):
        _fail(f"document must be a JSON object, got {type(document).__name__}")

    comparison_markers = sorted(set(document) & _COMPARISON_ONLY_TOP_LEVEL_KEYS)
    if comparison_markers:
        _fail(
            "document carries full-comparison fields; the optimizer "
            "must consume only the development-only optimizer_feedback document, never "
            "Comparison.to_dict() output"
        )
    _require_exact_keys(document, _TOP_LEVEL_KEYS, "document")

    version = document["feedback_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != FEEDBACK_VERSION:
        _fail(f"feedback_version must be the integer {FEEDBACK_VERSION}")

    suite_id = _require_slug(document["suite_id"], "suite_id")
    if suite_id != suite_id_policy:
        _fail("suite_id does not match the trusted suite")
    suite_hash = _require_sha256(document["suite_hash"], "suite_hash")
    if suite_hash != suite_hash_policy:
        _fail("suite_hash does not match the trusted suite definition")

    if document["capability_evidence"] is not False:
        _fail(
            "capability_evidence must be false; evidence-bearing or mislabeled documents "
            "are refused"
        )
    if document["holdout_outcomes_withheld"] is not True:
        _fail("holdout_outcomes_withheld must be true; a non-redacted document is refused")
    if document["note"] != EXPECTED_NOTE:
        _fail("note must be the fixed producer disclaimer; free-text notes are refused")

    development = _parse_development(document["development"])
    if development.task_count != len(known_task_ids):
        _fail("development.task_count does not match the trusted suite development task count")

    referenced = (
        set(development.regressions)
        | set(development.improvements)
        | set(development.critical_regressions)
    )
    foreign = sorted(referenced - known_task_ids)
    if foreign:
        _fail("document references task IDs outside the trusted development suite")

    expected_critical = set(development.regressions) & known_critical_task_ids
    if set(development.critical_regressions) != expected_critical:
        _fail("critical regression metadata does not match the trusted suite critical task policy")

    return CapabilityFeedback(suite_id=suite_id, suite_hash=suite_hash, development=development)


def load_optimizer_feedback(
    path: str | Path,
    *,
    policy: CapabilityFeedbackPolicy,
) -> CapabilityFeedback:
    """Load and validate an optimizer-feedback JSON file fail-closed.

    Strict JSON: duplicate object keys, ``NaN``/``Infinity`` constants,
    non-object roots, and oversized files are rejected with a typed error.
    Reads are byte-bounded so the size limit also bounds memory use.
    """
    try:
        with Path(path).open("rb") as handle:
            encoded = handle.read(_MAX_DOCUMENT_BYTES + 1)
        if len(encoded) > _MAX_DOCUMENT_BYTES:
            _fail(f"feedback file exceeds {_MAX_DOCUMENT_BYTES} bytes")
        raw = encoded.decode("utf-8")
    except CapabilityFeedbackError:
        raise
    except (OSError, UnicodeDecodeError):
        raise CapabilityFeedbackError(
            "optimizer feedback rejected: feedback file could not be read (fail closed)"
        ) from None

    def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate JSON key")
            result[key] = value
        return result

    def _no_non_finite(constant: str) -> float:
        _fail(f"non-finite JSON constant {constant!r}")

    try:
        document = json.loads(
            raw, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_non_finite
        )
    except CapabilityFeedbackError:
        raise
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise CapabilityFeedbackError(
            "optimizer feedback rejected: invalid strict JSON document (fail closed)"
        ) from exc

    return parse_optimizer_feedback(document, policy=policy)
