"""Structural live-readiness gate for the capability benchmark.

Paid live execution is blocked by two structural requirements (see
docs/CAPABILITY_BENCHMARK.md). This module turns them from prose into
testable fail-closed contracts:

- **Pre-spend USD enforcement** — :class:`PreSpendAttestation` validates only
  claim shape. Verification is guarded off until execution-context, approval,
  freshness, signature, and identity-bound verifier checks exist; the verifier
  registry is also permanently empty at runtime.
- **Filesystem confinement** — :class:`ConfinementBackend` is the seam a real
  OS-level sandbox must implement. :func:`probe_confinement` runs necessary
  multi-path write canaries, but deliberately cannot certify detached-process
  lifecycle or a universal deny-by-default policy. Therefore a canary pass is
  never itself confinement readiness. The bundled :class:`NoConfinementBackend`
  fails even the canaries.

Honesty contract: :func:`evaluate_live_requirements` is informational. Its
report can never flip ``live_executable`` — the Hermes adapter derives its
blockers from the static :func:`structural_live_blockers` list, so unblocking
live execution requires reviewed code changes after both requirements are
independently attested. No runtime probe, CLI flag, config file, or environment
variable can satisfy either requirement today.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Protocol
from urllib.parse import urlsplit

from benchmarks.capability.schema import SchemaError, utc_now_iso

PRE_SPEND_REQUIREMENT_ID = "pre-spend-usd-enforcement"
CONFINEMENT_REQUIREMENT_ID = "filesystem-confinement"

PRE_SPEND_ENFORCEMENT_POINT = "before-provider-call"

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


@dataclass(frozen=True)
class LiveRequirement:
    """One structural prerequisite for paid live execution."""

    requirement_id: str
    title: str
    blocker: str
    verification: str

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "title": self.title,
            "blocker": self.blocker,
            "verification": self.verification,
        }


# Single source of truth for the structural blockers. The Hermes adapter's
# live_executable gate reads these static strings; nothing evaluated at
# runtime can shrink this tuple.
LIVE_REQUIREMENTS: tuple[LiveRequirement, ...] = (
    LiveRequirement(
        requirement_id=PRE_SPEND_REQUIREMENT_ID,
        title="Pre-spend USD enforcement",
        blocker=(
            "current Hermes has no enforceable pre-spend USD ceiling; "
            "state.db cost is post-run accounting"
        ),
        verification=(
            "a provider/proxy/agent mechanism must reject calls before the approved "
            "USD ceiling is crossed, attested via PreSpendAttestation and verified by "
            "an independent registered verifier (registry is empty; registering one "
            "requires a reviewed code change plus supervised validation)"
        ),
    ),
    LiveRequirement(
        requirement_id=CONFINEMENT_REQUIREMENT_ID,
        title="OS-level filesystem confinement",
        blocker=(
            "TERMINAL_CWD pins the default cwd but does not sandbox terminal writes "
            "outside the task workspace"
        ),
        verification=(
            "a ConfinementBackend must pass necessary multi-path write canaries and "
            "separate reviewed OS-specific deny-by-default and detached-process lifecycle "
            "validation; the generic probe cannot satisfy this requirement"
        ),
    ),
)


def structural_live_blockers() -> tuple[str, ...]:
    """Static blocker strings consumed by the Hermes adapter's live gate."""
    return tuple(req.blocker for req in LIVE_REQUIREMENTS)


# ── pre-spend enforcement attestation seam ──


@dataclass(frozen=True)
class PreSpendAttestation:
    """What a real pre-spend USD enforcement mechanism must attest.

    This is shape validation only, not authorization. Verification remains
    statically blocked until execution-context, approval, freshness, signature,
    and identity-bound verifier checks are implemented by reviewed code.
    """

    mechanism: str
    enforcement_point: str
    max_usd: float
    verified_by: str
    evidence_uri: str
    verified_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.mechanism, str) or not _SLUG_RE.match(self.mechanism):
            raise SchemaError(
                f"pre-spend attestation: mechanism must match {_SLUG_RE.pattern}, "
                f"got {self.mechanism!r}"
            )
        if self.enforcement_point != PRE_SPEND_ENFORCEMENT_POINT:
            raise SchemaError(
                "pre-spend attestation: enforcement_point must be "
                f"{PRE_SPEND_ENFORCEMENT_POINT!r} — post-run accounting is not "
                f"enforcement (got {self.enforcement_point!r})"
            )
        usd = self.max_usd
        if isinstance(usd, bool) or not isinstance(usd, (int, float)):
            raise SchemaError("pre-spend attestation: max_usd must be a number")
        if not math.isfinite(float(usd)) or float(usd) <= 0:
            raise SchemaError("pre-spend attestation: max_usd must be finite and > 0")
        for name in ("verified_by", "evidence_uri", "verified_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SchemaError(f"pre-spend attestation: {name} must be a non-empty string")
        if not _SLUG_RE.match(self.verified_by):
            raise SchemaError(f"pre-spend attestation: verified_by must match {_SLUG_RE.pattern}")
        uri = urlsplit(self.evidence_uri)
        if (
            self.evidence_uri != self.evidence_uri.strip()
            or any(ord(char) < 32 for char in self.evidence_uri)
            or not uri.scheme
            or not (uri.netloc or uri.path)
        ):
            raise SchemaError(
                "pre-spend attestation: evidence_uri must be an absolute, control-free URI"
            )
        try:
            verified_at = datetime.fromisoformat(self.verified_at)
        except ValueError as exc:
            raise SchemaError(f"pre-spend attestation: verified_at is not ISO-8601: {exc}") from exc
        if verified_at.tzinfo is None:
            raise SchemaError("pre-spend attestation: verified_at must include a timezone")


# Permanently empty at runtime: there is intentionally no registration
# function, CLI flag, or config hook. Keys bind both the independent verifier
# identity and mechanism; a future implementation must also replace the
# explicit context-binding guard below with real provider/model/config/artifact,
# approval-ceiling, freshness, and signature verification.
_PRE_SPEND_CONTEXT_BINDING_IMPLEMENTED = False
_REGISTERED_PRE_SPEND_VERIFIERS: Mapping[
    tuple[str, str], Callable[[PreSpendAttestation], str | None]
] = MappingProxyType({})


def verify_pre_spend_attestation(attestation: PreSpendAttestation) -> str | None:
    """Fail-closed verification; returns the blocking reason or None."""
    if not isinstance(attestation, PreSpendAttestation):
        raise SchemaError("pre-spend verification requires a PreSpendAttestation")
    if not _PRE_SPEND_CONTEXT_BINDING_IMPLEMENTED:
        return (
            "pre-spend attestation verification is not execution-bound: provider, model, "
            "config/artifact fingerprints, approved run/task ceilings, freshness, and "
            "signature verification are not implemented (fail closed; reviewed code "
            "change plus supervised validation required)"
        )
    verifier = _REGISTERED_PRE_SPEND_VERIFIERS.get((attestation.verified_by, attestation.mechanism))
    if verifier is None:
        return (
            f"no registered independent verifier {attestation.verified_by!r} for pre-spend "
            f"mechanism {attestation.mechanism!r} (fail closed; registering one requires "
            "a reviewed code change plus supervised validation)"
        )
    return verifier(attestation)


# ── filesystem confinement backend seam ──


class ConfinementBackend(Protocol):
    """OS-level confinement seam: wrap an argv so the child can write only
    inside ``allowed_roots`` while retaining minimal provider network access."""

    @property
    def backend_id(self) -> str: ...

    def confine(self, argv: Sequence[str], allowed_roots: Sequence[Path]) -> list[str]: ...


@dataclass(frozen=True)
class NoConfinementBackend:
    """The honest default: no confinement at all. Provably fails the probe."""

    backend_id: str = "none"

    def confine(self, argv: Sequence[str], allowed_roots: Sequence[Path]) -> list[str]:
        return list(argv)


@dataclass(frozen=True)
class ConfinementProbeResult:
    backend_id: str
    canary_passed: bool
    detail: str
    checked_at: str
    lifecycle_verified: bool = False

    @property
    def confined(self) -> bool:
        """True only after both canaries and backend lifecycle validation.

        The generic stdlib probe cannot prove that a wrapper has no detached
        descendants, so it deliberately leaves lifecycle_verified false.
        """
        return self.canary_passed and self.lifecycle_verified

    def to_dict(self) -> dict[str, object]:
        return {
            "backend_id": self.backend_id,
            "canary_passed": self.canary_passed,
            "lifecycle_verified": self.lifecycle_verified,
            "confined": self.confined,
            "detail": self.detail,
            "checked_at": self.checked_at,
            "capability_evidence": False,
        }


# The child reports whether each write+fsync actually succeeded. Parent-side
# final-state checks are secondary: a backend cannot hide a successful escape
# merely by deleting the canary before it exits.
_PROBE_CHILD_SOURCE = """\
import json
import os
import sys
labels = ("inside", "sibling", "symlink", "external")
writes = []
for label, target in zip(labels, sys.argv[1:5], strict=True):
    success = False
    error = None
    try:
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("confinement-probe\\n")
            handle.flush()
            os.fsync(handle.fileno())
        success = True
    except OSError as exc:
        error = type(exc).__name__
    writes.append({"label": label, "success": success, "error": error})
print(json.dumps({"probe_version": 1, "writes": writes}, sort_keys=True, separators=(",", ":")))
"""


def _terminate_same_process_group(
    proc: subprocess.Popen[str], timeout_seconds: float = 0.25
) -> bool:
    """Kill/reap the direct child, then confirm its process group is gone."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        return False

    if proc.poll() is None:
        try:
            proc.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                return False

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.01)
    return False


def probe_confinement(
    backend: ConfinementBackend, *, timeout_seconds: float = 30.0
) -> ConfinementProbeResult:
    """Run necessary write canaries against a trusted backend.

    The child reports actual write+fsync outcomes, while parent-side file checks
    detect contradictions. A canary pass is still not confinement: this generic
    probe cannot verify detached-process lifecycle or every external path, so
    ``lifecycle_verified`` remains false and ``confined`` cannot become true.
    """
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise SchemaError("confinement probe timeout_seconds must be finite and > 0")
    backend_id = getattr(backend, "backend_id", None)
    if not isinstance(backend_id, str) or not _SLUG_RE.match(backend_id):
        raise SchemaError(f"confinement backend_id must match {_SLUG_RE.pattern}")
    root = Path(tempfile.mkdtemp(prefix="confinement-probe-"))
    external_root: Path | None = None
    try:
        external_root = Path(tempfile.mkdtemp(prefix="confinement-external-canary-"))
        allowed = root / "allowed"
        forbidden = root / "forbidden"
        allowed.mkdir()
        forbidden.mkdir()
        inside = allowed / "inside.txt"
        escape = forbidden / "escape.txt"
        symlink_dir = allowed / "escape-link"
        try:
            symlink_dir.symlink_to(forbidden, target_is_directory=True)
        except OSError as exc:
            return ConfinementProbeResult(
                backend_id=backend_id,
                canary_passed=False,
                detail=f"could not create symlink-escape fixture: {type(exc).__name__}: {exc} (fail closed)",
                checked_at=utc_now_iso(),
            )
        symlink_escape = symlink_dir / "symlink-escape.txt"
        external_escape = external_root / "external-escape.txt"
        child_argv = [
            sys.executable,
            "-I",
            "-c",
            _PROBE_CHILD_SOURCE,
            str(inside),
            str(escape),
            str(symlink_escape),
            str(external_escape),
        ]
        argv = backend.confine(child_argv, (allowed,))
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(a, str) and a and "\x00" not in a for a in argv)
        ):
            raise SchemaError(
                f"confinement backend {backend_id!r} returned an invalid argv (fail closed)"
            )
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(allowed),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            return ConfinementProbeResult(
                backend_id=backend_id,
                canary_passed=False,
                detail=f"probe child failed to run: {type(exc).__name__}: {exc} (fail closed)",
                checked_at=utc_now_iso(),
            )
        timed_out: subprocess.TimeoutExpired | None = None
        communication_error: OSError | None = None
        stdout = ""
        stderr = ""
        try:
            stdout, stderr = proc.communicate(timeout=float(timeout_seconds))
        except subprocess.TimeoutExpired as exc:
            # Do not call communicate() again: a detached descendant may still
            # own the inherited pipe writers and keep EOF open forever.
            timed_out = exc
        except OSError as exc:
            communication_error = exc
        finally:
            # Same-group descendants must be gone before filesystem inspection.
            # Detached descendants remain intentionally unverified, and parent
            # pipe readers are closed instead of waiting for their EOF.
            group_quiesced = _terminate_same_process_group(proc)
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            direct_process_quiesced = proc.poll() is not None
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
        if not group_quiesced or not direct_process_quiesced:
            return ConfinementProbeResult(
                backend_id=backend_id,
                canary_passed=False,
                detail="probe process group/direct child did not quiesce after SIGKILL (fail closed)",
                checked_at=utc_now_iso(),
            )
        if timed_out is not None:
            return ConfinementProbeResult(
                backend_id=backend_id,
                canary_passed=False,
                detail=(
                    f"probe child timed out: {type(timed_out).__name__}: {timed_out} "
                    "(fail closed)"
                ),
                checked_at=utc_now_iso(),
            )
        if communication_error is not None:
            return ConfinementProbeResult(
                backend_id=backend_id,
                canary_passed=False,
                detail=(
                    "probe child communication failed: "
                    f"{type(communication_error).__name__}: {communication_error} (fail closed)"
                ),
                checked_at=utc_now_iso(),
            )
        if proc.returncode != 0:
            detail = (stderr or stdout or "").strip()[-200:]
            return ConfinementProbeResult(
                backend_id=backend_id,
                canary_passed=False,
                detail=f"probe child exited {proc.returncode}: {detail} (fail closed)",
                checked_at=utc_now_iso(),
            )
        try:
            payload = json.loads(stdout)
            writes = payload["writes"]
            if set(payload) != {"probe_version", "writes"} or payload["probe_version"] != 1:
                raise ValueError("unexpected probe envelope")
            if not isinstance(writes, list) or len(writes) != 4:
                raise ValueError("expected four write results")
            expected_labels = ("inside", "sibling", "symlink", "external")
            outcomes: dict[str, bool] = {}
            for expected, item in zip(expected_labels, writes, strict=True):
                if (
                    not isinstance(item, dict)
                    or set(item) != {"label", "success", "error"}
                    or item.get("label") != expected
                    or not isinstance(item.get("success"), bool)
                    or (item.get("error") is not None and not isinstance(item.get("error"), str))
                ):
                    raise ValueError(f"invalid write result for {expected}")
                outcomes[expected] = item["success"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return ConfinementProbeResult(
                backend_id=backend_id,
                canary_passed=False,
                detail=f"probe child returned invalid result protocol: {exc} (fail closed)",
                checked_at=utc_now_iso(),
            )
        if not outcomes["inside"] or not inside.is_file():
            return ConfinementProbeResult(
                backend_id=backend_id,
                canary_passed=False,
                detail=(
                    "backend broke permitted writes inside the allowed root "
                    "(fail closed: confinement must not break task workspaces)"
                ),
                checked_at=utc_now_iso(),
            )
        for label, path in (
            ("sibling", escape),
            ("symlink", symlink_escape),
            ("external", external_escape),
        ):
            if outcomes[label]:
                return ConfinementProbeResult(
                    backend_id=backend_id,
                    canary_passed=False,
                    detail=(
                        f"{label} escape write+fsync succeeded; deleting its final file "
                        "cannot turn a successful escape into confinement"
                    ),
                    checked_at=utc_now_iso(),
                )
            if path.exists():
                return ConfinementProbeResult(
                    backend_id=backend_id,
                    canary_passed=False,
                    detail=f"{label} escape canary exists despite a denied result (fail closed)",
                    checked_at=utc_now_iso(),
                )
        return ConfinementProbeResult(
            backend_id=backend_id,
            canary_passed=True,
            detail=(
                "write-result protocol reports inside success and all escape canaries denied; "
                "necessary canaries passed but detached-process lifecycle is not verified"
            ),
            checked_at=utc_now_iso(),
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if external_root is not None:
            shutil.rmtree(external_root, ignore_errors=True)


# ── readiness evaluation (informational; can never unlock live) ──


@dataclass(frozen=True)
class LiveRequirementStatus:
    requirement_id: str
    title: str
    satisfied: bool
    detail: str
    blocker: str

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "title": self.title,
            "satisfied": self.satisfied,
            "detail": self.detail,
            "blocker": self.blocker,
        }


@dataclass(frozen=True)
class LiveReadinessReport:
    statuses: tuple[LiveRequirementStatus, ...]
    checked_at: str

    @property
    def live_ready(self) -> bool:
        return bool(self.statuses) and all(status.satisfied for status in self.statuses)

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(status.blocker for status in self.statuses if not status.satisfied)

    def to_dict(self) -> dict[str, object]:
        return {
            "live_ready": self.live_ready,
            "blockers": list(self.blockers),
            "requirements": [status.to_dict() for status in self.statuses],
            "checked_at": self.checked_at,
            "capability_evidence": False,
            "note": (
                "informational readiness probe; live_executable is derived from the "
                "static structural blocker list and cannot be unlocked by this report"
            ),
        }


def evaluate_live_requirements(
    *,
    pre_spend_attestation: PreSpendAttestation | None = None,
    confinement_backend: ConfinementBackend | None = None,
) -> LiveReadinessReport:
    """Evaluate both structural requirements fail-closed.

    With no arguments (the CLI path) this reports the honest current state:
    no attestation exists and the no-confinement backend fails its probe.
    """
    pre_spend_req, confinement_req = LIVE_REQUIREMENTS

    if pre_spend_attestation is None:
        pre_spend_detail = (
            "no pre-spend enforcement attestation exists; current Hermes reports "
            "cost only after the provider call (fail closed)"
        )
        pre_spend_ok = False
    else:
        failure = verify_pre_spend_attestation(pre_spend_attestation)
        pre_spend_ok = failure is None
        pre_spend_detail = failure or (
            f"pre-spend mechanism {pre_spend_attestation.mechanism!r} verified"
        )

    backend = confinement_backend if confinement_backend is not None else NoConfinementBackend()
    probe = probe_confinement(backend)

    statuses = (
        LiveRequirementStatus(
            requirement_id=pre_spend_req.requirement_id,
            title=pre_spend_req.title,
            satisfied=pre_spend_ok,
            detail=pre_spend_detail,
            blocker=pre_spend_req.blocker,
        ),
        LiveRequirementStatus(
            requirement_id=confinement_req.requirement_id,
            title=confinement_req.title,
            satisfied=probe.confined,
            detail=f"backend {probe.backend_id!r}: {probe.detail}",
            blocker=confinement_req.blocker,
        ),
    )
    return LiveReadinessReport(statuses=statuses, checked_at=utc_now_iso())
