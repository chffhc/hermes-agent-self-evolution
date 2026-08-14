"""Build and verify a wheel from a clean, read-only source snapshot.

A normal setuptools rebuild can reuse project-local ``build/lib`` and
``*.egg-info`` staging, leaking removed modules or bytecode into a wheel. This
entrypoint never deletes or reuses repository staging. It snapshots only the
explicit project inputs into a temporary directory, verifies the source did
not change while copied, builds in that private snapshot, validates the wheel
archive and RECORD, and publishes with no-clobber directory-relative syscalls.

Each successful publication also writes an unsigned machine-readable build
receipt (``<wheel>.receipt.json``). It records the verified-at-build-time source
manifest alongside the wheel digest, while explicitly disclaiming
authentication, byte-for-byte reproducibility, hermeticity, and capability
evidence. Its standalone verifier checks structure and digest consistency, not
authorship or historical source provenance.

Arbitrary third-party ``pip wheel .`` invocations that bypass this entrypoint
remain outside its stale-staging isolation guarantee. This workflow does not
claim byte-for-byte or environment-independent reproducibility.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import platform
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.parser import Parser
from pathlib import Path
from zipfile import BadZipFile, ZipFile, ZipInfo

try:
    from packaging.tags import Tag, parse_tag
    from packaging.utils import InvalidWheelFilename, parse_wheel_filename
except ModuleNotFoundError as exc:  # stable direct-script diagnostic before argparse starts
    raise SystemExit(
        "error: missing required dependency 'packaging>=23'; install project dependencies first"
    ) from exc

PROJECT_NAME = "hermes-agent-self-evolution"
DIST_NAME = "hermes_agent_self_evolution"
_SOURCE_FILES = ("pyproject.toml", "README.md", "MANIFEST.in")
_SOURCE_DIRECTORIES = ("benchmarks", "evolution")
_MAX_WHEEL_ENTRIES = 10_000
_MAX_WHEEL_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
_MAX_METADATA_BYTES = 1024 * 1024
_BUILD_TIMEOUT_SECONDS = 300.0
_CLEANUP_TIMEOUT_SECONDS = 5.0
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{number}" for number in range(1, 10)}
    | {f"lpt{number}" for number in range(1, 10)}
)

RECEIPT_SUFFIX = ".receipt.json"
_RECEIPT_SCHEMA = "hermes-wheel-build-receipt-v1"
_MAX_RECEIPT_BYTES = 1024 * 1024
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00")
# Fixed v1 assertion set.  The positive values are explicitly workflow
# assertions, not facts authenticated by the standalone receipt verifier.
_RECEIPT_CLAIMS = {
    "workflow_asserted_stale_staging_isolated": True,
    "workflow_asserted_source_snapshot_unchanged": True,
    "workflow_asserted_wheel_archive_verified": True,
    "receipt_authenticated": False,
    "byte_for_byte_reproducible": False,
    "environment_independent": False,
    "hermetic_build": False,
    "capability_evidence": False,
}
_RECEIPT_FIELDS = frozenset(
    {"schema", "project", "created_utc", "wheel", "source_snapshot", "toolchain", "claims"}
)

REQUIRED_WHEEL_ENTRIES = (
    "benchmarks/capability/suite.py",
    "benchmarks/capability/fixtures/fake_agent.py",
    "benchmarks/capability/fixtures/hermes_cli_stub.py",
    "benchmarks/capability/suites/native_v1/suite.json",
    "benchmarks/capability/suites/native_v1/tasks/repair-calculator/workspace/calculator.py",
    "evolution/core/capability_feedback.py",
)
_ALLOWED_TOP_LEVEL_PACKAGES = frozenset({"benchmarks", "evolution"})


class BuildWheelError(RuntimeError):
    """The snapshot/build/verification workflow aborted without a valid publication."""


@dataclass(frozen=True)
class _SourceRecord:
    relative_path: Path
    size: int
    digest: str
    device: int
    inode: int
    mtime_ns: int


def _is_cache_path(relative: Path) -> bool:
    return any(part == "__pycache__" for part in relative.parts) or relative.name.endswith(
        (".pyc", ".pyo")
    )


def _hash_regular_file(path: Path) -> tuple[str, os.stat_result]:
    try:
        initial = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(initial.st_mode):
            raise BuildWheelError(f"source input is not a regular file: {path}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as stream:
            opened_before = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_before.st_mode):
                raise BuildWheelError(f"source input is not a regular file: {path}")
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            opened_after = os.fstat(stream.fileno())
        final = path.stat(follow_symlinks=False)
    except BuildWheelError:
        raise
    except OSError as exc:
        raise BuildWheelError(f"cannot read source input: {path}") from exc
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (initial, opened_before, opened_after, final)
    }
    if len(identities) != 1:
        raise BuildWheelError(f"source input changed while hashing: {path}")
    return digest.hexdigest(), final


def _enumerate_source(project_root: Path) -> list[_SourceRecord]:
    candidates: list[tuple[Path, Path]] = []
    for name in _SOURCE_FILES:
        candidates.append((project_root / name, Path(name)))
    for directory_name in _SOURCE_DIRECTORIES:
        directory = project_root / directory_name
        try:
            directory_stat = directory.stat(follow_symlinks=False)
        except OSError as exc:
            raise BuildWheelError(f"missing source directory: {directory}") from exc
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise BuildWheelError(f"source directory is not a regular directory: {directory}")
        try:
            descendants = sorted(directory.rglob("*"))
        except OSError as exc:
            raise BuildWheelError(f"cannot enumerate source directory: {directory}") from exc
        for path in descendants:
            relative = path.relative_to(project_root)
            if _is_cache_path(relative):
                continue
            try:
                item_stat = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise BuildWheelError(f"cannot inspect source input: {path}") from exc
            if stat.S_ISLNK(item_stat.st_mode):
                raise BuildWheelError(f"source snapshot refuses symlink: {path}")
            if stat.S_ISDIR(item_stat.st_mode):
                continue
            if not stat.S_ISREG(item_stat.st_mode):
                raise BuildWheelError(f"source snapshot refuses non-regular input: {path}")
            candidates.append((path, relative))

    records: list[_SourceRecord] = []
    for source, relative in candidates:
        try:
            source_stat = source.stat(follow_symlinks=False)
        except OSError as exc:
            raise BuildWheelError(f"missing source input: {source}") from exc
        if stat.S_ISLNK(source_stat.st_mode):
            raise BuildWheelError(f"source snapshot refuses symlink: {source}")
        digest, stable_stat = _hash_regular_file(source)
        records.append(
            _SourceRecord(
                relative_path=relative,
                size=stable_stat.st_size,
                digest=digest,
                device=stable_stat.st_dev,
                inode=stable_stat.st_ino,
                mtime_ns=stable_stat.st_mtime_ns,
            )
        )
    return records


def create_source_snapshot(project_root: Path, snapshot_root: Path) -> list[_SourceRecord]:
    """Copy explicit source inputs and fail if any input changes during the snapshot."""
    records = _enumerate_source(project_root)
    try:
        snapshot_root.mkdir(parents=True, exist_ok=False)
        for record in records:
            source = project_root / record.relative_path
            destination = snapshot_root / record.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination, follow_symlinks=False)
    except OSError as exc:
        raise BuildWheelError("could not create private source snapshot") from exc

    current_records = _enumerate_source(project_root)
    if current_records != records:
        raise BuildWheelError("source input set or content changed during snapshot")

    for record in records:
        copied_digest, copied = _hash_regular_file(snapshot_root / record.relative_path)
        if copied.st_size != record.size or copied_digest != record.digest:
            raise BuildWheelError(f"private snapshot verification failed: {record.relative_path}")
    return records


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - exercised on Windows runners
                process.kill()
        except (ProcessLookupError, PermissionError):
            pass
    elif os.name == "posix":
        # The leader may have exited while same-group descendants retain pipes.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# Process-table scan the detached-descendant sweep depends on. Exposed as a
# constant so tests can probe the exact mechanism and skip where the
# environment denies it (e.g. sandboxes that forbid process introspection).
# The sweep remains best-effort: hermeticity is not claimed.
_PROCESS_TABLE_SCAN_ARGV: tuple[str, ...] = ("ps", "eww", "-axo", "pid=,command=")


def _token_process_ids(token: str) -> set[int]:
    """Find same-user POSIX processes that still inherit this private build token."""
    if os.name != "posix":
        return set()
    marker = f"HERMES_WHEEL_BUILD_PROCESS_TOKEN={token}"
    try:
        completed = subprocess.run(
            list(_PROCESS_TABLE_SCAN_ARGV),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    process_ids: set[int] = set()
    for line in completed.stdout.splitlines():
        pid_text, separator, command = line.strip().partition(" ")
        if separator and marker in command:
            try:
                process_ids.add(int(pid_text))
            except ValueError:
                continue
    return process_ids


def _terminate_build_processes(process: subprocess.Popen[str], token: str) -> None:
    """Kill the build group plus detached descendants that retained the build token."""
    for process_id in _token_process_ids(token):
        if process_id == os.getpid():
            continue
        try:
            os.kill(process_id, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    _terminate_process_group(process)


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _run_process(command: Sequence[str], *, cwd: Path, timeout: float) -> tuple[int, str, str]:
    token = secrets.token_hex(24)
    environment = os.environ.copy()
    environment["HERMES_WHEEL_BUILD_PROCESS_TOKEN"] = token
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        raise BuildWheelError("wheel build process could not start") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_build_processes(process, token)
        try:
            process.communicate(timeout=_CLEANUP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _close_process_pipes(process)
            try:
                process.wait(timeout=_CLEANUP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        raise BuildWheelError("wheel build timed out and its process tree was terminated") from exc
    finally:
        _terminate_build_processes(process, token)
    return process.returncode, stdout, stderr


def build_wheel(snapshot_root: Path, private_wheel_dir: Path) -> Path:
    """Build one wheel inside private snapshot-owned output."""
    try:
        private_wheel_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise BuildWheelError("could not create private wheel output") from exc
    returncode, stdout, stderr = _run_process(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(private_wheel_dir),
        ],
        cwd=snapshot_root,
        timeout=_BUILD_TIMEOUT_SECONDS,
    )
    if returncode != 0:
        raise BuildWheelError(f"wheel build failed:\n{stdout}\n{stderr}")
    wheels = sorted(private_wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise BuildWheelError(f"expected exactly one wheel in private staging, found {len(wheels)}")
    _parse_wheel_identity(wheels[0])
    return wheels[0]


def _canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_wheel_identity(wheel_path: Path) -> tuple[str, frozenset[Tag]]:
    try:
        distribution, version, _build, tags = parse_wheel_filename(wheel_path.name)
    except InvalidWheelFilename as exc:
        raise BuildWheelError(f"invalid wheel filename: {wheel_path.name}") from exc
    if _canonical_distribution(distribution) != _canonical_distribution(PROJECT_NAME):
        raise BuildWheelError(f"unexpected wheel distribution: {distribution}")
    return str(version), tags


def _validate_archive_path(name: str) -> tuple[str, str | None]:
    if not name or name.endswith("/") or name.startswith("/") or "\\" in name:
        return "", f"unsafe archive path: {name!r}"
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        return "", f"control character in archive path: {name!r}"
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return "", f"unsafe archive path: {name!r}"
    for part in parts:
        if part.endswith((" ", ".")) or ":" in part:
            return "", f"platform-ambiguous archive path: {name!r}"
        stem = part.split(".", 1)[0].casefold()
        if stem in _WINDOWS_RESERVED:
            return "", f"Windows-reserved archive path: {name!r}"
    canonical = "/".join(unicodedata.normalize("NFC", part).casefold() for part in parts)
    return canonical, None


def _read_member(archive: ZipFile, info: ZipInfo) -> bytes:
    if info.file_size > _MAX_WHEEL_UNCOMPRESSED_BYTES:
        raise BuildWheelError(f"wheel member exceeds size limit: {info.filename}")
    try:
        data = archive.read(info)
    except (BadZipFile, OSError, RuntimeError) as exc:
        raise BuildWheelError(f"wheel member failed CRC/decompression: {info.filename}") from exc
    if len(data) != info.file_size:
        raise BuildWheelError(f"wheel member size mismatch: {info.filename}")
    return data


def _verify_record(record_path: str, record_bytes: bytes, contents: dict[str, bytes]) -> None:
    try:
        rows = list(csv.reader(io.StringIO(record_bytes.decode("utf-8", errors="strict"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise BuildWheelError("wheel RECORD is not strict UTF-8 CSV") from exc
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3 or row[0] in seen:
            raise BuildWheelError("wheel RECORD contains malformed or duplicate rows")
        path, hash_field, size_field = row
        seen.add(path)
        if path not in contents:
            raise BuildWheelError(f"wheel RECORD references missing member: {path}")
        if path == record_path:
            if hash_field or size_field:
                raise BuildWheelError("wheel RECORD self-row must omit hash and size")
            continue
        expected_hash = base64.urlsafe_b64encode(hashlib.sha256(contents[path]).digest()).rstrip(
            b"="
        )
        if hash_field != f"sha256={expected_hash.decode('ascii')}":
            raise BuildWheelError(f"wheel RECORD hash mismatch: {path}")
        if size_field != str(len(contents[path])):
            raise BuildWheelError(f"wheel RECORD size mismatch: {path}")
    if seen != set(contents):
        missing = sorted(set(contents) - seen)
        raise BuildWheelError(f"wheel RECORD omits archive members: {missing}")


def verify_wheel(wheel_path: Path) -> None:
    """Verify archive identity, cross-platform paths, metadata, CRCs, and RECORD."""
    version, filename_tags = _parse_wheel_identity(wheel_path)
    expected_dist_info = f"{DIST_NAME}-{version}.dist-info"

    try:
        with ZipFile(wheel_path) as archive:
            infos = archive.infolist()
            if len(infos) > _MAX_WHEEL_ENTRIES:
                raise BuildWheelError(f"wheel has too many entries: {len(infos)}")
            total_size = sum(info.file_size for info in infos)
            if total_size > _MAX_WHEEL_UNCOMPRESSED_BYTES:
                raise BuildWheelError(f"wheel uncompressed size exceeds limit: {total_size}")
            contents = {info.filename: _read_member(archive, info) for info in infos}
    except BuildWheelError:
        raise
    except (BadZipFile, OSError) as exc:
        raise BuildWheelError(f"wheel verification could not read {wheel_path.name}") from exc

    names = [info.filename for info in infos]
    problems: list[str] = []
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        problems.append(f"duplicate archive entries: {duplicates}")

    canonical_paths: dict[str, str] = {}
    for info in infos:
        name = info.filename
        canonical, path_problem = _validate_archive_path(name)
        if path_problem:
            problems.append(path_problem)
            continue
        prior = canonical_paths.get(canonical)
        if prior is not None and prior != name:
            problems.append(f"cross-platform path alias: {prior!r} and {name!r}")
        canonical_paths[canonical] = name
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            problems.append(f"symlink archive entry must not ship: {name}")
        parts = name.split("/")
        if "__pycache__" in parts:
            problems.append(f"cache entry must not ship: {name}")
        if name.endswith((".pyc", ".pyo")):
            problems.append(f"bytecode entry must not ship: {name}")
        top = parts[0]
        if top not in _ALLOWED_TOP_LEVEL_PACKAGES and top != expected_dist_info:
            problems.append(f"unexpected top-level entry: {name}")

    for canonical, original in canonical_paths.items():
        parts = canonical.split("/")
        for index in range(1, len(parts)):
            prefix = "/".join(parts[:index])
            if prefix in canonical_paths:
                problems.append(
                    f"file/directory archive path collision: "
                    f"{canonical_paths[prefix]!r} and {original!r}"
                )

    present = set(names)
    for required in REQUIRED_WHEEL_ENTRIES:
        if required not in present:
            problems.append(f"missing required entry: {required}")
    metadata_path = f"{expected_dist_info}/METADATA"
    wheel_metadata_path = f"{expected_dist_info}/WHEEL"
    record_path = f"{expected_dist_info}/RECORD"
    for required in (metadata_path, wheel_metadata_path, record_path):
        if required not in present:
            problems.append(f"missing required wheel metadata: {required}")
    if problems:
        raise BuildWheelError(
            f"wheel verification failed for {wheel_path.name}:\n" + "\n".join(sorted(problems))
        )

    metadata_bytes = contents[metadata_path]
    wheel_metadata_bytes = contents[wheel_metadata_path]
    if len(metadata_bytes) > _MAX_METADATA_BYTES or len(wheel_metadata_bytes) > _MAX_METADATA_BYTES:
        raise BuildWheelError("wheel metadata exceeds validation limit")
    try:
        metadata_text = metadata_bytes.decode("utf-8", errors="strict")
        wheel_metadata_text = wheel_metadata_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BuildWheelError("wheel metadata is not strict UTF-8") from exc
    metadata = Parser().parsestr(metadata_text, headersonly=True)
    wheel_metadata = Parser().parsestr(wheel_metadata_text, headersonly=True)
    if metadata.defects or wheel_metadata.defects:
        raise BuildWheelError("wheel metadata contains malformed headers")
    names_in_metadata = metadata.get_all("Name", [])
    versions_in_metadata = metadata.get_all("Version", [])
    if len(names_in_metadata) != 1 or _canonical_distribution(
        names_in_metadata[0]
    ) != _canonical_distribution(PROJECT_NAME):
        raise BuildWheelError("wheel METADATA Name does not uniquely match wheel distribution")
    if versions_in_metadata != [version]:
        raise BuildWheelError("wheel METADATA Version does not uniquely match wheel filename")

    wheel_versions = wheel_metadata.get_all("Wheel-Version", [])
    if wheel_versions != ["1.0"]:
        raise BuildWheelError("wheel WHEEL metadata must contain exactly Wheel-Version: 1.0")
    tag_values = wheel_metadata.get_all("Tag", [])
    parsed_wheel_tags: set[Tag] = set()
    for tag_value in tag_values:
        try:
            value_tags = set(parse_tag(tag_value))
        except ValueError as exc:
            raise BuildWheelError("wheel WHEEL metadata contains an invalid Tag") from exc
        if not value_tags or parsed_wheel_tags.intersection(value_tags):
            raise BuildWheelError("wheel WHEEL metadata contains duplicate Tag coverage")
        parsed_wheel_tags.update(value_tags)
    if parsed_wheel_tags != set(filename_tags):
        raise BuildWheelError("wheel WHEEL Tag fields do not match wheel filename")
    _verify_record(record_path, contents[record_path], contents)


def _manifest_digest(files: Sequence[tuple[str, str, int]]) -> str:
    digest = hashlib.sha256()
    for path, sha256, size in files:
        digest.update(f"{path}\0{sha256}\0{size}\n".encode())
    return digest.hexdigest()


def _source_manifest(records: Sequence[_SourceRecord]) -> list[tuple[str, str, int]]:
    files = sorted(
        (record.relative_path.as_posix(), record.digest, record.size) for record in records
    )
    canonical_paths: dict[str, str] = {}
    for path, _digest, _size in files:
        canonical, problem = _validate_archive_path(path)
        if problem:
            raise BuildWheelError(f"unsafe source manifest path: {path!r}")
        prior = canonical_paths.get(canonical)
        if prior is not None:
            raise BuildWheelError(f"source manifest path alias: {prior!r} and {path!r}")
        canonical_paths[canonical] = path
    return files


def _verify_snapshot_unchanged(
    snapshot_root: Path, expected_records: Sequence[_SourceRecord]
) -> None:
    """Prove the private source inputs still match after the backend returns."""
    current_records = _enumerate_source(snapshot_root)
    if _source_manifest(current_records) != _source_manifest(expected_records):
        raise BuildWheelError("private source snapshot changed during wheel build")


def _render_receipt(
    wheel_name: str, wheel_sha256: str, wheel_size: int, records: Sequence[_SourceRecord]
) -> bytes:
    files = _source_manifest(records)
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "project": PROJECT_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wheel": {"filename": wheel_name, "sha256": wheel_sha256, "size_bytes": wheel_size},
        "source_snapshot": {
            "algorithm": "sha256",
            "file_count": len(files),
            "total_size_bytes": sum(size for _, _, size in files),
            "manifest_sha256": _manifest_digest(files),
            "files": [
                {"path": path, "sha256": sha256, "size_bytes": size} for path, sha256, size in files
            ],
        },
        "toolchain": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "platform": sys.platform,
        },
        "claims": dict(_RECEIPT_CLAIMS),
    }
    return (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("ascii")


def _read_bounded_regular_file(path: Path, *, limit: int, context: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        initial = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(initial.st_mode):
            raise BuildWheelError(f"{context} is not a regular file: {path}")
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise BuildWheelError(f"{context} is not a regular file: {path}")
            if before.st_size > limit:
                raise BuildWheelError(f"{context} exceeds validation size limit")
            data = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
    except BuildWheelError:
        raise
    except OSError as exc:
        raise BuildWheelError(f"cannot read {context}: {path}") from exc
    identity_initial = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if (
        identity_initial != identity_before
        or identity_before != identity_after
        or len(data) != before.st_size
    ):
        raise BuildWheelError(f"{context} changed while being read")
    if len(data) > limit:
        raise BuildWheelError(f"{context} exceeds validation size limit")
    return data


def _load_strict_json(data: bytes, *, context: str) -> object:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BuildWheelError(f"{context} is not strict UTF-8") from exc

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise BuildWheelError(f"{context} contains duplicate JSON key: {key!r}")
            document[key] = value
        return document

    def reject_constant(value: str) -> object:
        raise BuildWheelError(f"{context} contains non-finite number: {value}")

    try:
        return json.loads(text, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    except BuildWheelError:
        raise
    except (ValueError, RecursionError) as exc:
        raise BuildWheelError(f"{context} is not valid JSON") from exc


def _require_receipt_object(
    value: object, expected: frozenset[str], context: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise BuildWheelError(
            f"build receipt {context} must contain exactly the fields {sorted(expected)}"
        )
    return value


def _require_receipt_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise BuildWheelError(f"build receipt {context} must be a lowercase hex sha256 digest")
    return value


def _require_receipt_size(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildWheelError(f"build receipt {context} must be a non-negative integer")
    return value


def verify_receipt(receipt_path: Path, wheel_path: Path) -> dict[str, object]:
    """Check receipt structure and its digest consistency with published wheel bytes.

    The receipt is unsigned and therefore does not authenticate who produced
    the wheel or prove that its manifest was the actual build input.  It does
    not inspect archive contents (``verify_wheel`` does), re-hash a checkout,
    or support a reproducibility/capability claim.
    """
    raw = _read_bounded_regular_file(
        receipt_path, limit=_MAX_RECEIPT_BYTES, context="build receipt"
    )

    document = _require_receipt_object(
        _load_strict_json(raw, context="build receipt"), _RECEIPT_FIELDS, "document"
    )
    if document["schema"] != _RECEIPT_SCHEMA:
        raise BuildWheelError(f"build receipt schema must be {_RECEIPT_SCHEMA!r}")
    if document["project"] != PROJECT_NAME:
        raise BuildWheelError(f"build receipt project must be {PROJECT_NAME!r}")
    created = document["created_utc"]
    try:
        if not isinstance(created, str) or _UTC_TIMESTAMP.fullmatch(created) is None:
            raise BuildWheelError("build receipt created_utc must use YYYY-MM-DDTHH:MM:SS+00:00")
        parsed_created = datetime.fromisoformat(created)
        if (
            parsed_created.utcoffset() != timedelta(0)
            or parsed_created.isoformat(timespec="seconds") != created
        ):
            raise BuildWheelError("build receipt created_utc must use YYYY-MM-DDTHH:MM:SS+00:00")
    except ValueError as exc:
        raise BuildWheelError(
            "build receipt created_utc must use YYYY-MM-DDTHH:MM:SS+00:00"
        ) from exc

    wheel_field = _require_receipt_object(
        document["wheel"], frozenset({"filename", "sha256", "size_bytes"}), "wheel"
    )
    if not isinstance(wheel_field["filename"], str) or wheel_field["filename"] != wheel_path.name:
        raise BuildWheelError("build receipt wheel filename does not match the wheel file")
    _parse_wheel_identity(wheel_path)
    expected_sha256 = _require_receipt_sha256(wheel_field["sha256"], "wheel.sha256")
    expected_size = _require_receipt_size(wheel_field["size_bytes"], "wheel.size_bytes")
    actual_sha256, actual_stat = _hash_regular_file(wheel_path)
    if actual_sha256 != expected_sha256 or actual_stat.st_size != expected_size:
        raise BuildWheelError("wheel bytes do not match the build receipt digest")

    snapshot = _require_receipt_object(
        document["source_snapshot"],
        frozenset({"algorithm", "file_count", "total_size_bytes", "manifest_sha256", "files"}),
        "source_snapshot",
    )
    if snapshot["algorithm"] != "sha256":
        raise BuildWheelError("build receipt source_snapshot.algorithm must be 'sha256'")
    files = snapshot["files"]
    if not isinstance(files, list) or not files:
        raise BuildWheelError("build receipt source_snapshot.files must be a non-empty list")
    manifest: list[tuple[str, str, int]] = []
    previous_path: str | None = None
    canonical_paths: dict[str, str] = {}
    for index, entry in enumerate(files):
        record = _require_receipt_object(
            entry, frozenset({"path", "sha256", "size_bytes"}), f"files[{index}]"
        )
        path = record["path"]
        if not isinstance(path, str):
            raise BuildWheelError(f"build receipt manifest path is unsafe: {path!r}")
        canonical, path_problem = _validate_archive_path(path)
        if path_problem:
            raise BuildWheelError(f"build receipt manifest path is unsafe: {path!r}")
        prior = canonical_paths.get(canonical)
        if prior is not None:
            raise BuildWheelError(f"build receipt manifest path alias: {prior!r} and {path!r}")
        canonical_paths[canonical] = path
        if previous_path is not None and path <= previous_path:
            raise BuildWheelError("build receipt manifest paths must be unique and sorted")
        previous_path = path
        manifest.append(
            (
                path,
                _require_receipt_sha256(record["sha256"], f"files[{index}].sha256"),
                _require_receipt_size(record["size_bytes"], f"files[{index}].size_bytes"),
            )
        )
    if _require_receipt_size(snapshot["file_count"], "source_snapshot.file_count") != len(manifest):
        raise BuildWheelError("build receipt file_count does not match its manifest")
    total_size = _require_receipt_size(
        snapshot["total_size_bytes"], "source_snapshot.total_size_bytes"
    )
    if total_size != sum(size for _, _, size in manifest):
        raise BuildWheelError("build receipt total_size_bytes does not match its manifest")
    recorded_manifest = _require_receipt_sha256(
        snapshot["manifest_sha256"], "source_snapshot.manifest_sha256"
    )
    if recorded_manifest != _manifest_digest(manifest):
        raise BuildWheelError("build receipt manifest_sha256 does not match its manifest entries")

    toolchain = _require_receipt_object(
        document["toolchain"], frozenset({"implementation", "platform", "python"}), "toolchain"
    )
    for key in ("implementation", "platform", "python"):
        if not isinstance(toolchain[key], str) or not toolchain[key]:
            raise BuildWheelError(f"build receipt toolchain.{key} must be a non-empty string")

    claims = _require_receipt_object(document["claims"], frozenset(_RECEIPT_CLAIMS), "claims")
    for key, expected_claim in _RECEIPT_CLAIMS.items():
        value = claims[key]
        if not isinstance(value, bool) or value is not expected_claim:
            raise BuildWheelError(f"build receipt claim {key!r} must be exactly {expected_claim}")
    return document


def _reject_existing_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            item = current.stat(follow_symlinks=False)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise BuildWheelError(f"cannot inspect output path component: {current}") from exc
        if stat.S_ISLNK(item.st_mode):
            raise BuildWheelError(f"output path contains symlink component: {current}")


def _validate_output(wheel_dir: Path) -> None:
    _reject_existing_symlink_components(wheel_dir)
    try:
        if wheel_dir.exists() and not wheel_dir.is_dir():
            raise BuildWheelError(f"refusing output directory {wheel_dir}: expected a directory")
        existing = (
            sorted(
                [
                    *wheel_dir.glob(f"{DIST_NAME}-*.whl"),
                    *wheel_dir.glob(f"{DIST_NAME}-*.whl{RECEIPT_SUFFIX}"),
                ]
            )
            if wheel_dir.is_dir()
            else []
        )
    except OSError as exc:
        raise BuildWheelError(f"cannot inspect output directory: {wheel_dir}") from exc
    if existing:
        raise BuildWheelError(
            f"refusing output directory {wheel_dir}: project wheel or build receipt already "
            "exists; remove it or choose a fresh --wheel-dir"
        )


def _same_directory(path: Path, directory_fd: int) -> bool:
    try:
        path_stat = path.stat(follow_symlinks=False)
        fd_stat = os.fstat(directory_fd)
    except OSError:
        return False
    return stat.S_ISDIR(path_stat.st_mode) and (path_stat.st_dev, path_stat.st_ino) == (
        fd_stat.st_dev,
        fd_stat.st_ino,
    )


def _open_output_directory(wheel_dir: Path) -> int:
    """Create/open an absolute output path component-wise without following symlinks."""
    if os.name != "posix":  # pragma: no cover - fail closed on unsupported platforms
        raise BuildWheelError("secure wheel publication requires POSIX dirfd support")
    absolute = wheel_dir.absolute()
    if any(part in {".", ".."} for part in absolute.parts):
        raise BuildWheelError(f"output path contains ambiguous components: {wheel_dir}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        directory_fd = os.open(absolute.anchor, flags)
    except OSError as exc:
        raise BuildWheelError(f"cannot open output path root: {absolute.anchor}") from exc
    try:
        for part in absolute.parts[1:]:
            try:
                next_fd = os.open(part, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=directory_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(part, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
    except OSError as exc:
        os.close(directory_fd)
        raise BuildWheelError(f"cannot securely create output directory: {wheel_dir}") from exc
    return directory_fd


@dataclass(frozen=True)
class _PublishItem:
    source: Path
    destination_name: str
    sha256: str | None = None


def _copy_into_temporary(item: _PublishItem, directory_fd: int, temporary_name: str) -> None:
    temporary_fd = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=directory_fd,
    )
    digest = hashlib.sha256()
    try:
        with item.source.open("rb") as source_stream, os.fdopen(temporary_fd, "wb") as output:
            for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        try:
            os.close(temporary_fd)
        except OSError:
            pass
        raise
    if item.sha256 is not None and digest.hexdigest() != item.sha256:
        raise BuildWheelError(
            f"published bytes do not match verified digest: {item.destination_name}"
        )


def _atomic_publish_all(items: Sequence[_PublishItem], wheel_dir: Path) -> list[Path]:
    """No-clobber publish a rollback-safe ordered set of items."""
    if len({item.destination_name for item in items}) != len(items):
        raise BuildWheelError("publication items must have unique destination names")
    directory_fd = _open_output_directory(wheel_dir)
    temporary_names = [f".{item.destination_name}.{secrets.token_hex(12)}.tmp" for item in items]
    linked: list[tuple[str, tuple[int, int]]] = []
    committed = False
    failure: BuildWheelError | None = None
    rollback_failed = False
    try:
        if not _same_directory(wheel_dir, directory_fd):
            raise BuildWheelError("output directory identity changed before publication")
        for item, temporary_name in zip(items, temporary_names, strict=True):
            _copy_into_temporary(item, directory_fd, temporary_name)
        for item, temporary_name in zip(items, temporary_names, strict=True):
            temporary_stat = os.stat(temporary_name, dir_fd=directory_fd, follow_symlinks=False)
            expected_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
            try:
                os.link(
                    temporary_name,
                    item.destination_name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError:
                # A wrapper or filesystem can report an error after link(2)
                # created the entry. Detect that uncertain success by inode so
                # the destination still participates in rollback.
                try:
                    destination_stat = os.stat(
                        item.destination_name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                except OSError as inspect_error:
                    raise BuildWheelError(
                        f"publication state is ambiguous for {item.destination_name}"
                    ) from inspect_error
                else:
                    destination_identity = (
                        destination_stat.st_dev,
                        destination_stat.st_ino,
                    )
                    if destination_identity == expected_identity:
                        linked.append((item.destination_name, expected_identity))
                raise
            linked.append((item.destination_name, expected_identity))
        if not _same_directory(wheel_dir, directory_fd):
            raise BuildWheelError("output directory identity changed during publication")
        for temporary_name in temporary_names:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except OSError:
                # A hidden cleanup residue does not invalidate the verified links,
                # but path identity must still be rechecked after this side effect.
                pass
        if not _same_directory(wheel_dir, directory_fd):
            raise BuildWheelError("output directory identity changed before commit")
        committed = True
    except BuildWheelError as exc:
        failure = exc
    except OSError as exc:
        failure = BuildWheelError(f"failed to publish verified wheel to {wheel_dir}")
        failure.__cause__ = exc
    finally:
        if not committed:
            # Reverse publication order removes the wheel commit marker before
            # its receipt and never unlinks a path whose inode was replaced.
            for destination_name, expected_identity in reversed(linked):
                try:
                    destination_stat = os.stat(
                        destination_name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if (destination_stat.st_dev, destination_stat.st_ino) != expected_identity:
                        rollback_failed = True
                        continue
                    os.unlink(destination_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    continue
                except OSError:
                    rollback_failed = True
            for temporary_name in temporary_names:
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except OSError:
                    pass
        os.close(directory_fd)

    if rollback_failed:
        raise BuildWheelError("wheel publication rollback could not be verified") from failure
    if failure is not None:
        raise failure
    return [wheel_dir / item.destination_name for item in items]


def _atomic_publish(source: Path, wheel_dir: Path) -> Path:
    return _atomic_publish_all([_PublishItem(source, source.name)], wheel_dir)[0]


def build_verify_publish(project_root: Path, wheel_dir: Path) -> Path:
    """Snapshot source, privately build/verify, then atomically publish wheel + receipt."""
    if not (project_root / "pyproject.toml").is_file():
        raise BuildWheelError(f"{project_root} does not contain pyproject.toml")
    _validate_output(wheel_dir)
    try:
        with tempfile.TemporaryDirectory(prefix="hermes-wheel-build-") as temporary_dir:
            temporary_root = Path(temporary_dir)
            snapshot_root = temporary_root / "source"
            private_output = temporary_root / "wheel"
            records = create_source_snapshot(project_root, snapshot_root)
            wheel_path = build_wheel(snapshot_root, private_output)
            _verify_snapshot_unchanged(snapshot_root, records)
            verify_wheel(wheel_path)
            wheel_sha256, wheel_stat = _hash_regular_file(wheel_path)
            receipt_bytes = _render_receipt(
                wheel_path.name, wheel_sha256, wheel_stat.st_size, records
            )
            receipt_source = temporary_root / f"{wheel_path.name}{RECEIPT_SUFFIX}"
            receipt_source.write_bytes(receipt_bytes)
            verify_receipt(receipt_source, wheel_path)
            _published_receipt, published_wheel = _atomic_publish_all(
                [
                    # The receipt may be transiently visible on its own, but the wheel
                    # is the commit marker and is never visible before its receipt.
                    _PublishItem(
                        receipt_source,
                        receipt_source.name,
                        hashlib.sha256(receipt_bytes).hexdigest(),
                    ),
                    _PublishItem(wheel_path, wheel_path.name, wheel_sha256),
                ],
                wheel_dir,
            )
            return published_wheel
    except BuildWheelError:
        raise
    except OSError as exc:
        raise BuildWheelError("private wheel workflow failed") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and verify a wheel from a clean private source snapshot."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project directory containing pyproject.toml (default: this repository).",
    )
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        default=None,
        help="Output directory for the verified wheel (default: <project-root>/dist).",
    )
    parser.add_argument(
        "--verify-receipt",
        type=Path,
        default=None,
        metavar="RECEIPT",
        help="Verify an existing build receipt against --wheel instead of building.",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help="Published wheel checked in --verify-receipt mode.",
    )
    args = parser.parse_args(argv)

    if (args.verify_receipt is None) != (args.wheel is None):
        print("error: --verify-receipt and --wheel must be used together", file=sys.stderr)
        return 1
    if args.verify_receipt is not None:
        try:
            receipt = verify_receipt(args.verify_receipt, args.wheel)
        except BuildWheelError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        wheel_field = receipt["wheel"]
        sha256 = wheel_field["sha256"] if isinstance(wheel_field, dict) else "unknown"
        print(f"verified receipt/wheel digest consistency: {args.wheel.name} sha256={sha256}")
        return 0

    try:
        project_root = args.project_root.resolve()
        wheel_dir = args.wheel_dir if args.wheel_dir is not None else project_root / "dist"
        if not wheel_dir.is_absolute():
            wheel_dir = Path.cwd() / wheel_dir
        wheel_path = build_verify_publish(project_root, wheel_dir)
    except BuildWheelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError) as exc:
        print(f"error: cannot resolve build paths: {exc}", file=sys.stderr)
        return 1
    print(f"verified clean wheel: {wheel_path}")
    print(f"unsigned build receipt: {wheel_path.with_name(wheel_path.name + RECEIPT_SUFFIX)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
