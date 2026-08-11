"""Capability-suite loading and fail-closed validation."""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.capability.schema import (
    SchemaError,
    TaskSpec,
    _read_stable_snapshot,
    canonical_json,
    is_ignored_fixture_cache_path,
)
from benchmarks.capability.verifiers import VERIFIERS

_MAX_SUITE_BYTES = 1_000_000


@dataclass(frozen=True)
class CapabilitySuite:
    suite_id: str
    description: str
    root: Path = field(compare=False)
    tasks: tuple[TaskSpec, ...]
    suite_hash: str
    _snapshot_owner: tempfile.TemporaryDirectory[str] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def close(self) -> None:
        """Release an owned private snapshot root, if this suite has one.

        Manually constructed suites may point at caller-owned roots and have no
        snapshot owner; closing those instances is intentionally a no-op.
        """

        owner = self._snapshot_owner
        if owner is None:
            return
        owner.cleanup()
        object.__setattr__(self, "_snapshot_owner", None)

    def __enter__(self) -> CapabilitySuite:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _non_owning_copy(self) -> CapabilitySuite:
        return CapabilitySuite(
            suite_id=self.suite_id,
            description=self.description,
            root=self.root,
            tasks=self.tasks,
            suite_hash=self.suite_hash,
        )

    def __copy__(self) -> CapabilitySuite:
        return self._non_owning_copy()

    def __deepcopy__(self, memo: dict[int, object]) -> CapabilitySuite:
        clone = self._non_owning_copy()
        memo[id(self)] = clone
        return clone

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    @property
    def development_task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks if task.split == "development")

    @property
    def holdout_task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks if task.split == "holdout")


def _update_framed(digest: Any, value: bytes) -> None:
    """Bind one byte string with an unambiguous fixed-width length prefix."""

    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _suite_hash(
    raw: Any,
    directories: list[tuple[Path, int]],
    assets: list[tuple[Path, bytes, int]],
) -> str:
    digest = hashlib.sha256()
    _update_framed(digest, b"hermes-capability-suite-hash-v2")
    _update_framed(digest, canonical_json(raw).encode("utf-8"))
    for relative, mode in sorted(directories):
        _update_framed(digest, b"directory")
        _update_framed(digest, relative.as_posix().encode("utf-8"))
        _update_framed(digest, mode.to_bytes(4, "big"))
    for relative, data, mode in sorted(assets):
        _update_framed(digest, b"file")
        _update_framed(digest, relative.as_posix().encode("utf-8"))
        _update_framed(digest, mode.to_bytes(4, "big"))
        _update_framed(digest, data)
    return digest.hexdigest()


def _capture_tree(
    root: Path,
    suite_path: Path,
    copied_fixture_roots: list[Path],
    *,
    ctx_prefix: str,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, bytes, int]]]:
    def is_ignored_copied_asset(path: Path) -> bool:
        for copied_root in copied_fixture_roots:
            try:
                relative = path.relative_to(copied_root)
            except ValueError:
                continue
            return is_ignored_fixture_cache_path(relative)
        return False

    directories: list[tuple[Path, int]] = []
    assets: list[tuple[Path, bytes, int]] = []
    for asset in sorted(root.rglob("*")):
        asset_info = asset.stat(follow_symlinks=False)
        if stat.S_ISLNK(asset_info.st_mode):
            raise SchemaError(f"{ctx_prefix} asset must not be a symlink: {asset}")
        relative_asset = asset.relative_to(root)
        if stat.S_ISDIR(asset_info.st_mode):
            if not is_ignored_copied_asset(asset):
                directories.append((relative_asset, stat.S_IMODE(asset_info.st_mode)))
            continue
        if not stat.S_ISREG(asset_info.st_mode):
            raise SchemaError(f"special {ctx_prefix} asset not allowed: {asset}")
        if asset == suite_path or is_ignored_copied_asset(asset):
            continue
        asset_data = _read_stable_snapshot(
            asset,
            asset_info.st_size,
            f"{ctx_prefix} asset {asset.name!r}",
        )
        assets.append((relative_asset, asset_data, stat.S_IMODE(asset_info.st_mode)))
    return directories, assets


def _validate_private_suite_document(
    suite_path: Path,
    expected: bytes,
    expected_identity: tuple[int, int],
) -> None:
    captured = _read_stable_snapshot(
        suite_path,
        _MAX_SUITE_BYTES,
        "private suite document",
    )
    if captured != expected:
        raise SchemaError("private suite document changed before return (fail closed)")
    try:
        final_info = suite_path.lstat()
    except OSError as exc:
        raise SchemaError("private suite document could not be revalidated (fail closed)") from exc
    if not stat.S_ISREG(final_info.st_mode):
        raise SchemaError("private suite document type changed before return (fail closed)")
    if (final_info.st_dev, final_info.st_ino) != expected_identity:
        raise SchemaError("private suite document was replaced before return (fail closed)")
    if stat.S_IMODE(final_info.st_mode) != 0o600:
        raise SchemaError("private suite document mode changed before return (fail closed)")


def _require_keys(obj: Any, required: set[str], optional: set[str], ctx: str) -> None:
    if not isinstance(obj, dict):
        raise SchemaError(f"{ctx}: expected an object")
    unknown = set(obj) - required - optional
    missing = required - set(obj)
    if unknown:
        raise SchemaError(f"{ctx}: unknown keys {sorted(unknown)}")
    if missing:
        raise SchemaError(f"{ctx}: missing keys {sorted(missing)}")


def _load_suite(path: str | Path) -> CapabilitySuite:
    """Load and validate a bounded, strict-JSON suite and its fixtures.

    The suite document is captured through the shared stable-snapshot
    reader under a pre-resolved canonical parent: a symlinked, FIFO, or
    otherwise non-regular ``suite.json`` and mid-read replacement/change are
    typed rejections. Each regular fixture/verifier asset is captured through
    its own stable snapshot and materialized into a private root owned by the
    returned suite. Source capture is not a filesystem-wide atomic transaction,
    but later source-path changes cannot alter the returned executable root.
    """
    given_path = Path(path)
    try:
        root = given_path.parent.resolve()
    except (OSError, RuntimeError) as exc:
        raise SchemaError("cannot resolve suite parent (fail closed)") from exc
    suite_path = root / given_path.name

    def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SchemaError("suite: duplicate JSON key (fail closed)")
            result[key] = value
        return result

    def _no_non_finite(_constant: str) -> float:
        raise SchemaError("suite: non-finite JSON constant (fail closed)")

    encoded = _read_stable_snapshot(suite_path, _MAX_SUITE_BYTES, "suite")
    try:
        raw = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_no_non_finite,
        )
    except SchemaError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise SchemaError(f"cannot load suite {suite_path}: invalid strict JSON") from exc

    _require_keys(raw, {"schema_version", "suite_id", "description", "tasks"}, set(), "suite")
    if raw["schema_version"] != 1:
        raise SchemaError(f"suite: unsupported schema_version {raw['schema_version']!r}")
    if not isinstance(raw["suite_id"], str) or not raw["suite_id"]:
        raise SchemaError("suite: suite_id must be a non-empty string")
    if not isinstance(raw["description"], str):
        raise SchemaError("suite: description must be a string")
    if not isinstance(raw["tasks"], list) or not raw["tasks"]:
        raise SchemaError("suite: tasks must be a non-empty list")

    tasks = tuple(TaskSpec.from_dict(item) for item in raw["tasks"])
    seen: set[str] = set()
    seen_fixtures: set[Path] = set()
    copied_fixture_roots: list[Path] = []
    for task in tasks:
        if task.task_id in seen:
            raise SchemaError(f"suite: duplicate task_id {task.task_id!r}")
        seen.add(task.task_id)
        fixture_path = root / task.fixture
        current = root
        for part in Path(task.fixture).parts:
            current /= part
            if current.is_symlink():
                raise SchemaError(
                    f"task {task.task_id!r}: symlink fixture path not allowed: {current}"
                )
        task_dir = fixture_path.resolve()
        if root != task_dir and root not in task_dir.parents:
            raise SchemaError(f"task {task.task_id!r}: fixture escapes suite root")
        overlapping = next(
            (
                existing
                for existing in seen_fixtures
                if existing == task_dir
                or existing in task_dir.parents
                or task_dir in existing.parents
            ),
            None,
        )
        if overlapping is not None:
            raise SchemaError(
                f"task {task.task_id!r}: overlapping fixture directory {task.fixture!r}; "
                "sharing or nesting fixtures across tasks blurs the development/holdout boundary"
            )
        seen_fixtures.add(task_dir)
        workspace = task_dir / "workspace"
        if not workspace.is_dir():
            raise SchemaError(f"task {task.task_id!r}: fixture workspace missing: {workspace}")
        replay = task_dir / "replay"
        if not replay.is_dir():
            raise SchemaError(f"task {task.task_id!r}: replay workspace missing: {replay}")
        copied_fixture_roots.extend((workspace, replay))
        for asset in task_dir.rglob("*"):
            asset_info = asset.stat(follow_symlinks=False)
            if stat.S_ISLNK(asset_info.st_mode):
                raise SchemaError(f"task {task.task_id!r}: symlink asset not allowed: {asset}")
            if not (stat.S_ISDIR(asset_info.st_mode) or stat.S_ISREG(asset_info.st_mode)):
                raise SchemaError(
                    f"task {task.task_id!r}: special fixture asset not allowed: {asset}"
                )
        for verifier in task.verifiers:
            verifier_type = VERIFIERS.get(verifier.type)
            if verifier_type is None:
                raise SchemaError(
                    f"task {task.task_id!r}: unknown verifier {verifier.type!r}; "
                    f"known={sorted(VERIFIERS)}"
                )
            verifier_type.validate(verifier.params, task_dir)

    snapshot_directories, snapshot_assets = _capture_tree(
        root,
        suite_path,
        copied_fixture_roots,
        ctx_prefix="suite",
    )
    suite_hash = _suite_hash(raw, snapshot_directories, snapshot_assets)
    if _read_stable_snapshot(suite_path, _MAX_SUITE_BYTES, "suite") != encoded:
        raise SchemaError("suite document changed during validation (fail closed)")

    snapshot_owner: tempfile.TemporaryDirectory[str] | None = None
    try:
        snapshot_owner = tempfile.TemporaryDirectory(prefix="capability-suite-snapshot-")
        snapshot_root = Path(snapshot_owner.name)
        for relative_directory, _mode in sorted(
            snapshot_directories, key=lambda item: len(item[0].parts)
        ):
            destination = snapshot_root / relative_directory
            destination.mkdir(parents=True, exist_ok=True)

        snapshot_suite_path = snapshot_root / suite_path.name
        snapshot_suite_path.write_bytes(encoded)
        snapshot_suite_path.chmod(0o600)
        if (
            _read_stable_snapshot(
                snapshot_suite_path,
                _MAX_SUITE_BYTES,
                "private suite snapshot",
            )
            != encoded
        ):
            raise SchemaError("private suite snapshot write could not be confirmed")
        snapshot_suite_info = snapshot_suite_path.lstat()
        snapshot_suite_identity = (
            snapshot_suite_info.st_dev,
            snapshot_suite_info.st_ino,
        )

        for relative_asset, asset_data, mode in snapshot_assets:
            destination = snapshot_root / relative_asset
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(asset_data)
            destination.chmod(mode)
            if (
                _read_stable_snapshot(
                    destination,
                    len(asset_data),
                    f"private suite asset {destination.name!r}",
                )
                != asset_data
            ):
                raise SchemaError("private suite asset write could not be confirmed")

        # Restore source directory modes only after all descendants have been
        # materialized. Applying a read-only mode during creation would make a
        # valid 0555 source tree impossible to snapshot.
        for relative_directory, mode in sorted(
            snapshot_directories,
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            (snapshot_root / relative_directory).chmod(mode)

        # Revalidate execution and verifier dependencies against the exact
        # private tree callers receive, not the mutable source path.
        for task in tasks:
            task_dir = snapshot_root / task.fixture
            if not (task_dir / "workspace").is_dir():
                raise SchemaError(f"task {task.task_id!r}: snapshot workspace missing")
            if not (task_dir / "replay").is_dir():
                raise SchemaError(f"task {task.task_id!r}: snapshot replay workspace missing")
            for verifier in task.verifiers:
                VERIFIERS[verifier.type].validate(verifier.params, task_dir)

        private_copied_roots = [
            snapshot_root / copied_root.relative_to(root) for copied_root in copied_fixture_roots
        ]
        final_directories, final_assets = _capture_tree(
            snapshot_root,
            snapshot_suite_path,
            private_copied_roots,
            ctx_prefix="private suite",
        )
        if _suite_hash(raw, final_directories, final_assets) != suite_hash:
            raise SchemaError("private suite tree changed before return (fail closed)")
        _validate_private_suite_document(
            snapshot_suite_path,
            encoded,
            snapshot_suite_identity,
        )

        suite = CapabilitySuite(
            suite_id=raw["suite_id"],
            description=raw["description"],
            root=snapshot_root,
            tasks=tasks,
            suite_hash=suite_hash,
        )
        object.__setattr__(suite, "_snapshot_owner", snapshot_owner)
        return suite
    except BaseException:
        if snapshot_owner is not None:
            snapshot_owner.cleanup()
        raise


def load_suite(path: str | Path) -> CapabilitySuite:
    """Load a suite through one typed, fail-closed public boundary."""

    try:
        return _load_suite(path)
    except SchemaError:
        raise
    except (OSError, RuntimeError, RecursionError, ValueError) as exc:
        raise SchemaError("cannot load suite: validation failed (fail closed)") from exc
