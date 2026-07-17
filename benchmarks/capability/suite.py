"""Capability-suite loading and fail-closed validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.capability.schema import (
    SchemaError,
    TaskSpec,
    canonical_json,
    is_ignored_fixture_cache_path,
)
from benchmarks.capability.verifiers import VERIFIERS


@dataclass(frozen=True)
class CapabilitySuite:
    suite_id: str
    description: str
    root: Path
    tasks: tuple[TaskSpec, ...]
    suite_hash: str

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    @property
    def development_task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks if task.split == "development")

    @property
    def holdout_task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks if task.split == "holdout")


def _require_keys(obj: Any, required: set[str], optional: set[str], ctx: str) -> None:
    if not isinstance(obj, dict):
        raise SchemaError(f"{ctx}: expected an object")
    unknown = set(obj) - required - optional
    missing = required - set(obj)
    if unknown:
        raise SchemaError(f"{ctx}: unknown keys {sorted(unknown)}")
    if missing:
        raise SchemaError(f"{ctx}: missing keys {sorted(missing)}")


def load_suite(path: str | Path) -> CapabilitySuite:
    """Load and validate a suite JSON document and all referenced fixtures."""
    suite_path = Path(path).resolve()
    try:
        raw_text = suite_path.read_text(encoding="utf-8")
        raw = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"cannot load suite {suite_path}: {exc}") from exc

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
    root = suite_path.parent
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
            if asset.is_symlink():
                raise SchemaError(f"task {task.task_id!r}: symlink asset not allowed: {asset}")
        for verifier in task.verifiers:
            verifier_type = VERIFIERS.get(verifier.type)
            if verifier_type is None:
                raise SchemaError(
                    f"task {task.task_id!r}: unknown verifier {verifier.type!r}; "
                    f"known={sorted(VERIFIERS)}"
                )
            verifier_type.validate(verifier.params, task_dir)

    # Bind both the logical task document and every fixture/verifier asset.
    digest = hashlib.sha256(canonical_json(raw).encode("utf-8"))

    def is_ignored_copied_asset(path: Path) -> bool:
        for copied_root in copied_fixture_roots:
            try:
                relative = path.relative_to(copied_root)
            except ValueError:
                continue
            return is_ignored_fixture_cache_path(relative)
        return False

    def is_bound_asset(path: Path) -> bool:
        # Cache/OS entries are excluded only under copied workspace/replay
        # roots, where copy_fixture_tree applies the same predicate. Assets
        # used directly by verifiers (for example expected files) stay bound
        # even if their names look cache-like.
        return path.is_file() and path != suite_path and not is_ignored_copied_asset(path)

    for asset in sorted(p for p in root.rglob("*") if is_bound_asset(p)):
        digest.update(asset.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(asset.read_bytes())
        digest.update(b"\0")
    suite_hash = digest.hexdigest()
    return CapabilitySuite(
        suite_id=raw["suite_id"],
        description=raw["description"],
        root=root,
        tasks=tasks,
        suite_hash=suite_hash,
    )
