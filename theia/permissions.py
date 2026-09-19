"""Prepare Theia's runtime and workspace storage before application startup."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast


_ACCESS_FLAGS = os.R_OK | os.W_OK | os.X_OK
_CONTAINER_ENV = "THEIA_CONTAINER"


class StoragePermissionError(RuntimeError):
    """Theia could not make one of her startup storage roots usable."""

    def __init__(self, area: str, reason: str) -> None:
        self.area = area
        self.reason = reason
        super().__init__(f"{area}: {reason}")


def _effective_uid() -> int | None:
    getter = getattr(os, "geteuid", None) or getattr(os, "getuid", None)
    return getter() if getter is not None else None


def _effective_gid() -> int | None:
    getter = getattr(os, "getegid", None) or getattr(os, "getgid", None)
    return getter() if getter is not None else None


def _container_enabled() -> bool:
    return os.getenv(_CONTAINER_ENV, "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _configured_id(name: str, fallback: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return fallback
    try:
        configured = int(value, 10)
    except ValueError as exc:
        raise StoragePermissionError(
            "runtime data", "invalid container identity"
        ) from exc
    if configured < 0:
        raise StoragePermissionError("runtime data", "invalid container identity")
    return configured


def _resolved_path(value: Path) -> Path:
    try:
        path = Path(os.path.abspath(os.fspath(value.expanduser())))
    except (OSError, RuntimeError) as exc:
        raise StoragePermissionError(
            "runtime data", "storage path could not be resolved"
        ) from exc
    if path == Path(path.anchor):
        raise StoragePermissionError(
            "runtime data", "storage root cannot be the filesystem root"
        )
    return path


def _storage_roots(
    runtime_root: Path,
    workspace_root: Path,
    state_path: Path | None,
) -> tuple[tuple[str, Path], ...]:
    candidates: list[tuple[str, Path]] = [
        ("runtime data", _resolved_path(runtime_root)),
        ("workspace", _resolved_path(workspace_root)),
    ]
    if state_path is not None:
        candidates.append(("runtime data", _resolved_path(state_path).parent))

    roots: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for area, path in candidates:
        if path not in seen:
            seen.add(path)
            roots.append((area, path))
    return tuple(roots)


def _ensure_directory(area: str, root: Path) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StoragePermissionError(area, "storage root could not be created") from exc
    if not root.is_dir():
        raise StoragePermissionError(area, "storage root is not a directory")


def _walk_root(root: Path) -> tuple[tuple[Path, ...], bool]:
    entries: list[Path] = [root]
    walk_failed = False

    def on_error(_error: OSError) -> None:
        nonlocal walk_failed
        walk_failed = True

    try:
        for current, directories, files in os.walk(
            root,
            topdown=True,
            followlinks=False,
            onerror=on_error,
        ):
            entries.extend(Path(current) / name for name in directories)
            entries.extend(Path(current) / name for name in files)
    except OSError:
        walk_failed = True
    return tuple(entries), walk_failed


def _repair_ownership(root: Path, uid: int, gid: int) -> tuple[bool, bool]:
    entries, repair_failed = _walk_root(root)
    changed = False
    chown = cast(Callable[..., Any] | None, getattr(os, "chown", None))
    if chown is None:
        return True, False
    for path in entries:
        try:
            ownership = os.lstat(path)
        except OSError:
            repair_failed = True
            continue
        if ownership.st_uid == uid and ownership.st_gid == gid:
            continue
        try:
            chown(path, uid, gid, follow_symlinks=False)  # pylint: disable=not-callable
            changed = True
        except (OSError, TypeError):
            repair_failed = True
    return repair_failed, changed


def _drop_privileges(uid: int, gid: int) -> None:
    setgroups = cast(Callable[..., Any] | None, getattr(os, "setgroups", None))
    setgid = cast(Callable[..., Any] | None, getattr(os, "setgid", None))
    setuid = cast(Callable[..., Any] | None, getattr(os, "setuid", None))
    if setgroups is None or setgid is None or setuid is None:
        raise StoragePermissionError(
            "runtime data", "Theia could not drop container privileges"
        )
    try:
        setgroups([])  # pylint: disable=not-callable
        setgid(gid)  # pylint: disable=not-callable
        setuid(uid)  # pylint: disable=not-callable
    except (AttributeError, OSError) as exc:
        raise StoragePermissionError(
            "runtime data", "Theia could not drop container privileges"
        ) from exc

    if _effective_uid() != uid or _effective_gid() != gid:
        raise StoragePermissionError(
            "runtime data", "container privilege drop was not verified"
        )


def _check_access(area: str, root: Path) -> None:
    try:
        accessible = os.access(root, _ACCESS_FLAGS)
    except OSError as exc:
        raise StoragePermissionError(
            area, "storage permissions could not be checked"
        ) from exc
    if not accessible:
        raise StoragePermissionError(area, "storage permissions remain unusable")


def _check_state_access(state_path: Path | None) -> None:
    if state_path is None:
        return
    path = _resolved_path(state_path)
    try:
        exists = path.exists()
        is_file = path.is_file() if exists else True
        accessible = not exists or os.access(path, os.R_OK | os.W_OK)
    except OSError as exc:
        raise StoragePermissionError(
            "runtime data", "state permissions could not be checked"
        ) from exc
    if not is_file:
        raise StoragePermissionError("runtime data", "state path is not a file")
    if not accessible:
        raise StoragePermissionError(
            "runtime data", "state permissions remain unusable"
        )


def prepare_runtime_storage(
    runtime_root: Path,
    workspace_root: Path,
    state_path: Path | None = None,
) -> bool:
    """Repair Docker bind-mount ownership and verify startup access.

    Docker starts Theia as root only when ``THEIA_CONTAINER`` is enabled. In
    that mode ownership is repaired to ``THEIA_UID:THEIA_GID`` before Theia
    drops to that identity. Other launches retain their existing identity and
    do not inspect startup storage during module import.
    """
    containerized = _container_enabled()
    if not containerized:
        return False

    roots = _storage_roots(runtime_root, workspace_root, state_path)
    current_uid = _effective_uid()
    current_gid = _effective_gid()
    running_as_container_root = containerized and current_uid == 0
    target_uid = current_uid
    target_gid = current_gid
    if running_as_container_root:
        if current_gid is None:
            raise StoragePermissionError(
                "runtime data", "container identity is unavailable"
            )
        target_uid = _configured_id("THEIA_UID", current_uid)
        target_gid = _configured_id("THEIA_GID", current_gid)

    for area, root in roots:
        _ensure_directory(area, root)

    if running_as_container_root:
        if target_uid is None or target_gid is None:
            raise StoragePermissionError(
                "runtime data", "container identity is unavailable"
            )
        repaired = False
        for _area, root in roots:
            _failed, changed = _repair_ownership(root, target_uid, target_gid)
            repaired = repaired or changed
        if target_uid != current_uid or target_gid != current_gid:
            _drop_privileges(target_uid, target_gid)
    else:
        repaired = False

    for area, root in roots:
        _check_access(area, root)
    _check_state_access(state_path)
    return repaired
