"""Portable project folders with atomic manifests and immutable copied assets.

The UI decides when a state has been accepted and calls ``save_current``.
Only the selected source image and an explicitly supplied mesh are copied.
No project operation copies a source directory or stores authentication data.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tempfile
from uuid import uuid4

from session_state import SessionDocument, _decode, _encode, _read_json, save_session
from user_state import UserPaths, atomic_write

PROJECT_FILE = "project.json"
PROJECT_VERSION = 1
MAX_SOURCE_BYTES = 512 * 1024 * 1024
MAX_MESH_BYTES = 2 * 1024 * 1024 * 1024
_CREDENTIAL_FIELDS = frozenset({"password", "api_key", "access_token", "refresh_token",
                               "client_secret", "private_key", "authorization"})


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_bytes(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2).encode("utf-8")


def _copy_document(document):
    encoded = _encode(document)
    _reject_credentials(encoded)
    return _decode(encoded)


def _reject_credentials(value):
    """Reject credential fields without inspecting or logging their contents."""
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.casefold().replace("-", "_")
            if any(normalized == field or normalized.endswith(f"_{field}") for field in _CREDENTIAL_FIELDS):
                raise ValueError("Authentication credentials cannot be stored in a project")
            _reject_credentials(child)
    elif isinstance(value, list):
        for child in value:
            _reject_credentials(child)


def _relative_asset(directory, value):
    """Resolve a portable reference without allowing foreign drives or symlinks out."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Invalid project asset reference")
    normalized = value.replace("\\", "/")
    portable = PurePosixPath(normalized)
    windows = PureWindowsPath(value)
    if portable.is_absolute() or windows.drive or ".." in portable.parts:
        raise ValueError("Project assets must stay inside the project folder")
    result = (directory / Path(*portable.parts)).resolve()
    if not result.is_relative_to(directory.resolve()) or result == directory.resolve():
        raise ValueError("Project assets must stay inside the project folder")
    return result


def _hash_file(path, limit):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            if size > limit:
                raise ValueError("Project asset exceeds its supported size")
            digest.update(block)
    return digest.hexdigest(), size


def _validate_asset(directory, record, limit):
    if not isinstance(record, dict):
        raise ValueError("Invalid project asset metadata")
    path = _relative_asset(directory, record.get("path"))
    expected = record.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Invalid project asset checksum")
    if type(record.get("size")) is not int or not 0 <= record["size"] <= limit:
        raise ValueError("Invalid project asset size")
    if not path.is_file():
        raise FileNotFoundError(f"Project asset is missing: {record['path']}")
    actual, size = _hash_file(path, limit)
    if actual != expected or size != record["size"]:
        raise ValueError(f"Project asset has changed or is corrupt: {record['path']}")
    return path


def _import_asset(directory, source, role, limit):
    """Stream one file to a new checksum-addressed asset; never replace an old asset."""
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"The selected {role} file is missing")
    if source.stat().st_size > limit:
        raise ValueError(f"The selected {role} file is too large")
    assets = directory / "assets"
    assets.mkdir(exist_ok=True)
    # Validate even an existing assets directory: it may have been replaced by a link.
    if not assets.resolve().is_relative_to(directory.resolve()):
        raise ValueError("The assets directory must stay inside the project folder")
    temporary = None
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as incoming, tempfile.NamedTemporaryFile(
            dir=assets, prefix=".asset-", delete=False
        ) as outgoing:
            temporary = Path(outgoing.name)
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                size += len(block)
                if size > limit:
                    raise ValueError(f"The selected {role} file is too large")
                digest.update(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        checksum = digest.hexdigest()
        suffix = source.suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,12}", suffix):
            suffix = ".bin"
        relative = f"assets/{role}-{checksum}{suffix}"
        destination = _relative_asset(directory, relative)
        if destination.exists():
            if _hash_file(destination, limit) != (checksum, size):
                raise ValueError("An existing project asset has been changed")
        else:
            os.replace(temporary, destination)
            temporary = None
        return {"path": relative, "sha256": checksum, "size": size}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _asset_for_save(directory, source, role, limit, previous):
    # Managed assets were copied on import. Revalidate and reuse their bytes on
    # settings-only saves instead of writing another large temporary image.
    if previous is not None:
        previous_path = _relative_asset(directory, previous.get("path"))
        if previous_path == Path(source).expanduser().resolve():
            _validate_asset(directory, previous, limit)
            return copy.deepcopy(previous)
    return _import_asset(directory, source, role, limit)


@dataclass(frozen=True)
class ProjectSnapshot:
    """An inspected candidate; the caller may restore the UI before accepting it."""

    path: Path
    document: SessionDocument
    metadata: dict
    legacy: bool = False

    @property
    def mesh_path(self):
        mesh = self.metadata.get("assets", {}).get("mesh")
        return _relative_asset(self.path.parent, mesh["path"]) if mesh else None


class ProjectStore:
    """Manage one accepted project without taking ownership of its surrounding files.

    ``create`` / ``save_current`` return a manifest path; ``load`` returns a fresh
    SessionDocument. ``inspect`` then ``accept`` supports a staged UI restore.
    Changing ``root`` changes only where new folders are created.
    """

    def __init__(self, root=None):
        self.root = Path(root).expanduser().resolve() if root is not None else UserPaths.discover().root / "projects"
        self._current = None

    @property
    def current_path(self):
        return self._current.path if self._current is not None else None

    @property
    def current_document(self):
        return _copy_document(self._current.document) if self._current is not None else None

    @property
    def current_metadata(self):
        return copy.deepcopy(self._current.metadata) if self._current is not None else None

    @property
    def current_mesh_path(self):
        return self._current.mesh_path if self._current is not None else None

    @property
    def is_legacy(self):
        return self._current is not None and self._current.legacy

    def set_root(self, root):
        candidate = Path(root).expanduser().resolve()
        if candidate.exists() and not candidate.is_dir():
            raise ValueError("The projects location must be a directory")
        self.root = candidate

    def inspect(self, path):
        """Read and verify a project without changing the current accepted project."""
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            path = path / PROJECT_FILE
        payload = _read_json(path)
        _reject_credentials(payload)
        if isinstance(payload, dict) and payload.get("format") == "edgemesh-session":
            document = _decode(payload)
            source = Path(document.source_path).expanduser()
            if not source.is_absolute():
                document.source_path = str((path.parent / source).resolve())
            return ProjectSnapshot(path, document, {}, legacy=True)
        if not isinstance(payload, dict) or payload.get("format") != "edgemesh-project":
            raise ValueError("Not an EdgeMesh project")
        if type(payload.get("version")) is not int or payload["version"] != PROJECT_VERSION:
            raise ValueError("Unsupported project version")
        if not isinstance(payload.get("name"), str) or len(payload["name"]) > 256:
            raise ValueError("Invalid project name")
        if not isinstance(payload.get("id"), str) or not re.fullmatch(r"[0-9a-f]{32}", payload["id"]):
            raise ValueError("Invalid project identifier")
        assets = payload.get("assets")
        if not isinstance(assets, dict) or "source" not in assets:
            raise ValueError("The project has no source asset")
        document = _decode(payload.get("session"))
        source = _validate_asset(path.parent, assets["source"], MAX_SOURCE_BYTES)
        if document.source_path != assets["source"]["path"]:
            raise ValueError("The project source reference does not match its asset")
        if assets.get("mesh") is not None:
            _validate_asset(path.parent, assets["mesh"], MAX_MESH_BYTES)
        document.source_path = str(source)
        return ProjectSnapshot(path, document, copy.deepcopy(payload))

    def accept(self, snapshot):
        """Accept a successfully restored candidate; copies prevent accidental mutation."""
        if not isinstance(snapshot, ProjectSnapshot):
            raise TypeError("Expected an inspected ProjectSnapshot")
        self._current = ProjectSnapshot(snapshot.path, _copy_document(snapshot.document),
                                        copy.deepcopy(snapshot.metadata), snapshot.legacy)
        return self.current_document

    def load(self, path):
        return self.accept(self.inspect(path))

    def create(self, document, name=None, *, directory=None, mesh_path=None):
        """Create a new, unique project; an explicit directory must not already exist."""
        accepted = _copy_document(document)
        source = Path(accepted.source_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError("Choose a source image before creating a project")
        title = (name if name is not None else source.stem).strip()
        if not title or len(title) > 256:
            raise ValueError("A project name must contain 1 to 256 characters")
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", title).strip("-_")[:60] or "untitled"
        project_id = uuid4().hex
        folder = Path(directory).expanduser().resolve() if directory is not None else self.root / f"project-{slug}-{project_id[:8]}"
        folder.mkdir(parents=True, exist_ok=False)
        payload = {"format": "edgemesh-project", "version": PROJECT_VERSION,
                   "id": project_id, "name": title, "created_at": _now(), "updated_at": _now()}
        # A failed write may leave this newly allocated folder for recovery, but it
        # never replaces the current project or deletes any user-owned directory.
        snapshot = self._save(folder / PROJECT_FILE, accepted, payload, mesh_path)
        self.accept(snapshot)
        return snapshot.path

    def save_current(self, document, *, mesh_path=None):
        """Atomically save accepted state. Passing no mesh clears the cached mesh.

        A caller that retains valid geometry can pass ``current_mesh_path``.
        This prevents silently associating previous geometry with edited settings.
        """
        if self._current is None:
            return self.create(document, mesh_path=mesh_path)
        accepted = _copy_document(document)
        if self._current.legacy:
            save_session(self._current.path, accepted)
            snapshot = ProjectSnapshot(self._current.path, accepted, {}, legacy=True)
        else:
            snapshot = self._save(self._current.path, accepted, self._current.metadata, mesh_path)
        self.accept(snapshot)
        return snapshot.path

    def _save(self, path, document, metadata, mesh_path):
        payload = copy.deepcopy(metadata)
        encoded = _encode(document)
        previous_assets = metadata.get("assets", {})
        assets = {"source": _asset_for_save(path.parent, document.source_path, "source", MAX_SOURCE_BYTES,
                                             previous_assets.get("source"))}
        if mesh_path is not None:
            assets["mesh"] = _asset_for_save(path.parent, mesh_path, "mesh", MAX_MESH_BYTES,
                                              previous_assets.get("mesh"))
        encoded["source_path"] = assets["source"]["path"]
        payload.update(session=encoded, assets=assets, updated_at=_now())
        content = _json_bytes(payload)
        # Use the same read bound as sessions so every successfully saved manifest
        # is subsequently readable (mask and history can make JSON surprisingly large).
        from session_state import MAX_DOCUMENT_BYTES
        if len(content) > MAX_DOCUMENT_BYTES:
            raise ValueError("Project metadata is too large")
        atomic_write(path, content)
        accepted = _copy_document(document)
        accepted.source_path = str(_relative_asset(path.parent, assets["source"]["path"]))
        return ProjectSnapshot(path, accepted, payload)

    def list_projects(self):
        """List lightweight folder metadata; invalid entries are omitted, never changed."""
        entries = []
        if not self.root.is_dir():
            return entries
        for path in self.root.glob(f"*/{PROJECT_FILE}"):
            try:
                payload = _read_json(path)
                if not isinstance(payload, dict) or payload.get("format") != "edgemesh-project":
                    continue
                entries.append({"path": str(path), "name": payload.get("name", path.parent.name),
                                "updated_at": payload.get("updated_at", "")})
            except (OSError, ValueError):
                from log_utils import get_logger
                get_logger().exception("Could not read project listing entry")
        return sorted(entries, key=lambda item: str(item["updated_at"]), reverse=True)
