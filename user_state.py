"""Portable, atomic local state and privacy-preserving support exports."""
from dataclasses import dataclass
import configparser
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import zipfile


@dataclass(frozen=True)
class UserPaths:
    root: Path
    config_file: Path
    presets_dir: Path
    work_dir: Path
    logs_dir: Path

    @classmethod
    def discover(cls):
        override = os.environ.get("EDGEMESH_DATA_DIR")
        if override:
            root = Path(override).expanduser().resolve()
        elif sys.platform == "win32":
            root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "EdgeMesh"
        elif sys.platform == "darwin":
            root = Path.home() / "Library" / "Application Support" / "EdgeMesh"
        else:
            root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "EdgeMesh"
        paths = cls(root, root / "config.ini", root / "presets", root / "work", root / "logs")
        for directory in (root, paths.presets_dir, paths.work_dir, paths.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


def _error(message):
    # Late import avoids a cycle: log_utils uses UserPaths too.
    from log_utils import get_logger
    try:
        get_logger().exception(message)
    except OSError:
        logging.getLogger(__name__).exception(message)


def atomic_write(path, data):
    """Write bytes to a sibling temporary file, replacing the target only on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_config(config, path):
    try:
        content = io.StringIO()
        config.write(content)
        atomic_write(path, content.getvalue().encode("utf-8"))
    except (OSError, configparser.Error):
        _error("Could not save application settings")
        raise


def _read_config(path):
    config = configparser.ConfigParser(interpolation=None)
    with Path(path).open(encoding="utf-8-sig") as stream:
        config.read_file(stream)
    if not config.has_section("Settings"):
        config.add_section("Settings")
    return config


def migrate_config(legacy_path, paths):
    """Copy parseable legacy settings once, preserving originals and allowing retry.

    Malformed user settings are preserved on disk; callers receive defaults and
    can repair them. Failed legacy parsing leaves the destination absent so a
    repaired legacy file can migrate on the next call.
    """
    if paths.config_file.exists():
        try:
            return _read_config(paths.config_file)
        except (OSError, UnicodeError, configparser.Error):
            _error("User settings could not be read; using defaults")
    else:
        legacy = Path(legacy_path) if legacy_path else None
        if legacy is not None and legacy.is_file():
            try:
                config = _read_config(legacy)
                save_config(config, paths.config_file)
                return config
            except (OSError, UnicodeError, configparser.Error):
                _error("Legacy settings migration failed; original retained for retry")
        else:
            config = configparser.ConfigParser(interpolation=None)
            config["Settings"] = {"last_used_image": ""}
            save_config(config, paths.config_file)
            return config
    default = configparser.ConfigParser(interpolation=None)
    default["Settings"] = {"last_used_image": ""}
    return default


def export_diagnostics(destination, paths):
    """Export metadata and redacted log summaries, never raw text or source files.

    Arbitrary tracebacks may embed credentials or image paths, so the export
    keeps severity counts and file sizes rather than attempting unsafe heuristic
    redaction. Configuration values and filesystem paths are never included.
    """
    try:
        metadata = {"format": "edgemesh-diagnostics", "version": 1,
                    "platform": sys.platform, "python": list(sys.version_info[:3]),
                    "config_present": paths.config_file.is_file()}
        if paths.config_file.is_file():
            try:
                config = _read_config(paths.config_file)
                metadata["config_sections"] = len(config.sections())
                metadata["config_options"] = sum(len(config[s]) for s in config.sections())
            except (OSError, UnicodeError, configparser.Error):
                metadata["config_parse_error"] = True
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metadata.json", json.dumps(metadata, indent=2))
            for index, path in enumerate(sorted(paths.logs_dir.glob("*.log*"))):
                if not path.is_file() or path.is_symlink():
                    continue
                counts = {name: 0 for name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")}
                # Bound work even if an external tool wrote a huge log.
                with path.open("rb") as stream:
                    content = stream.read(10_000_000).decode("utf-8", errors="replace")
                for level in re.findall(r" - (DEBUG|INFO|WARNING|ERROR|CRITICAL) - ", content):
                    counts[level] += 1
                summary = {"bytes": path.stat().st_size, "levels": counts,
                           "messages": "omitted to protect private paths and credentials"}
                archive.writestr(f"logs/log-{index:03d}.json", json.dumps(summary, indent=2))
        atomic_write(destination, buffer.getvalue())
        return Path(destination)
    except (OSError, ValueError, zipfile.BadZipFile):
        _error("Could not export diagnostics")
        raise
