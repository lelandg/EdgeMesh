"""Versioned, validated sessions, reusable presets and bounded snapshot history."""
import base64
import copy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import zlib

import numpy as np

from user_state import atomic_write
from da3_backend import DA3_ALIASES

VERSION = 1
MAX_PIXELS = 50_000_000
MAX_DOCUMENT_BYTES = 20_000_000
BOOL_KEYS = frozenset({"grayscale_enabled", "edge_detection_enabled", "invert_colors_enabled",
    "project_on_original", "flat_back_enabled", "drop_background_enabled", "use_processed_image_enabled",
    "use_selected_color"})
RANGES = {"resolution": (0, 10000), "depth_amount": (0, 100), "depth_drop_percentage": (0, 100),
    "sensitivity": (1, 200), "line_thickness": (1, 10), "blend_amount": (0, 100), "background_tolerance": (0, 255)}
INTEGER_KEYS = {"resolution", "sensitivity", "line_thickness", "blend_amount"}
SMOOTHING = {"anisotropic", "gaussian", "bilateral", "median", "(none)", "none"}
DEPTH_MODELS = frozenset({"MiDaS", "DPT", "DepthAnythingV2", "Depth Pro", *DA3_ALIASES})


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("Settings must be an object")
    result = {}
    for key, value in settings.items():
        if key in BOOL_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
        elif key in RANGES:
            low, high = RANGES[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
            if key in INTEGER_KEYS and not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
        elif key == "smoothing_method":
            if not isinstance(value, str) or value not in SMOOTHING:
                raise ValueError("Unknown smoothing method")
            value = "(none)" if value == "none" else value
        elif key == "model":
            if not isinstance(value, str) or value not in DEPTH_MODELS:
                raise ValueError("Unknown depth model")
        elif key == "current_selected_color":
            if value is not None and (not isinstance(value, (list, tuple)) or len(value) != 3 or
                    any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in value)):
                raise ValueError("Selected color must be an RGB triple")
            value = list(value) if value is not None else None
        else:
            raise ValueError(f"Unknown setting: {key}")
        result[key] = copy.deepcopy(value)
    return result


@dataclass
class SessionDocument:
    source_path: str
    settings: dict
    model_info: dict = field(default_factory=dict)
    mask: np.ndarray | None = None
    history: list = field(default_factory=list)


def _json_copy(value):
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Session metadata must contain finite JSON values") from error
    if len(encoded) > 1_000_000:
        raise ValueError("Session metadata is too large")
    return json.loads(encoded)


def _encode_mask(mask):
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.bool_ or not 0 < mask.size <= MAX_PIXELS:
        raise ValueError("Mask must be a nonempty two-dimensional boolean array")
    packed = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
    return {"shape": list(mask.shape), "encoding": "zlib-packbits-little",
            "data": base64.b64encode(zlib.compress(packed)).decode("ascii")}


def _decode_mask(value):
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("encoding") != "zlib-packbits-little":
        raise ValueError("Unsupported mask encoding")
    shape = value.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("Invalid mask dimensions")
    pixels = math.prod(shape)
    if pixels > MAX_PIXELS:
        raise ValueError("Mask is too large")
    try:
        compressed = base64.b64decode(value["data"], validate=True)
        inflater = zlib.decompressobj()
        expected = (pixels + 7) // 8
        packed = inflater.decompress(compressed, expected + 1)
        if len(packed) != expected or not inflater.eof or inflater.unused_data:
            raise ValueError("Invalid compressed mask size")
    except (KeyError, TypeError, zlib.error, ValueError) as error:
        raise ValueError("Invalid mask data") from error
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little", count=pixels).reshape(shape).astype(bool)


def _encode(document):
    if not isinstance(document, SessionDocument) or not isinstance(document.source_path, str) or len(document.source_path) > 32768:
        raise ValueError("A session requires a valid source reference")
    if not isinstance(document.model_info, dict) or not isinstance(document.history, list):
        raise ValueError("Invalid session metadata")
    return {"format": "edgemesh-session", "version": VERSION,
        "source_path": document.source_path, "settings": validate_settings(document.settings),
        "model_info": _json_copy(document.model_info), "mask": _encode_mask(document.mask),
        "history": _json_copy(document.history)}


def _decode(payload):
    if not isinstance(payload, dict) or payload.get("format") != "edgemesh-session":
        raise ValueError("Not an EdgeMesh session")
    # Version zero was a settings/source-only draft; migrate without inventing data.
    version = payload.get("version")
    if type(version) is not int or version not in (0, VERSION):
        raise ValueError("Unsupported session version")
    document = SessionDocument(payload.get("source_path"), payload.get("settings"),
        payload.get("model_info", {}), _decode_mask(payload.get("mask")), payload.get("history", []))
    _encode(document)
    return document


def _read_json(path):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_DOCUMENT_BYTES + 1)
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("Session file is too large")
    try:
        return json.loads(data, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid JSON file") from error


def save_session(path, document):
    atomic_write(path, json.dumps(_encode(document), allow_nan=False, indent=2).encode("utf-8"))


def load_session(path):
    # Missing images are allowed: the UI can relocate source_path before generation.
    return _decode(_read_json(path))


def save_preset(path, settings):
    payload = {"format": "edgemesh-preset", "version": VERSION, "settings": validate_settings(settings)}
    atomic_write(path, json.dumps(payload, allow_nan=False, indent=2).encode("utf-8"))


def load_preset(path):
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("format") != "edgemesh-preset" or type(payload.get("version")) is not int or payload["version"] != VERSION:
        raise ValueError("Unsupported preset format")
    return validate_settings(payload.get("settings"))


class SessionHistory:
    def __init__(self, limit=30):
        if type(limit) is not int or limit < 1:
            raise ValueError("History limit must be positive")
        self.limit = limit
        self._snapshots = []
        self._index = -1
        self._revision = 0

    @property
    def index(self):
        """Index of the active state, or -1 when the history is empty."""
        return self._index

    @property
    def revision(self):
        """Monotonic change marker for history contents and the active cursor."""
        return self._revision

    def snapshot_at(self, index):
        """Read an independent state without changing the active history entry."""
        self._validate_index(index)
        return _decode(copy.deepcopy(self._snapshots[index]))

    def select(self, index):
        """Commit a restored state as active after the caller restores it successfully."""
        self._validate_index(index)
        if index != self._index:
            self._index = index
            self._revision += 1

    def relocate_source(self, previous_path, source_path):
        """Reanchor matching snapshots after a source is copied into its project.

        This is a location change, not an edit: preserve the cursor, redo states,
        masks and model provenance. Return the number of reanchored snapshots.
        """
        if any(not isinstance(path, str) or not path or len(path) > 32768
               for path in (previous_path, source_path)):
            raise ValueError("Source relocation requires two valid source references")
        if previous_path == source_path:
            return 0
        changed = sum(snapshot["source_path"] == previous_path for snapshot in self._snapshots)
        if changed:
            self._snapshots = [
                {**snapshot, "source_path": source_path}
                if snapshot["source_path"] == previous_path else snapshot
                for snapshot in self._snapshots
            ]
            self._revision += 1
        return changed

    def _validate_index(self, index):
        if type(index) is not int or not 0 <= index < len(self._snapshots):
            raise IndexError("History index is out of range")

    def summaries(self):
        """Return copied display metadata without decompressing any mask payloads."""
        summaries = []
        for index, snapshot in enumerate(self._snapshots):
            previous = self._snapshots[index - 1] if index else None
            settings = snapshot["settings"]
            previous_settings = previous["settings"] if previous else {}
            changed_settings = {
                key: {"before": previous_settings.get(key), "after": settings.get(key)}
                for key in sorted(settings.keys() | previous_settings.keys())
                if settings.get(key) != previous_settings.get(key)
            } if previous else {}
            mask = snapshot["mask"]
            summaries.append({
                "index": index,
                "source_path": snapshot["source_path"],
                "settings": copy.deepcopy(settings),
                "model_info": copy.deepcopy(snapshot["model_info"]),
                "has_mask": mask is not None,
                "mask_shape": tuple(mask["shape"]) if mask is not None else None,
                "source_changed": previous is not None and snapshot["source_path"] != previous["source_path"],
                "mask_changed": previous is not None and mask != previous["mask"],
                "changed_settings": copy.deepcopy(changed_settings),
            })
        return summaries

    @property
    def current(self):
        return self.snapshot_at(self._index) if self._index >= 0 else None

    @property
    def can_undo(self):
        return self._index > 0

    @property
    def can_redo(self):
        return self._index + 1 < len(self._snapshots)

    def record(self, document):
        snapshot = _encode(document)
        # Settings-only edits reuse the immutable compressed mask payload.
        if self._index >= 0 and snapshot["mask"] == self._snapshots[self._index]["mask"]:
            snapshot["mask"] = self._snapshots[self._index]["mask"]
        if self._index >= 0 and snapshot == self._snapshots[self._index]:
            return self.current
        self._snapshots = self._snapshots[:self._index + 1]
        self._snapshots.append(snapshot)
        self._snapshots = self._snapshots[-self.limit:]
        self._index = len(self._snapshots) - 1
        self._revision += 1
        return self.current

    def undo(self):
        if not self.can_undo:
            return None
        self.select(self._index - 1)
        return self.current

    def redo(self):
        if not self.can_redo:
            return None
        self.select(self._index + 1)
        return self.current
