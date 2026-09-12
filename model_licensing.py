"""Checkpoint license policy and local, tamper-evident mesh provenance.

This catalog describes weights, not EdgeMesh's own license. A local HMAC binds
model identities to geometry; it is not DRM or proof from a model publisher.
No ML, NumPy, keyring, or GUI dependency is imported at module load time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import secrets
import stat
import struct
import sys
import tempfile
import threading
from types import MappingProxyType


LOGGER = logging.getLogger(__name__)
_KEY_LOCK = threading.RLock()
CC_NC = "https://creativecommons.org/licenses/by-nc/4.0/"
APACHE = "https://www.apache.org/licenses/LICENSE-2.0"
DA2_SOURCE = "https://github.com/DepthAnything/Depth-Anything-V2#license"
DA3_SOURCE = "https://github.com/ByteDance-Seed/Depth-Anything-3#-model-cards"
GENERAL_NOTICE = (
    "General guidance, not legal advice. Read the linked weight license for your "
    "intended use. EdgeMesh being free or non-commercial does not change a model's terms. "
    "This label records the model used; it does not decide ownership or license of every output."
)


def _log_event(level, message, *, exc_info=False):
    """Use the app's per-user log without creating files at module import."""
    try:
        from log_utils import get_logger
        logger = get_logger(__name__)
    except OSError:
        logger = LOGGER
    logger.log(level, message, exc_info=exc_info)


@dataclass(frozen=True)
class ModelPolicy:
    model_type: str
    model_id: str
    display_name: str
    license: str
    usage_class: str
    license_url: str
    source_url: str
    guidance: str
    restriction: str = ""
    commercial_status: str = "not-published"
    commercial_url: str = ""
    commercial_info: str = "No separate commercial licensing offer is published in the reviewed official sources."
    policy_version: int = 1

    @property
    def requires_acceptance(self):
        return self.usage_class in {"noncommercial", "custom", "unknown"}

    @property
    def badge(self):
        if self.restriction == "research-only":
            return "NC · research only"
        return {
            "noncommercial": "NC · non-commercial",
            "permissive": "Permissive model license",
            "custom": "Custom model terms",
        }.get(self.usage_class, "Model license unverified")

    def as_dict(self):
        return {
            **asdict(self),
            "requires_acceptance": self.requires_acceptance,
            "badge": self.badge,
            "notice": GENERAL_NOTICE,
        }


def _depth_policy(model_type, model_id, display_name, noncommercial, source):
    return ModelPolicy(
        model_type, model_id, display_name,
        "cc-by-nc-4.0" if noncommercial else "apache-2.0",
        "noncommercial" if noncommercial else "permissive",
        CC_NC if noncommercial else APACHE, source,
        (
            "Use the model for non-commercial purposes. Keep attribution, license links, "
            "and notices when sharing licensed material, and identify changes. Do not use "
            "these weights for paid client work, a paid service, or other commercial "
            "advantage without separate permission from the rights holder. A non-profit "
            "organization or free app does not automatically make every use non-commercial."
            if noncommercial else
            "The weight license permits commercial and non-commercial use subject to its "
            "terms. Keep required copyright, license, and NOTICE material when distributing "
            "covered material; identify modifications and do not imply endorsement."
        ),
        commercial_status="not-published" if noncommercial else "included",
        commercial_url="" if noncommercial else APACHE,
        commercial_info=(
            "No separate commercial licensing offer is published in the reviewed official sources."
            if noncommercial else "Commercial use is included under Apache 2.0; follow its conditions."
        ),
    )


_POLICIES = [
    _depth_policy("depth_anything_v1", "LiheYoung/depth-anything-large-hf", "Depth Anything V1 Large", False,
                  "https://huggingface.co/LiheYoung/depth-anything-large-hf"),
    _depth_policy("depth_anything_v2", "depth-anything/Depth-Anything-V2-Large-hf", "Depth Anything V2 Large", True, DA2_SOURCE),
    _depth_policy("depth_anything_v2_small", "depth-anything/Depth-Anything-V2-Small-hf", "Depth Anything V2 Small", False, DA2_SOURCE),
    _depth_policy("depth_anything_v2_base", "depth-anything/Depth-Anything-V2-Base-hf", "Depth Anything V2 Base", True, DA2_SOURCE),
    ModelPolicy(
        "depth_pro", "apple/DepthPro-hf", "Depth Pro", "apple-amlr", "noncommercial",
        "https://huggingface.co/apple/DepthPro/blob/main/LICENSE",
        "https://huggingface.co/apple/DepthPro-hf",
        "These weights are restricted to non-commercial scientific research and academic "
        "development. The license excludes product development and commercial products or "
        "services. General hobby, creative, or free-product use is not automatically covered. "
        "Redistributing the model or model derivatives also requires the agreement, attribution, "
        "and modification notices specified by Apple.",
        restriction="research-only",
    ),
    ModelPolicy(
        "sam2", "facebook/sam2.1-hiera-tiny", "SAM 2.1 Tiny", "apache-2.0", "permissive",
        APACHE, "https://github.com/facebookresearch/sam2#license",
        "The model weights permit commercial and non-commercial use under Apache 2.0. "
        "Preserve required license, attribution, and NOTICE material when distributing covered material.",
        commercial_status="included", commercial_url=APACHE,
        commercial_info="Commercial use is included under Apache 2.0; follow its conditions.",
    ),
]
for _suffix, _nc in (
    ("SMALL", False), ("BASE", False), ("LARGE", True), ("LARGE-1.1", True),
    ("GIANT", True), ("GIANT-1.1", True), ("MONO-LARGE", False),
    ("METRIC-LARGE", False), ("NESTED-GIANT-LARGE", True), ("NESTED-GIANT-LARGE-1.1", True),
):
    _hf_name = "DA3-" + _suffix if _suffix.split("-")[0] in {"SMALL", "BASE", "LARGE", "GIANT"} else "DA3" + _suffix
    _POLICIES.append(_depth_policy(
        "depth_anything_3_" + _suffix.lower().replace("-", "_"),
        "depth-anything/" + _hf_name, _hf_name.replace("-", " "), _nc, DA3_SOURCE,
    ))

CATALOG = MappingProxyType({policy.model_type: policy for policy in _POLICIES})
_BY_ID = MappingProxyType({policy.model_id: policy for policy in _POLICIES})
_ALIASES = MappingProxyType({
    "DepthAnythingV1": "depth_anything_v1",
    "DepthAnythingV2": "depth_anything_v2", "Depth Pro": "depth_pro", "SAM2": "sam2",
    "DepthAnythingV2Small": "depth_anything_v2_small", "DepthAnythingV2Base": "depth_anything_v2_base",
})
del _POLICIES, _suffix, _nc, _hf_name


def policy_for(model_type_or_id):
    """Return built-in policy, never policy supplied by a project/session."""
    name = str(model_type_or_id or "")
    found = _BY_ID.get(name) or CATALOG.get(_ALIASES.get(name, name))
    if found is not None:
        return found
    return _unknown_policy(name)


def _unknown_policy(model_id):
    name = str(model_id or "")
    return ModelPolicy(
        name, name, name or "Unknown model", "unknown", "unknown", "", "",
        "The exact checkpoint's weight terms have not been verified. Review its publisher's "
        "license and any separate checkpoint conditions before use or redistribution. "
        "A source-code license alone does not establish the license of custom weights.",
        commercial_status="unverified",
        commercial_info="Commercial permissions have not been verified for this checkpoint.",
    )


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def policy_digest(policy):
    return hashlib.sha256(canonical_json(policy.as_dict())).hexdigest()


def canonical_model_identity(metadata):
    """Preserve identity facts while replacing untrusted license claims.

    An explicit unknown model_id stays unknown even if model_type names a known
    permissive model. Known IDs take precedence over contradictory type labels.
    """
    result = json.loads(canonical_json(dict(metadata)))
    if "model_id" in result:
        model_id = result["model_id"]
        policy = _BY_ID.get(model_id) if isinstance(model_id, str) else None
        if policy is None:
            policy = _unknown_policy(model_id)
    else:
        policy = policy_for(result.get("model_type"))
    result["license"] = policy.license
    result["license_policy"] = policy.as_dict()
    result["policy_sha256"] = policy_digest(policy)
    if policy.usage_class != "unknown":
        result["model_id"] = policy.model_id
        result["model_type"] = policy.model_type
    return result


def geometry_sha256(vertices, faces):
    """Hash exact indexed geometry with fixed dtype/order and framed dimensions."""
    import numpy as np

    points = np.asarray(vertices, dtype="<f8")
    source_faces = np.asarray(faces)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Vertices must be a finite N by 3 array.")
    if source_faces.ndim != 2 or source_faces.shape[1] != 3 or source_faces.dtype.kind not in "iu":
        raise ValueError("Faces must be an integer N by 3 array.")
    if source_faces.size and (source_faces.min() < 0 or source_faces.max() >= len(points)):
        raise ValueError("A face references a vertex outside the mesh.")
    triangles = np.asarray(source_faces, dtype="<i8")
    digest = hashlib.sha256(b"EdgeMesh geometry v1\x00")
    for label, array in ((b"vertices", points), (b"triangles", triangles)):
        digest.update(label + struct.pack("<QQ", *array.shape))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _dpapi(data, protect):
    """Windows current-user DPAPI, with interactive credential UI disabled."""
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source = Blob(len(data), buffer)
    destination = Blob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    method = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    method.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.POINTER(Blob), ctypes.c_void_p,
                       ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    method.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    if not method(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(destination)):
        raise OSError("Windows could not protect/unprotect the local provenance key.")
    try:
        return ctypes.string_at(destination.data, destination.size)
    finally:
        kernel32.LocalFree(destination.data)


class ProvenanceStore:
    """Local HMAC seal; the key never travels with the project or mesh.

    Windows uses current-user DPAPI encryption. Other platforms use a mode-0600
    key in a mode-0700 directory. Users/admins who control this app can replace
    its code/key and re-sign data; this is tamper evidence, not enforcement.
    """

    def __init__(self, root=None):
        if root is None:
            from user_state import UserPaths
            root = UserPaths.discover().root
        self.key_path = Path(root) / "provenance" / "local-key.bin"

    def _key(self, create=False):
        with _KEY_LOCK:
            if not self.key_path.exists():
                if not create:
                    return None
                self.key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if sys.platform != "win32":
                    self.key_path.parent.chmod(0o700)
                value = secrets.token_bytes(32)
                encoded = b"DPAPI1\x00" + _dpapi(value, True) if sys.platform == "win32" else b"POSIX1\x00" + value
                # Publish only a complete file. Another process may be creating
                # its first key too; never overwrite the winning instance's key.
                descriptor, temporary = tempfile.mkstemp(prefix=".key-", dir=self.key_path.parent)
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(encoded)
                        stream.flush()
                        os.fsync(stream.fileno())
                    try:
                        os.link(temporary, self.key_path)
                    except FileExistsError:
                        pass
                finally:
                    Path(temporary).unlink(missing_ok=True)
            if self.key_path.is_symlink():
                raise OSError("Refusing a symlink as the local provenance key.")
            if sys.platform != "win32" and stat.S_IMODE(self.key_path.stat().st_mode) & 0o077:
                raise OSError("The local provenance key must have user-only file permissions.")
            encoded = self.key_path.read_bytes()
            if encoded.startswith(b"DPAPI1\x00") and sys.platform == "win32":
                value = _dpapi(encoded[7:], False)
            elif encoded.startswith(b"POSIX1\x00") and sys.platform != "win32":
                value = encoded[7:]
            else:
                raise OSError("The local provenance key belongs to a different platform or is invalid.")
            if len(value) != 32:
                raise OSError("The local provenance key is invalid.")
            return value

    def seal(self, vertices, faces, model_identities, parameters=None):
        payload = {
            "schema_version": 1,
            "geometry_sha256": geometry_sha256(vertices, faces),
            "models": [canonical_model_identity(identity) for identity in model_identities],
            "parameters": dict(parameters or {}),
        }
        # Normalize once so the signed object is exactly what is returned/saved.
        payload = json.loads(canonical_json(payload))
        envelope = {"format": "edgemesh-provenance", "payload": payload, "algorithm": "hmac-sha256"}
        try:
            key = self._key(create=True)
            envelope["key_id"] = hashlib.sha256(key).hexdigest()[:24]
            envelope["signature"] = hmac.new(key, canonical_json(payload), hashlib.sha256).hexdigest()
        except OSError:
            _log_event(logging.ERROR, "Could not seal mesh provenance with the local key", exc_info=True)
            envelope.update(key_id="", signature="", integrity="unverified")
        return envelope

    def verify(self, envelope, vertices=None, faces=None):
        """Never create a key while verifying imported/project metadata."""
        try:
            if not isinstance(envelope, dict) or envelope.get("format") != "edgemesh-provenance":
                raise ValueError("Provenance is missing or has an unsupported format.")
            payload = envelope["payload"]
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError("Unsupported provenance schema.")
            if envelope.get("algorithm") != "hmac-sha256":
                raise ValueError("Unsupported provenance signature algorithm.")
            canonical_json(payload)
            if (vertices is None) != (faces is None):
                raise ValueError("Both geometry arrays are required for verification.")
            if vertices is not None and geometry_sha256(vertices, faces) != payload["geometry_sha256"]:
                _log_event(logging.WARNING, "Geometry does not match the recorded model provenance")
                return {"status": "tampered", "reason": "Geometry does not match the recorded model provenance."}
            key = self._key()
            if key is None or hashlib.sha256(key).hexdigest()[:24] != envelope.get("key_id"):
                return {"status": "unverified", "reason": "The original local signing key is unavailable on this profile."}
            signature = envelope.get("signature")
            if not isinstance(signature, str) or not signature.isascii() or not hmac.compare_digest(
                signature, hmac.new(key, canonical_json(payload), hashlib.sha256).hexdigest()
            ):
                _log_event(logging.WARNING, "Model provenance or its signature has changed")
                return {"status": "tampered", "reason": "Model provenance or its signature has changed."}
            return {"status": "verified", "reason": "Recorded provenance matches this profile's local signature."
                    + (" Geometry matches." if vertices is not None else " Geometry has not been checked.")}
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            _log_event(logging.ERROR, "Could not verify mesh provenance", exc_info=True)
            return {"status": "unverified", "reason": "Provenance or the local signing key is missing, invalid, or unreadable."}
