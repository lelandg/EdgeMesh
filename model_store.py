"""Lazy, offline-first model pairs with recorded immutable identities.

HF downloads require an explicit caller opt-in. Legacy MiDaS source and weights
must be registered locally; moving torch.hub branch names are never identities.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import tempfile
import threading

from model_licensing import canonical_model_identity, policy_digest, policy_for
from da3_backend import DA3_ALIASES, DA3_MODELS


HF_MODELS = {
    "depth_anything_v1": ("LiheYoung/depth-anything-large-hf", "apache-2.0"),
    "depth_anything_v2": ("depth-anything/Depth-Anything-V2-Large-hf", "cc-by-nc-4.0"),
    "depth_pro": ("apple/DepthPro-hf", "apple-amlr; review repository license"),
    "sam2": ("facebook/sam2.1-hiera-tiny", "apache-2.0"),
}
ALIASES = {
    **DA3_ALIASES,
    "DepthAnythingV1": "depth_anything_v1",
    "DepthAnythingV2": "depth_anything_v2",
    "Depth Pro": "depth_pro",
    "MiDaS": "midas",
    "DPT": "dpt",
}
SHA = re.compile(r"^[0-9a-f]{40}$")
_LEGACY_LOAD_LOCK = threading.RLock()


class ModelSetupError(RuntimeError):
    """A model cannot be loaded with a trustworthy recorded identity."""


class ModelPreparationCancelled(RuntimeError):
    """A caller cancelled between model preparation stages."""


class ModelLicenseConsentRequired(ModelSetupError):
    """A restricted download needs an explicit, model-specific acknowledgment."""


def _check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise ModelPreparationCancelled("Model preparation cancelled.")


def _default_root():
    from user_state import UserPaths

    return UserPaths.discover().root


def _digest(path, cancelled=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            _check_cancel(cancelled)
            digest.update(chunk)
    return digest.hexdigest()


class ModelStore:
    """One shared store per app; no torch/transformers imports until requested.

    Returned metadata is JSON-serializable and independent of mutable cache state.
    Cancellation is cooperative; downloads and GPU work finish their current stage.
    """

    def __init__(self, root=None, cache_dir=None, consent_callback=None):
        self.root = Path(root) if root is not None else _default_root()
        self.manifest_path = self.root / "models" / "manifest.json"
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self._pairs = {}
        self._lock = threading.RLock()
        self.consent_callback = consent_callback
        self._pending_license_acceptance = {}
        self._manifest = self._read_manifest()

    def _read_manifest(self):
        if not self.manifest_path.exists():
            return {"schema_version": 1, "hf": {}, "midas": {}}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if (
                data.get("schema_version") != 1
                or not isinstance(data.get("hf"), dict)
                or not isinstance(data.get("midas"), dict)
            ):
                raise ValueError("unsupported schema")
            return data
        except (OSError, ValueError, AttributeError) as error:
            raise ModelSetupError(
                f"Cannot read model manifest {self.manifest_path}: {error}. Preserve it for diagnosis before restoring a known-good copy."
            ) from error

    def _save_manifest(self):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix="manifest-", suffix=".json", dir=self.manifest_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self._manifest, stream, indent=2)
                stream.write("\n")
            os.replace(name, self.manifest_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _local_revision(self, model_type):
        """Inspect existing metadata/cache only; never ask the Hub for a pin."""
        model_type = ALIASES.get(model_type, model_type)
        if model_type not in HF_MODELS:
            return None
        model_id = HF_MODELS[model_type][0]
        recorded = self._manifest["hf"].get(model_id, {})
        revision = recorded.get("revision") if isinstance(recorded, dict) else None
        if isinstance(revision, str) and SHA.fullmatch(revision):
            return revision
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return None
        cached = try_to_load_from_cache(model_id, "config.json", cache_dir=self.cache_dir, revision="main")
        if isinstance(cached, str):
            parts = Path(cached).parts
            if "snapshots" in parts:
                index = parts.index("snapshots")
                if len(parts) > index + 1 and SHA.fullmatch(parts[index + 1]):
                    return parts[index + 1]
        return None

    @staticmethod
    def _consent_key(policy, revision):
        return hashlib.sha256(
            f"{policy.model_id}\n{revision}\n{policy_digest(policy)}".encode("utf-8")
        ).hexdigest()

    def has_license_consent(self, model_type, revision=None):
        model_type = ALIASES.get(model_type, model_type)
        policy = policy_for(HF_MODELS[model_type][0] if model_type in HF_MODELS else model_type)
        if not policy.requires_acceptance:
            return True
        with self._lock:
            revision = revision or self._local_revision(model_type)
            if revision is None:
                return False
            consents = self._manifest.get("license_consents", {})
            record = consents.get(self._consent_key(policy, revision), {}) if isinstance(consents, dict) else {}
            return isinstance(record, dict) and record.get("accepted") is True

    def consent_request(self, model_type):
        """Local-only GUI preflight; display before starting a worker/download.

        A missing revision means this first preparation will resolve it. An
        acknowledgment in that case lasts only until the next online preparation
        of this model in this process; persistence starts once its pin is known.
        """
        model_type = ALIASES.get(model_type, model_type)
        policy = policy_for(HF_MODELS[model_type][0] if model_type in HF_MODELS else model_type)
        with self._lock:
            revision = self._local_revision(model_type)
            return {
                **policy.as_dict(), "revision": revision,
                "policy_sha256": policy_digest(policy),
                "consent_needed": model_type in HF_MODELS and policy.requires_acceptance
                and not self.has_license_consent(model_type, revision),
            }

    def accept_license(self, model_type, revision=None):
        """Call only after the user explicitly accepts the displayed policy.

        With no pin yet, queue one in-memory approval; do not persist an approval
        that would silently cover arbitrary future model revisions.
        """
        model_type = ALIASES.get(model_type, model_type)
        if model_type not in HF_MODELS:
            raise ValueError("Only a supported download model can record license consent.")
        policy = policy_for(HF_MODELS[model_type][0])
        with self._lock:
            revision = revision or self._local_revision(model_type)
            if revision is None:
                self._pending_license_acceptance[model_type] = policy_digest(policy)
                return
            if not isinstance(revision, str) or not SHA.fullmatch(revision):
                raise ValueError("License consent requires a full immutable revision.")
            record = {
                "model_id": policy.model_id, "revision": revision,
                "license": policy.license, "policy_sha256": policy_digest(policy), "accepted": True,
            }
            consents = self._manifest.setdefault("license_consents", {})
            if not isinstance(consents, dict):
                raise ModelSetupError("Invalid license consent records in model manifest.")
            consents[self._consent_key(policy, revision)] = record
            self._save_manifest()

    def _require_license_consent(self, model_type, revision, cancelled):
        policy = policy_for(HF_MODELS[model_type][0])
        pending = self._pending_license_acceptance.pop(model_type, None)
        if not policy.requires_acceptance or self.has_license_consent(model_type, revision):
            return
        _check_cancel(cancelled)
        accepted = pending == policy_digest(policy)
        if not accepted and self.consent_callback is not None:
            accepted = self.consent_callback({**policy.as_dict(), "revision": revision,
                                             "policy_sha256": policy_digest(policy)}) is True
        _check_cancel(cancelled)
        if not accepted:
            raise ModelLicenseConsentRequired(
                f"Review and explicitly accept {policy.display_name}'s {policy.license} weight terms "
                f"before downloading revision {revision}. Open Models / Setup and prepare the model "
                "after reviewing its license notice. Download permission alone is not license acknowledgment."
            )
        self.accept_license(model_type, revision)

    def _weight_hashes(self, model_id, revision, cancelled):
        """Hash cached checkpoint files when present; never fetch missing files."""
        from huggingface_hub import try_to_load_from_cache

        filenames = {"model.safetensors", "pytorch_model.bin"}
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            cached = try_to_load_from_cache(model_id, index_name, cache_dir=self.cache_dir, revision=revision)
            if isinstance(cached, str) and Path(cached).is_file():
                try:
                    index = json.loads(Path(cached).read_text(encoding="utf-8"))
                    filenames.update(index.get("weight_map", {}).values())
                except (OSError, ValueError, AttributeError, TypeError):
                    from log_utils import get_logger
                    get_logger().exception("Could not read cached model weight index for provenance")
        result = {}
        for filename in sorted(name for name in filenames if isinstance(name, str)):
            _check_cancel(cancelled)
            # Cache APIs validate names, but indexes are untrusted local data.
            normalized = filename.replace("\\", "/")
            if PureWindowsPath(filename).anchor or Path(filename).is_absolute() or ".." in normalized.split("/"):
                continue
            cached = try_to_load_from_cache(model_id, filename, cache_dir=self.cache_dir, revision=revision)
            if isinstance(cached, str) and Path(cached).is_file():
                result[filename] = _digest(cached, cancelled)
        return result

    def _hf_revision(self, model_id, allow_download, cancelled):
        recorded = self._manifest["hf"].get(model_id)
        if recorded is not None:
            if not isinstance(recorded, dict) or not SHA.fullmatch(
                str(recorded.get("revision", ""))
            ):
                raise ModelSetupError(
                    f"Invalid immutable revision recorded for {model_id}."
                )
            return recorded["revision"]
        from huggingface_hub import HfApi, try_to_load_from_cache

        _check_cancel(cancelled)
        cached = try_to_load_from_cache(
            model_id, "config.json", cache_dir=self.cache_dir, revision="main"
        )
        revision = None
        if isinstance(cached, str):
            parts = Path(cached).parts
            if "snapshots" in parts:
                index = parts.index("snapshots")
                if len(parts) > index + 1 and SHA.fullmatch(parts[index + 1]):
                    revision = parts[index + 1]
        if revision is None and allow_download:
            revision = HfApi().model_info(model_id).sha
        _check_cancel(cancelled)
        if not isinstance(revision, str) or not SHA.fullmatch(revision):
            raise ModelSetupError(
                f"No pinned cached revision for {model_id}. Review https://huggingface.co/{model_id}, then enable model downloads for one preparation run. Offline mode never contacts the Hub."
            )
        self._manifest["hf"][model_id] = {"revision": revision}
        self._save_manifest()
        return revision

    def pin_revision(self, model_type, revision):
        """Select an explicit known HF SHA (e.g. restoring a saved session)."""
        model_type = ALIASES.get(model_type, model_type)
        if (
            model_type not in HF_MODELS
            or not isinstance(revision, str)
            or not SHA.fullmatch(revision)
        ):
            raise ValueError(
                "A supported Hugging Face model and full 40-character commit SHA are required."
            )
        with self._lock:
            self._manifest["hf"][HF_MODELS[model_type][0]] = {"revision": revision}
            self._save_manifest()

    def _get_hf(self, model_type, device, allow_download, cancelled, preprocessing):
        _check_cancel(cancelled)
        model_id, license_name = HF_MODELS[model_type]
        revision = self._hf_revision(model_id, allow_download, cancelled)
        if allow_download:
            self._require_license_consent(model_type, revision, cancelled)
        preprocessing = dict(preprocessing or {})
        reserved = {"revision", "local_files_only", "trust_remote_code", "cache_dir"}
        if reserved.intersection(preprocessing):
            raise ValueError(
                "Preprocessing settings cannot override model identity or offline policy."
            )
        key = (
            model_id,
            revision,
            str(device),
            json.dumps(preprocessing, sort_keys=True),
        )
        if key not in self._pairs:
            if model_type == "sam2":
                from transformers import Sam2Model, Sam2Processor

                model_class, processor_class = Sam2Model, Sam2Processor
            else:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                model_class, processor_class = (
                    AutoModelForDepthEstimation,
                    AutoImageProcessor,
                )
            kwargs = {
                "revision": revision,
                "local_files_only": not allow_download,
                "trust_remote_code": False,
            }
            if self.cache_dir is not None:
                kwargs["cache_dir"] = self.cache_dir
            try:
                _check_cancel(cancelled)
                processor = processor_class.from_pretrained(
                    model_id, **kwargs, **preprocessing
                )
                _check_cancel(cancelled)
                model = (
                    model_class.from_pretrained(model_id, **kwargs).to(device).eval()
                )
                _check_cancel(cancelled)
            except OSError as error:
                raise ModelSetupError(
                    f"Could not prepare {model_id} at {revision}: {error}. {'Enable downloads after reviewing the model license if these exact files are not cached.' if not allow_download else 'Check connectivity, disk space, and model access.'}"
                ) from error
            metadata = canonical_model_identity({
                "backend": "huggingface",
                "model_type": model_type,
                "model_id": model_id,
                "revision": revision,
                "device": str(device),
                "preprocessing": preprocessing,
                "license": license_name,
                "model_card": f"https://huggingface.co/{model_id}/tree/{revision}",
                "weight_files_sha256": self._weight_hashes(model_id, revision, cancelled),
            })
            self._pairs[key] = model, processor, metadata
        model, processor, metadata = self._pairs[key]
        _check_cancel(cancelled)
        return model, processor, json.loads(json.dumps(metadata))

    @staticmethod
    def _source_revision(source_dir):
        def git(*args):
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={Path(source_dir).as_posix()}",
                    "-C",
                    str(source_dir),
                    *args,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()

        try:
            revision = git("rev-parse", "HEAD")
            if not SHA.fullmatch(revision) or git(
                "status", "--porcelain", "--untracked-files=all"
            ):
                raise ModelSetupError(
                    "MiDaS source must be a clean git checkout at a full immutable commit; commit or relocate its local changes first."
                )
            if not (Path(source_dir) / "hubconf.py").is_file():
                raise ModelSetupError("MiDaS source directory must contain hubconf.py.")
            return revision
        except subprocess.CalledProcessError as error:
            raise ModelSetupError(
                "Cannot verify the local MiDaS git checkout. Verify its ownership, git metadata, and source location."
            ) from error

    def register_midas(self, model_type, source_dir, weights_path, *, cancelled=None):
        """Record user-provided local source and checkpoint, without downloading."""
        model_type = ALIASES.get(model_type, model_type)
        if model_type not in ("midas", "dpt"):
            raise ValueError("Only midas or dpt can use a MiDaS source registration.")
        source_dir, weights_path = (
            Path(source_dir).resolve(),
            Path(weights_path).resolve(),
        )
        revision = self._source_revision(source_dir)
        checksum = _digest(weights_path, cancelled)
        with self._lock:
            _check_cancel(cancelled)
            self._manifest["midas"][model_type] = {
                "source_dir": str(source_dir),
                "revision": revision,
                "weights_path": str(weights_path),
                "weights_sha256": checksum,
            }
            self._save_manifest()

    def _get_midas(self, model_type, device, cancelled):
        record = self._manifest["midas"].get(model_type)
        if not isinstance(record, dict):
            raise ModelSetupError(
                f"{model_type} needs a pinned local MiDaS source checkout and matching checkpoint. Open Models → Register local MiDaS / DPT… after obtaining and reviewing those files, or select Depth Anything V2 / Depth Pro. Unpinned torch.hub main downloads are disabled."
            )
        if (
            not all(
                isinstance(record.get(field), str)
                for field in (
                    "source_dir",
                    "revision",
                    "weights_path",
                    "weights_sha256",
                )
            )
            or not SHA.fullmatch(record["revision"])
            or not re.fullmatch(r"[0-9a-f]{64}", record["weights_sha256"])
        ):
            raise ModelSetupError(
                "Invalid local MiDaS identity record. Register the intended source and checkpoint again."
            )
        _check_cancel(cancelled)
        if (
            self._source_revision(record["source_dir"]) != record["revision"]
            or _digest(record["weights_path"], cancelled) != record["weights_sha256"]
        ):
            raise ModelSetupError(
                "MiDaS source revision or checkpoint checksum changed. Restore the registered files or explicitly register the intended new identity."
            )
        key = (model_type, record["revision"], record["weights_sha256"], str(device))
        if key not in self._pairs:
            import torch

            architecture = "MiDaS" if model_type == "midas" else "DPT_Large"
            model, processor, adapter = self._load_legacy_architecture(
                record, model_type
            )
            _check_cancel(cancelled)
            model.load_state_dict(
                torch.load(
                    record["weights_path"], map_location="cpu", weights_only=True
                )
            )
            model = model.to(device).eval()
            _check_cancel(cancelled)
            metadata = canonical_model_identity({
                "backend": "torch_hub",
                "model_type": model_type,
                "model_id": "intel-isl/MiDaS:" + architecture,
                "revision": record["revision"],
                "weights_sha256": record["weights_sha256"],
                "device": str(device),
                "preprocessing": "default_transform"
                if model_type == "midas"
                else "dpt_transform",
                "backbone_adapter": adapter,
            })
            self._pairs[key] = model, processor, metadata
        model, processor, metadata = self._pairs[key]
        return model, processor, dict(metadata)

    @staticmethod
    def _load_legacy_architecture(record, model_type):
        """Avoid MiDaS v2.1's nested, unpinned WSL-Images hub download.

        WSL's resnext101_32x8d architecture is the torchvision architecture.
        The full registered MiDaS state dict supplies every backbone weight;
        constructor weights are neither needed nor downloaded. The helper swap
        is confined to construction under a process-wide legacy loader lock.
        """
        import torch

        with _LEGACY_LOAD_LOCK:
            existing = sys.modules.get("midas.blocks")
            if (
                existing is not None
                and getattr(existing, "_edgemesh_revision", None) != record["revision"]
            ):
                raise ModelSetupError(
                    "Another MiDaS source revision is already imported. Restart EdgeMesh after changing the registered MiDaS source."
                )
            transforms = torch.hub.load(
                record["source_dir"], "transforms", source="local"
            )
            blocks = importlib.import_module("midas.blocks")
            if (
                not Path(blocks.__file__)
                .resolve()
                .is_relative_to(Path(record["source_dir"]).resolve())
            ):
                raise ModelSetupError(
                    "Imported MiDaS code does not match the registered source directory. Restart EdgeMesh with the intended source."
                )
            blocks._edgemesh_revision = record["revision"]
            if model_type == "dpt":
                return (
                    torch.hub.load(
                        record["source_dir"],
                        "DPT_Large",
                        source="local",
                        pretrained=False,
                    ),
                    transforms.dpt_transform,
                    "upstream-dpt",
                )
            import torchvision

            original = blocks._make_pretrained_resnext101_wsl

            def local_backbone(use_pretrained):
                return blocks._make_resnet_backbone(
                    torchvision.models.resnext101_32x8d(weights=None)
                )

            try:
                blocks._make_pretrained_resnext101_wsl = local_backbone
                model = torch.hub.load(
                    record["source_dir"], "MiDaS", source="local", pretrained=False
                )
            finally:
                blocks._make_pretrained_resnext101_wsl = original
            return (
                model,
                transforms.default_transform,
                f"torchvision-{torchvision.__version__}:resnext101_32x8d:full-checkpoint-v1",
            )

    def get_depth(
        self,
        model_type,
        device,
        allow_download=False,
        *,
        cancelled=None,
        preprocessing=None,
    ):
        model_type = ALIASES.get(model_type, model_type)
        _check_cancel(cancelled)
        with self._lock:
            _check_cancel(cancelled)
            if model_type in ("midas", "dpt"):
                return self._get_midas(model_type, device, cancelled)
            if model_type in DA3_MODELS:
                from da3_backend import load_da3

                # The separate process owns inference memory. Recheck its runtime
                # and pinned files for each preparation instead of caching a model.
                return load_da3(
                    model_type, device, root=self.root, cache_dir=self.cache_dir,
                    allow_download=allow_download, cancelled=cancelled,
                )
            if model_type not in ("depth_anything_v1", "depth_anything_v2", "depth_pro"):
                raise ModelSetupError(f"Unsupported depth model: {model_type}")
            return self._get_hf(
                model_type, device, allow_download, cancelled, preprocessing
            )

    def get_sam2(self, device, allow_download=False, *, cancelled=None):
        _check_cancel(cancelled)
        with self._lock:
            return self._get_hf("sam2", device, allow_download, cancelled, None)

    def clear(self):
        """Release stored model references; callers must release their own pairs."""
        with self._lock:
            self._pairs.clear()

    unload = clear

    def identities(self):
        with self._lock:
            return json.loads(json.dumps([pair[2] for pair in self._pairs.values()]))
