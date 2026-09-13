"""DA3 worker. Run only with the separate backend Python interpreter."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile

MODEL_IDS = frozenset({
    "depth-anything/DA3-SMALL", "depth-anything/DA3-BASE",
    "depth-anything/DA3MONO-LARGE", "depth-anything/DA3METRIC-LARGE",
})
SHA = re.compile(r"^[0-9a-f]{40}$")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="da3-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def prepare_snapshot(request):
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache

    model_id = request["model_id"]
    if model_id not in MODEL_IDS:
        raise ValueError("Unsupported DA3 model identity.")
    manifest = Path(request["root"]) / "models" / "da3" / (model_id.split("/")[1] + ".json")
    recorded = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    revision = request.get("revision") or recorded.get("revision")
    allow_download = request.get("allow_download", False)
    cache_dir = request.get("cache_dir")
    if revision is None:
        cached = try_to_load_from_cache(model_id, "config.json", cache_dir=cache_dir)
        if isinstance(cached, str) and "snapshots" in Path(cached).parts:
            parts = Path(cached).parts
            revision = parts[parts.index("snapshots") + 1]
        elif allow_download:
            revision = HfApi().model_info(model_id).sha
        else:
            raise RuntimeError("DA3 is not cached. Enable downloads and prepare this model first.")
    if not isinstance(revision, str) or not SHA.fullmatch(revision):
        raise ValueError("DA3 requires a full immutable checkpoint revision.")
    snapshot = Path(snapshot_download(
        model_id, revision=revision, cache_dir=cache_dir,
        local_files_only=not allow_download,
        allow_patterns=["config.json", "model.safetensors"],
    ))
    hashes = {name: digest(snapshot / name) for name in ("config.json", "model.safetensors")}
    if recorded.get("revision") == revision and recorded.get("weight_files_sha256") != hashes:
        raise ValueError("DA3 cached files differ from their recorded hashes. Restore the checkpoint before use.")
    metadata = {"revision": revision, "weight_files_sha256": hashes}
    return snapshot, manifest, metadata


def padded_image(image, resolution=504):
    """Resize proportionally and pad to patch multiples; retain the crop bounds."""
    import numpy as np
    from PIL import Image

    h, w = image.shape[:2]
    scale = resolution / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.Resampling.LANCZOS))
    pad_h, pad_w = (-new_h) % 14, (-new_w) % 14
    return np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge"), (new_h, new_w)


def run(request):
    # Import before downloading so missing runtime dependencies fail early.
    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3

    snapshot, manifest, metadata = prepare_snapshot(request)
    model = DepthAnything3.from_pretrained(str(snapshot), local_files_only=True)
    model = model.to(request["device"]).eval()
    if request["action"] == "infer":
        from PIL import Image

        image = np.load(request["input"], allow_pickle=False)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError("DA3 input must be a nonempty RGB image.")
        padded, (h, w) = padded_image(image)
        # Padding makes both dimensions patch multiples before upstream processing.
        with torch.inference_mode():
            prediction = model.inference(
                [padded], process_res=504, process_res_method="upper_bound_resize",
                infer_gs=False, export_dir=None,
            )
        values = np.asarray(prediction.depth, dtype=np.float32)
        if values.shape != (1, *padded.shape[:2]) or not np.isfinite(values).all():
            raise ValueError("DA3 returned an unexpected depth shape or nonfinite values.")
        depth = np.asarray(Image.fromarray(values[0, :h, :w]).resize(
            (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR,
        ), dtype=np.float32)
        # Relief processing intentionally normalizes this, including Metric-Large.
        np.save(request["depth_output"], depth, allow_pickle=False)
    elif request["action"] != "prepare":
        raise ValueError("Unsupported DA3 worker operation.")
    write_json(manifest, metadata)
    write_json(request["output"], metadata)


def main():
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        logs = Path(request["root"]) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=logs / "da3-worker.log", level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
        run(request)
        return 0
    except Exception:
        logging.exception("Depth Anything 3 worker failed")
        # The parent captures this and records it through log_utils.get_logger.
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
