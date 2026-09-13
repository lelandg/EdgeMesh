"""Optional Depth Anything 3 process adapter, separate from the GUI runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

DA3_MODELS = {
    "depth_anything_3_small": ("depth-anything/DA3-SMALL", "apache-2.0"),
    "depth_anything_3_base": ("depth-anything/DA3-BASE", "apache-2.0"),
    "depth_anything_3_mono_large": ("depth-anything/DA3MONO-LARGE", "apache-2.0"),
    "depth_anything_3_metric_large": ("depth-anything/DA3METRIC-LARGE", "apache-2.0"),
}
DA3_ALIASES = {
    "Depth Anything 3 Small": "depth_anything_3_small",
    "Depth Anything 3 Base": "depth_anything_3_base",
    "Depth Anything 3 Mono-Large": "depth_anything_3_mono_large",
    "Depth Anything 3 Metric-Large": "depth_anything_3_metric_large",
}


def runtime_python(root):
    """Use an explicitly selected runtime or the managed per-user environment."""
    override = os.environ.get("EDGEMESH_DA3_PYTHON")
    if override:
        return Path(override).expanduser().resolve()
    folder = Path(root) / "runtimes" / "depth-anything-3"
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _check_cancel(cancelled):
    if cancelled is not None and cancelled():
        from model_store import ModelPreparationCancelled
        raise ModelPreparationCancelled("Depth Anything 3 operation cancelled.")


def _run(request, root, cancelled=None, rgb_image=None):
    import numpy as np
    from log_utils import get_logger
    from model_store import ModelSetupError

    _check_cancel(cancelled)
    python = runtime_python(root)
    if not python.is_file():
        message = (
            "Depth Anything 3 needs its separate Python runtime. "
            "Run the setup in docs/Depth_Anything_3.html, or set EDGEMESH_DA3_PYTHON "
            "to an existing DA3 environment's Python executable."
        )
        get_logger().error(message)
        raise ModelSetupError(message)
    work = Path(root) / "work"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="da3-", dir=work) as directory:
        folder = Path(directory)
        request = dict(request, root=str(Path(root).resolve()))
        if rgb_image is not None:
            np.save(folder / "image.npy", rgb_image, allow_pickle=False)
            request["input"] = str(folder / "image.npy")
        request["output"] = str(folder / "result.json")
        request["depth_output"] = str(folder / "depth.npy")
        request_file = folder / "request.json"
        request_file.write_text(json.dumps(request), encoding="utf-8")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        environment["PYTHONNOUSERSITE"] = "1"
        if not request.get("allow_download", False):
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
        command = [str(python), "-I", str(Path(__file__).with_name("da3_worker.py")), str(request_file)]
        try:
            with (folder / "worker.log").open("w+b") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    env=environment, cwd=folder,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                try:
                    while True:
                        _check_cancel(cancelled)
                        try:
                            code = process.wait(timeout=0.1)
                            break
                        except subprocess.TimeoutExpired:
                            continue
                    _check_cancel(cancelled)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 16000))
                details = log.read().decode("utf-8", errors="replace")
            if code:
                get_logger().error("Depth Anything 3 worker failed (%s): %s", code, details)
                raise ModelSetupError(
                    "Depth Anything 3 could not complete the operation. "
                    "Check the per-user log and docs/Depth_Anything_3.html for runtime setup."
                )
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            depth = None
            if rgb_image is not None:
                depth = np.load(folder / "depth.npy", allow_pickle=False)
                if depth.ndim != 2 or depth.size == 0 or not np.isfinite(depth).all():
                    raise ValueError("DA3 returned an invalid depth map.")
                depth = depth.astype(np.float32)
            return result, depth
        except (OSError, ValueError) as error:
            get_logger().exception("Depth Anything 3 runtime operation failed")
            raise ModelSetupError(f"Depth Anything 3 runtime failed: {error}") from error


class DA3DepthModel:
    """Small proxy; each job releases its worker's model and GPU allocation."""

    def __init__(self, request, root):
        self.request = dict(request)
        self.root = Path(root)

    def predict_depth(self, rgb_image, cancelled=None):
        import numpy as np
        image = np.asarray(rgb_image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError("DA3 expects a nonempty uint8 RGB image.")
        # Generation uses the prepared immutable snapshot and cannot download.
        request = dict(self.request, action="infer", allow_download=False)
        _, depth = _run(request, self.root, cancelled, image)
        return depth


def load_da3(model_type, device, *, root, cache_dir=None, allow_download=False, cancelled=None):
    """Prepare a pinned snapshot and return the standard ModelStore tuple."""
    from model_licensing import canonical_model_identity

    model_type = DA3_ALIASES.get(model_type, model_type)
    model_id, license_name = DA3_MODELS[model_type]
    request = {
        "action": "prepare", "model_id": model_id, "device": str(device),
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "allow_download": bool(allow_download),
    }
    result, _ = _run(request, root, cancelled)
    request["revision"] = result["revision"]
    metadata = canonical_model_identity({
        **result, "backend": "depth_anything_3", "model_type": model_type,
        "model_id": model_id, "license": license_name, "device": str(device),
        "preprocessing": {"process_res": 504, "aspect_ratio": "resize_and_pad",
                          "relief_transform": "normalized_inverse_distance"},
        "depth_semantics": "nearer_is_higher_relative_relief; metric scale is not retained",
        "model_card": f"https://huggingface.co/{model_id}/tree/{result['revision']}",
    })
    return DA3DepthModel(request, root), None, metadata
