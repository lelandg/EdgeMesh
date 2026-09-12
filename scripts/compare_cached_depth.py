"""Measure cached DA1/DA2 models using the same small, offline CPU protocol."""
import argparse
import configparser
import hashlib
from contextlib import redirect_stdout
import inspect
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def statistics(value):
    import numpy as np
    value = np.asarray(value)
    return {"shape": list(value.shape), "finite": bool(np.isfinite(value).all()),
            "minimum": float(value.min()), "maximum": float(value.max()),
            "standard_deviation": float(value.std()),
            "percentiles_5_50_95": np.percentile(value, [5, 50, 95]).tolist()}


def main(argv=None):
    entered = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", choices=("depth_anything_v1", "depth_anything_v2"), default="depth_anything_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--preview-config", type=Path, help="Apply saved UI preview settings before inference and report pixel differences")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--inference-size", type=int, default=112)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--depth-amount", type=float, default=1.0)
    parser.add_argument("--smoothing", choices=("none", "anisotropic", "gaussian", "bilateral", "median"), default="none")
    args = parser.parse_args(argv)
    if args.width < 2 or args.threads < 1:
        parser.error("width must be at least 2 and threads must be positive")
    if args.inference_size < 56 or args.inference_size % 14:
        parser.error("inference-size must be a multiple of 14, at least 56")
    if not math.isfinite(args.depth_amount) or not 0 <= args.depth_amount <= 100:
        parser.error("depth-amount must be finite and between 0 and 100")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    from log_utils import setup_logger
    logger = setup_logger(f"depth_comparison.{args.output_dir.resolve()}", log_file=args.output_dir / "diagnostic.log")
    raw_stats = {}
    pipeline = None
    try:
        print("Loading local depth runtime...", file=sys.stderr, flush=True)
        with redirect_stdout(sys.stderr):
            import cv2
            import numpy as np
            import torch
            import trimesh
            from data_contracts import as_bgr, proportional_shape
            from depth_to_3d import DepthTo3D
            from model_licensing import ProvenanceStore
            from model_store import ModelStore

        image = cv2.imread(str(args.input), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read input image: {args.input}")
        original = as_bgr(image)
        image = original
        input_processing = {
            "mode": "original",
            "original_pixels_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
            "settings": None,
        }
        if args.preview_config:
            from feature_workflows import preview_image
            config = configparser.ConfigParser()
            if not config.read(args.preview_config, encoding="utf-8"):
                raise FileNotFoundError(args.preview_config)
            section = config["UI_Settings"]
            settings = {
                "use_processed_image_enabled": section.getboolean("use_processed_image"),
                "project_on_original": section.getboolean("project_on_original"),
                "blend_amount": section.getfloat("blend_amount"),
                "edge_detection_enabled": section.getboolean("edge_detection"),
                "grayscale_enabled": section.getboolean("grayscale"),
                "invert_colors_enabled": section.getboolean("invert_colors"),
                "sensitivity": section.getint("sensitivity"),
                "line_thickness": section.getint("line_thickness"),
            }
            if any(value is None for value in settings.values()):
                raise ValueError("Saved preview configuration is missing a required processing field")
            image = preview_image(original.copy(), settings)
            difference = np.abs(image.astype(np.int16) - original.astype(np.int16))
            input_processing.update({
                "mode": "saved_preview", "settings": settings,
                "identical_to_original": bool(np.array_equal(image, original)),
                "changed_pixels": int(np.count_nonzero(np.any(image != original, axis=-1))),
                "total_pixels": int(original.shape[0] * original.shape[1]),
                "mean_absolute_channel_difference": float(difference.mean()),
                "max_absolute_channel_difference": int(difference.max()),
            })
            if not cv2.imwrite(str(args.output_dir / "prepared_input.png"), image):
                raise OSError("Could not save the prepared diagnostic input")
        input_processing["inferred_pixels_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
        shape = proportional_shape(image.shape, args.width)
        if min(shape) < 2:
            raise ValueError("The proportional mesh must have at least two rows and columns")
        torch.set_num_threads(args.threads)
        preprocessing = {"size": {"height": args.inference_size, "width": args.inference_size},
                         "keep_aspect_ratio": True, "ensure_multiple_of": 14}
        state_root = args.output_dir / "model-state"
        state_root.mkdir(parents=True, exist_ok=True)
        # The manifest is reproducible from report.json; the local signing key
        # must never be staged when a diagnostic directory lives in a checkout.
        ignore_path = state_root / ".gitignore"
        if not ignore_path.exists():
            ignore_path.write_text("*\n", encoding="utf-8")

        class DiagnosticPipeline(DepthTo3D):
            def load_model(self):
                self.device = torch.device("cpu")
                model, processor, self.model_info = self.model_store.get_depth(
                    self.model_type, self.device, allow_download=False,
                    preprocessing=preprocessing)
                return model, processor

        started = time.perf_counter()
        with redirect_stdout(sys.stderr):
            print(f"Preparing cached {args.model} on CPU...", flush=True)
            store = ModelStore(root=state_root, cache_dir=args.cache_dir)
            # Use explicit CPU when the installed pipeline supports it; the
            # diagnostic loader selects CPU in either case.
            kwargs = {"device": "cpu"} if "device" in inspect.signature(DepthTo3D).parameters else {}
            pipeline = DiagnosticPipeline(model_type=args.model, verbose=False, model_store=store, **kwargs)
            loaded = time.perf_counter()
            print(f"Estimating depth with a {args.inference_size}px processor and {shape[1]}x{shape[0]} mesh...", flush=True)

            def capture_prediction(_model, _inputs, output):
                prediction = getattr(output, "predicted_depth", output)
                raw_stats.update(statistics(prediction.detach().cpu().numpy()))

            hook = pipeline.model.register_forward_hook(capture_prediction)
            try:
                mesh_path, _ = pipeline.process_image(
                    args.input, image_data=image, target_size=shape,
                    smoothing_method=args.smoothing, depth_amount=args.depth_amount,
                    flat_back=True, output_dir=args.output_dir)
            finally:
                hook.remove()
            mesh = trimesh.load_mesh(mesh_path)
            finished = time.perf_counter()
            provenance = ProvenanceStore(root=state_root).seal(
                np.asarray(mesh.vertices), np.asarray(mesh.faces), [pipeline.model_info],
                parameters={"depth_amount": args.depth_amount, "smoothing": args.smoothing,
                            "output_shape": list(shape), "processor_override": preprocessing, "input_processing": input_processing,
                            "subject_mask": None, "flat_back": True})
        report = {
            "schema_version": 1, "status": "ok", "model": pipeline.model_info,
            "input": str(args.input.resolve()), "input_shape": list(image.shape),
            "output_shape": list(shape), "processor_override": preprocessing, "input_processing": input_processing,
            "device": "cpu", "smoothing": args.smoothing, "depth_amount": args.depth_amount,
            "raw_prediction": raw_stats,
            "smoothed_normalized_depth": statistics(pipeline.depth_map),
            "mesh": {"path": str(Path(mesh_path).resolve()), "vertices": len(mesh.vertices),
                     "faces": len(mesh.faces), "extents": mesh.extents.tolist(),
                     "depth_to_longest_side": float(mesh.extents[2] / max(mesh.extents[:2])),
                     "watertight": bool(mesh.is_watertight)},
            "provenance": provenance,
            "seconds": {"runtime_setup": started - entered, "model_load": loaded - started,
                        "pipeline": finished - loaded, "total": time.perf_counter() - entered},
            "offline": True,
        }
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        logger.exception("Cached depth comparison failed")
        failure = {"status": "error", "model": getattr(pipeline, "model_info", {}),
                   "raw_prediction": raw_stats, "error_type": type(error).__name__, "error": str(error)}
        (args.output_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())