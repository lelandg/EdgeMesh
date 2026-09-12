"""Run an offline, small CPU depth/mesh diagnostic and save measured evidence.

This is a developer research tool. It never downloads weights or changes the
application's model settings. Model processor defaults remain in effect unless
the caller explicitly chooses a smaller DA2 inference size.
"""
import argparse
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def statistics(value):
    import numpy as np

    value = np.asarray(value)
    return {
        "shape": list(value.shape),
        "finite": bool(np.isfinite(value).all()),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "standard_deviation": float(value.std()),
        "percentiles_5_50_95": np.percentile(value, [5, 50, 95]).tolist(),
    }


def main(argv=None):
    from edgemesh_bootstrap.runtime import initialize_platform_runtime

    initialize_platform_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source image; left unchanged")
    parser.add_argument("--model", default="depth_anything_v2",
                        choices=("depth_anything_v2", "depth_pro", "dpt", "midas"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model-state", type=Path,
                        help="Model registration state directory; defaults to the application's existing state")
    parser.add_argument("--width", type=int, default=96, help="Mesh width in pixels; aspect ratio is retained")
    parser.add_argument("--inference-size", type=int, default=0,
                        help="Optional DA2 processor size, a multiple of 14; zero keeps model defaults")
    parser.add_argument("--smoothing", default="none", choices=("none", "gaussian", "anisotropic", "bilateral", "median"))
    parser.add_argument("--depth-amount", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.width < 2 or args.threads < 1:
        parser.error("width must be at least 2 and threads must be positive")
    if not math.isfinite(args.depth_amount) or not 0 <= args.depth_amount <= 100:
        parser.error("depth-amount must be finite and between 0 and 100")
    if args.inference_size and (args.model != "depth_anything_v2" or args.inference_size < 56 or args.inference_size % 14):
        parser.error("inference-size supports DA2 only and must be a multiple of 14, at least 56")

    import cv2
    import torch
    import trimesh

    from data_contracts import proportional_shape
    from depth_to_3d import DepthTo3D
    from log_utils import get_logger
    from model_store import ModelStore

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(__name__)
    try:
        image = cv2.imread(str(args.input), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read input image: {args.input}")
        shape = proportional_shape(image.shape, args.width)
        if min(shape) < 2:
            raise ValueError("The proportional mesh must have at least two rows and columns")
        torch.set_num_threads(args.threads)
        preprocessing = {}
        if args.inference_size:
            preprocessing = {"size": {"height": args.inference_size, "width": args.inference_size},
                             "keep_aspect_ratio": True, "ensure_multiple_of": 14}

        class DiagnosticPipeline(DepthTo3D):
            def load_model(self):
                self.device = torch.device(args.device)
                model, processor, self.model_info = self.model_store.get_depth(
                    self.model_type, self.device, allow_download=False,
                    preprocessing=preprocessing)
                return model, processor

        started = time.perf_counter()
        raw_stats = {}
        with redirect_stdout(sys.stderr):
            store = ModelStore(root=args.model_state, cache_dir=args.cache_dir)
            pipeline = DiagnosticPipeline(model_type=args.model, verbose=False, model_store=store,
                                          device=args.device)
            loaded = time.perf_counter()

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
        report = {
            "model": pipeline.model_info,
            "input": str(args.input.resolve()),
            "input_shape": list(image.shape),
            "output_shape": list(shape),
            "processor_override": preprocessing,
            "device": args.device,
            "smoothing": args.smoothing,
            "depth_amount": args.depth_amount,
            "raw_prediction": raw_stats,
            "smoothed_normalized_depth": statistics(pipeline.depth_map),
            "mesh": {"path": str(Path(mesh_path).resolve()), "vertices": len(mesh.vertices),
                     "faces": len(mesh.faces), "extents": mesh.extents.tolist(),
                     "depth_to_longest_side": float(mesh.extents[2] / max(mesh.extents[:2])),
                     "watertight": bool(mesh.is_watertight)},
            "seconds": {"model_load": loaded - started, "pipeline": time.perf_counter() - loaded},
            "offline": True,
        }
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    except Exception:
        logger.exception("Offline depth diagnostic failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
