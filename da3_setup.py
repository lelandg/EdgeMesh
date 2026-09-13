"""Install the optional DA3 runtime without changing EdgeMesh dependencies."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
SOURCE_URL = f"https://github.com/ByteDance-Seed/Depth-Anything-3/archive/{SOURCE_REVISION}.zip"


def install(root, uv, python="3.12"):
    from log_utils import get_logger

    root = Path(root).resolve()
    folder = root / "runtimes" / "depth-anything-3"
    executable = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = folder / "edgemesh-runtime.json"
    if folder.exists() and not marker.exists():
        raise RuntimeError(f"Existing unmanaged environment at {folder}. Select a different --root.")
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(root / "cache" / "uv-da3")
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with (logs / "da3-setup.log").open("a", encoding="utf-8") as log:
        def command(args):
            get_logger().info("Depth Anything 3 setup: %s", args)
            subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=environment,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

        if not executable.exists():
            command([uv, "venv", "--python", python, str(folder)])
            marker.write_text(json.dumps({"source_revision": SOURCE_REVISION}), encoding="utf-8")
        # The source archive is immutable and older than seven days. uv enforces
        # the same minimum age on every resolved package, including build tools.
        command([uv, "pip", "install", "--python", str(executable),
                 "--exclude-newer", cutoff, SOURCE_URL, "opencv-python<4.12", "addict==2.4.0"])
        command([str(executable), "-I", "-c",
                 "from depth_anything_3.api import DepthAnything3; import torch; "
                 "print('DA3 runtime import passed; CUDA available:', torch.cuda.is_available())"])
        command([uv, "pip", "check", "--python", str(executable)])
        with (folder / "installed-packages.txt").open("w", encoding="utf-8") as packages:
            subprocess.run([uv, "pip", "freeze", "--python", str(executable)], check=True,
                           stdout=packages, stderr=log, env=environment,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    get_logger().info("Depth Anything 3 runtime ready at %s", executable)
    return executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="EdgeMesh data directory (default: per-user directory)")
    parser.add_argument("--uv", help="Path to an existing uv executable")
    parser.add_argument("--python", default="3.12", help="Python 3.12 executable or uv version selector")
    args = parser.parse_args()
    from log_utils import get_logger
    from user_state import UserPaths

    try:
        uv = args.uv or shutil.which("uv")
        if not uv:
            raise RuntimeError("uv is required. Install uv, then run DA3 setup again.")
        root = args.root or UserPaths.discover().root
        executable = install(root, uv, args.python)
        sys.stdout.write(f"Depth Anything 3 runtime ready: {executable}\n")
        return 0
    except Exception:
        get_logger().exception("Depth Anything 3 runtime setup failed")
        sys.stderr.write("Depth Anything 3 setup failed. See the per-user log and logs/da3-setup.log.\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
