"""Console entry points for the desktop workspace and offline depth diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys


def _run_depth(arguments: list[str]) -> int:
    try:
        from depth_diagnostics import main as depth_main

        return int(depth_main(arguments) or 0)
    except ModuleNotFoundError as error:
        from log_utils import get_logger

        get_logger(__name__).exception("Depth diagnostics dependency is unavailable")
        print(
            f"Depth diagnostics could not load {error.name!r}. "
            "Install EdgeMesh with the [depth] extra, then retry.",
            file=sys.stderr,
        )
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    from edgemesh_bootstrap.runtime import initialize_platform_runtime

    initialize_platform_runtime()
    from version import __version__

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "diagnose-depth":
        return _run_depth(arguments[1:])

    parser = argparse.ArgumentParser(
        prog="edgemesh",
        description="Create 3D meshes from images in the EdgeMesh desktop workspace.",
        epilog="For offline model checks: edgemesh diagnose-depth --help",
    )
    parser.add_argument("--version", action="version", version=f"EdgeMesh {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed application output")
    parser.add_argument("project", nargs="?", type=Path, help="Open a project file or project directory")
    args = parser.parse_args(arguments)

    # Help and version stay available without importing the desktop/depth stack.
    from PySide6.QtWidgets import QApplication

    from edge_mesh import MainWindowImageProcessing

    app = QApplication([sys.argv[0]])
    window = MainWindowImageProcessing(
        verbose=args.verbose, restore_last_project=args.project is None
    )
    if args.project is not None:
        window.open_project_path(str(args.project.expanduser().absolute()))
    window.show()
    return app.exec()
