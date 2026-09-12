"""Locate read-only assets in a checkout, installed wheel, or frozen application."""

from pathlib import Path
import sys


def resource_path(name: str) -> Path:
    """Resolve a package-relative asset independently of the working directory."""
    relative = Path(name)
    if relative.anchor or ".." in relative.parts:
        raise ValueError("A resource name must stay inside the EdgeMesh resources.")
    package_root = Path(__file__).resolve().parent
    roots = [package_root / "assets", package_root.parent]
    if getattr(sys, "frozen", False):
        roots.insert(0, Path(sys.executable).resolve().parent / "edgemesh_bootstrap" / "assets")
    for root in roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"EdgeMesh resource is missing: {name}")
