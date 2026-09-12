"""Setuptools support for the existing flat source tree; no runtime dependencies."""

from pathlib import Path
import shutil

from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parent
ASSETS = ("EdgeMesh.ico", "Images/example.png")
# These are development scripts, diagnostics, or alternate standalone applications.
# Other root Python files are application modules, including optional integrations.
EXCLUDED_MODULES = {
    "_build_support", "setup", "grep", "install_open3d", "install_requirements",
    "pkg_assist", "pt_to_onnx", "py_oxidizer_config_generator", "spacemouse_demo",
    "torch_hub_model_list", "torch_test", "update_tasks",
}


def application_modules() -> list[str]:
    return sorted(path.stem for path in ROOT.glob("*.py") if path.stem not in EXCLUDED_MODULES)


class BuildApplication(build_py):
    """Bundle only the app's declared assets, without user caches or model weights."""

    def run(self):
        super().run()
        for asset in ASSETS:
            source = ROOT / asset
            if not source.is_file():
                raise FileNotFoundError(f"Required application resource is missing: {source}")
            destination = Path(self.build_lib) / "edgemesh_bootstrap" / "assets" / asset
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [
            str(Path(self.build_lib) / "edgemesh_bootstrap" / "assets" / asset) for asset in ASSETS
        ]
