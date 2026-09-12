"""Standard Python packaging, with an explicit optional cx_Freeze command."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

# PEP 517 frontends need not put the source directory on sys.path.
support_spec = spec_from_file_location(
    "_build_support", Path(__file__).resolve().with_name("_build_support.py")
)
if support_spec is None or support_spec.loader is None:
    raise RuntimeError("Cannot locate EdgeMesh's packaging support module.")
support = module_from_spec(support_spec)
sys.modules[support_spec.name] = support
support_spec.loader.exec_module(support)

configuration = {
    "py_modules": support.application_modules(),
    "packages": ["edgemesh_bootstrap", "MeshTools"],
    "cmdclass": {"build_py": support.BuildApplication},
    "package_data": {"MeshTools": ["LICENSE"]},
}

if "build_exe" in sys.argv:
    from cx_Freeze import Executable, setup

    setup(
        **configuration,
        executables=[Executable("edge_mesh.py")],
        options={
            "build_exe": {
                "include_files": [
                    (str(support.ROOT / asset), f"edgemesh_bootstrap/assets/{asset}")
                    for asset in support.ASSETS
                ]
            }
        },
    )
else:
    from setuptools import setup

    setup(**configuration)
