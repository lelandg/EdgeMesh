# AGENTS.md — EdgeMesh

Canonical instructions for every AI coding agent in this repo. `CLAUDE.md` and
`GEMINI.md` import this file and add only tool-specific notes.

EdgeMesh is a PyQt6 desktop app. It runs edge detection on images, estimates
depth with torch models (MiDaS, DPT, ZoeDepth, Depth-Anything), and builds 3D
meshes from the depth maps. `edge_mesh.py` is the entry point.

## Gotchas

- **Minimum Python is 3.12.** `mesh_generator.py` uses PEP 701 f-strings
  (nested same-type quotes). Python 3.11 and earlier cannot parse the file.
- **Open3D wheels depend on the Python version.** On 3.12, install the released
  `open3d==0.19.0` from PyPI. PyPI wheels stop at cp312. On 3.13/3.14, there
  are no PyPI wheels. `install_open3d.py` downloads the upstream `main-devel`
  prerelease wheel once and installs it from a pinned local file
  (`~/.cache/edgemesh/wheels/`, or the path in `EDGEMESH_OPEN3D_WHEEL`). Do not
  install from the `main-devel` URL directly. Upstream overwrites those URLs in
  place as main moves, so the URL is not reproducible.
- **`pygame` has no cp314 wheels.** The project depends on `pygame-ce`, a
  drop-in fork. The import stays `import pygame`.
- **`MeshTools/` is a separate git repository** nested in this tree. Commit
  changes to MeshTools inside `MeshTools/`, not from the EdgeMesh root.
- **Color order:** images load and process in BGR (OpenCV). Convert to RGB only
  for the Qt preview. Depth maps normalize to 0–255 before mesh generation.
- Module-level flags in `edge_mesh.py` control diagnostics: `debug` (log
  output) and `visualize_images` (OpenCV windows). Both default to `False`.
- Edge clustering (`edge_clustering_analyzer.py`), shape analysis
  (`shape_analyzer.py`), and surface partitioning exist but the GUI does not
  fully expose them.

## Build / test

- `python3 install_requirements.py` installs all dependencies. Run
  `python3 install_open3d.py` after it on Python 3.13+ (see Gotchas).
- Standalone Windows builds: `build.bat` (Nuitka). `build-cx-Freeze.bat` and
  `build-nuitka.bat` are alternates.

## Pointers

- Code map: `Docs/CodeMap.md`.
- Python 3.14 migration record: `Docs/python-3.14-migration-2026-08-07.md`.
- Mesh pipeline walkthrough: `Docs/3D_Mesh_Creation_Flow.md`.
- Global house rules apply (`~/.config/agents/AGENTS.md`).
