# Python 3.14 Migration — Implementation

**Date:** 2026-08-07 17:12
**Implements:** `docs/python-3.14-feasibility-2026-08-07.md`
**Interpreter:** CPython 3.14.7 (`.venv_linux`, rebuilt from `~/.local/bin/python3.14`)

## What changed

### Environment
- `.venv_linux` recreated on Python 3.14.7 (was 3.12.3). All dependencies
  installed and at the versions from the feasibility matrix: torch 2.13.0,
  numpy 2.5.1, opencv-python 5.0.0.93, PySide6 6.11.1, transformers 5.14.1,
  trimesh 5.0.0, pygame-ce 2.5.7, open3d 0.19.0 (cp314).
- **Open3D pinned locally** — PyPI has no cp314 wheels, so the `main-devel`
  prerelease wheel was downloaded once to
  `~/.cache/edgemesh/wheels/open3d-0.19.0-cp314-cp314-manylinux_2_35_x86_64.whl`
  (355 MB) and installed from that file. The rolling URL is overwritten
  upstream as `main` moves; the pinned copy cannot shift.
  - sha256: `4590992edcf0724adcfd1a60be441711649da25bf7f787482248377ecbf06b5f`

### Code fixes (`mesh_generator.py`)
1. **`MeshGenerator()` default-argument crash** — `options=None` was
   dereferenced immediately; now defaults to `{}`.
2. **Out-of-range triangle indices / heap corruption** — the face loop in
   `mesh_from_shapes` ran to `len(shape) - 1` while each face references
   `i + 2`, so 2-point shapes (what `extrude()` yields) emitted indices one
   past the shape's vertices, causing intermittent `malloc()` aborts inside
   Open3D's OBJ writer. Loop bound corrected to `len(shape) - 2`; vertex/face
   arrays reshaped to `(-1, 3)` so empty inputs stay well-formed.
3. **Silent `.stl` export failure** — `generate()` now calls
   `compute_vertex_normals()` (STL requires normals; the writer returned
   `False` before).

### Dependency / installer changes
- `requirements.txt`: `pygame` → `pygame-ce` (drop-in fork, still
  `import pygame`; upstream pygame has no cp314 wheels).
- `install_open3d.py`: rewritten for 3.13+/3.14 — installs from the pinned
  local wheel (`~/.cache/edgemesh/wheels/` or `EDGEMESH_OPEN3D_WHEEL`), with
  download instructions if none is present. ≤3.12 behavior unchanged
  (PyPI `open3d==0.19.0`).
- `install_requirements.py`: now installs the root `requirements.txt`
  (a superset) instead of only `MeshTools/requirements.txt`, so the submodule
  list can't reinstall plain pygame over pygame-ce.
- `AGENTS.md`: compatibility section corrected — minimum Python is **3.12**
  (PEP 701 f-strings in `mesh_generator.py`), 3.14 supported via the pinned
  Open3D wheel; the old "3.13+ incompatible, use 3.12 or earlier" claim removed.

## Verification (all on 3.14.7, WSL)

15/15 checks passed (`verify_314.py`, scratchpad):

- `MeshGenerator()` constructs without options.
- `mesh_from_shapes` emits zero faces for 2-point shapes and in-range indices
  for 3/4-point and empty inputs.
- End-to-end on `Images/Alien Face.png`: 90 222 verts / 42 916 tris, **all
  triangle indices in range**, normals present.
- `.ply`, `.obj`, and `.stl` exports all succeed (`.stl` returned `False`
  before the normals fix).
- `install_open3d.py` pinned-wheel path exercised for real.
- `pygetwindow`/`hid` were stubbed for the test — both are pre-existing
  Linux-environment failures (Windows-only package / missing system
  `libhidapi`), identical on 3.12.

## Left as-is (known, out of scope)

- `MeshTools/requirements.txt` (submodule) still lists `pygame` — changing it
  means a commit in the MeshTools repo. Root requirements now take precedence
  during install.
- `depth_anything_v2` is imported by `depth_anything.py` but listed in no
  requirements file (pre-existing gap).
- `requirements.txt` remains unpinned (major-version risk noted in the
  feasibility doc).
- The pinned Open3D wheel is an unreleased nightly (rebuilt 2026-08-07);
  re-verify against the sha256 above before ever re-downloading.
