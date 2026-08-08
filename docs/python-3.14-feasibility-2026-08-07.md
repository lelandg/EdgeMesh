# Python 3.14 Feasibility — Smoke Test Findings

**Date:** 2026-08-07 16:25
**Interpreter tested:** CPython 3.14.7 (`~/.local/bin/python3.14`)
**Method:** throwaway venv outside the repo; no repo files were modified.

## Verdict

**EdgeMesh runs on Python 3.14.** Every dependency resolves, all 23 project
modules import, and the edge-detection → mesh → export pipeline produces a real
mesh and writes `.ply` / `.obj`.

The one blocking caveat is **where Open3D comes from**, not whether it works.

## Dependency matrix (3.14 / cp314)

| Package | Version installed | Source |
|---|---|---|
| torch / torchvision | 2.13.0+cpu / 0.28.0+cpu | PyPI cp314 |
| numpy / scipy / scikit-learn | 2.5.1 / 1.18.0 / 1.9.0 | PyPI cp314 |
| shapely / matplotlib / pillow | 2.1.2 / 3.11.1 / 12.3.0 | PyPI cp314 |
| opencv-python | 5.0.0.93 | PyPI `cp37-abi3` |
| PySide6 | 6.11.1 | PyPI `cp310-abi3` (declares `<3.15`) |
| transformers / pyvista / trimesh | 5.14.1 / 0.48.4 / 5.0.0 | pure-Python |
| hidapi | 0.15.0 | PyPI cp314 |
| **pygame** | — | **no cp314; stops at cp313** |
| **open3d** | 0.19.0 | **not on PyPI for cp314** (see below) |

### pygame → pygame-ce

`pygame` 2.6.1 has no cp314 wheel. `pygame-ce` 2.5.7 does, is a drop-in fork
(still `import pygame`), and was verified working here. Affects only
`space_mouse_event_handler.py` and `spacemouse_demo.py`.

### Open3D — the real decision

PyPI's `open3d` tops out at **cp312** (0.19.0, no newer release). A cp314 wheel
exists only on the rolling `main-devel` prerelease tag:

```
https://github.com/isl-org/Open3D/releases/download/main-devel/
  open3d-0.19.0-cp314-cp314-manylinux_2_35_x86_64.whl
```

- 355 MB; the x86_64 build was **rebuilt 2026-08-07 10:11 UTC** and that URL is
  overwritten in place as `main` moves.
- Requires glibc ≥ 2.35 — this WSL box has 2.39. OK.
- Under test it was fully functional: mesh construction, `Vector3d/3i/2iVector`,
  `LineSet`, normals, `estimate_normals`, `.obj`/`.stl`/`.ply` round-trip, and
  an **offscreen GL render** (`create_window(visible=False)`) all worked.

This is the whole risk: the 3D core would sit on an unreleased nightly with no
upstream release to track. If adopted, download the wheel once and install from
that pinned local file so the rolling tag cannot shift underneath the project.

## Smoke test results

10/13 checks passed outright. All three failures are **pre-existing and
environmental**, not 3.14 regressions — each was reproduced identically on
Python 3.12:

| Failure | Cause | Verified control |
|---|---|---|
| `pygetwindow` | Raises `NotImplementedError` on *any* Linux — Windows-only by design (`__init__.py:347`) | Same error on 3.12.3 |
| `hid` | ctypes loader for system `libhidapi`, which is not installed on this box (`ldconfig` shows none) | Same error on 3.12.3 |
| `depth_anything_v2` | Imported by `depth_anything.py:6` but not listed in `requirements.txt` at all | Missing regardless of version |

With `pygetwindow` and `hid` stubbed (they import fine on Windows, where the app
actually ships), **all 23 project modules import on 3.14.7**, including
`edge_mesh`, `viewport_3d`, `mesh_generator`, `depth_to_3d`, and `text_3d`.

## Pre-existing bugs surfaced (independent of Python version)

These were found while testing and are **not** 3.14 issues. Filing them here
because the second one causes memory corruption.

### 1. `MeshGenerator(options=None)` crashes on the default argument

`mesh_generator.py:18-19` — the default is `None`, then `options.get(...)` is
called immediately, so `MeshGenerator()` always raises `AttributeError`.

### 2. `mesh_from_shapes` emits out-of-range triangle indices → heap corruption

`mesh_generator.py:51-53`:

```python
for i in range(0, len(shape) - 1, 2):
    faces.append([num_vertices + i, num_vertices + i + 1, num_vertices + i + 2])
```

`extrude()` yields 2-point shapes, so `range(0, 1, 2)` gives `i = 0` and the
face references `num_vertices + 2` — one vertex past the end of that shape.

Measured on the real pipeline (`Images/Alien Face.png`, 256×256):

```
mesh: 20004 verts, 10002 tris
  tri idx range: 0..20004  (max valid = 20003)
  OUT-OF-RANGE TRIANGLE INDICES: True
```

Open3D's C++ writer reads those indices unchecked. This produced an intermittent
`malloc(): invalid size (unsorted)` abort inside
`open3d::io::WriteTriangleMeshToOBJ` — a real out-of-bounds read, nondeterministic
run to run.

Verified version-independent: the extracted loop yields the identical
out-of-range faces `[[0,1,2],[2,3,4],[4,5,6]]` on both 3.14.7 and 3.12.3.

### 3. `.stl` export silently fails

`o3d.io.write_triangle_mesh(..., '.stl')` returns `False` with
`[Open3D WARNING] Write STL failed: compute normals first.` — `generate()` never
calls `compute_vertex_normals()`. `.ply` and `.obj` succeed.

## Documentation drift

- `AGENTS.md:80-81` and `install_open3d.py:11-49` both assert Open3D is
  incompatible with 3.13+ and advise "use Python 3.12 or earlier". The first
  half is true of *PyPI* Open3D only; the `main-devel` builds cover 3.13 and 3.14.
- "3.12 **or earlier**" is also wrong in the other direction: `mesh_generator.py`
  uses an f-string with nested same-type quotes
  (`f"{now.strftime("%Y%m%d_%H%M%S")}"`), which is PEP 701 syntax and requires
  3.12+. The project cannot run on 3.11.

## Unrelated risk worth noting

`requirements.txt` is fully unpinned, so a rebuild on **any** interpreter now
pulls opencv 5.0 (from 4.x), trimesh 5.0, and transformers 5.x — major-version
bumps with their own API-break surface. The opencv 5.x call surface EdgeMesh
actually uses (44 distinct `cv2.*` functions) was exercised here and passed.

## Reproduction

Scratch artifacts (not in the repo), ~4 GB:

```
<scratchpad>/venv314/       throwaway 3.14.7 venv
<scratchpad>/smoke314.py    13-check dependency + API smoke test
<scratchpad>/stubtest.py    project-module imports with platform shims
<scratchpad>/e2e.py         end-to-end image -> edges -> mesh -> export
<scratchpad>/iso.py         isolates the OBJ export crash
<scratchpad>/bugctl.py      3.12 vs 3.14 control for the index bug
```
