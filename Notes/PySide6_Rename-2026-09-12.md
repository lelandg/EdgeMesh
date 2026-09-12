# PySide6 rename — 2026-09-12

Request: replace every PyQt/Qt reference with PySide6. The app already used
PySide6. This task removed the stale PyQt6 leftovers and renamed the prose.

## Changes

| Area | Before | After |
|------|--------|-------|
| `build.bat`, `build.sh`, `build-nuitka.bat` (ignored) | `--enable-plugin=pyqt6` | `--enable-plugin=pyside6` |
| `edge_mesh.yml` (conda env, UTF-16) | `PyQt6` | `PySide6` |
| `edge_mesh.py` line 780 comment | `PyQt6: Updated Enum` | `PySide6 enum` |
| `README.md` | `Qt/PyQt6` credit, `PySide6-Qt6` badge | `PySide6` credit, `PySide6-6.11.2` badge |
| `Plans/3D_Face_Reconstruction_PRD.md` | `PyQt6` (4 places) | `PySide6` |
| `qt_extensions.py` | module name | `pyside6_extensions.py`; import in `edge_mesh.py` and `docs/CodeMap.md` updated |
| Prose "Qt" in docs, plans, docstrings, comments, error strings | `Qt` | `PySide6` (47 lines across 21 files) |
| `docs/AI_Product_Direction.md` | "EdgeMesh's supplied PyQt6 description is stale" | sentence removed; the description is now correct |

Kept as they are, because the names belong to other software:

- `PySide6.QtCore`, `PySide6.QtWidgets`, `Qt.AlignmentFlag` and similar API names.
- `QT_QPA_PLATFORM` (a Qt runtime variable that CI and the tests set).
- `vtkmodules.qt.QVTKRenderWindowInteractor` (VTK's module path).
- `Doxyfile` (`QT_AUTOBRIEF` is a Doxygen option).
- Historical records: `docs/CHANGELOG.md`, `Notes/`, `.scan-reports/`, `*.bak-*`.
- `PyVistaQt` and `QtInteractor` product names in `docs/AI_Product_Direction.md`.

## Verification

- `ruff check` on the touched Python files: 46 findings, identical to the
  HEAD baseline (E402, E701, F401, F841). None introduced.
- `py_compile` and `import pyside6_extensions`: pass.
- Straggler grep for `pyqt`, `qt_extensions`, and the word `Qt` outside the
  kept categories: none.
- Test log: `Notes/PySide6_Rename_Tests-2026-09-12.txt` (WSL, Python 3.14.7,
  offscreen, offline).

## Test results (WSL)

| Suite | Result |
|-------|--------|
| `tests/` up to `test_parameter_suggestions` | all ok |
| `tests/test_pipeline_jobs` | 4 FAIL, then process abort |
| `tests/test_pr*` | 39 ok |
| `tests/test_[q-z]*` | 81 run, 2 FAIL in `test_workflow_ui` |
| `tests/test_depth_relief` | import error |
| `MeshTools/tests` | import error |

None of the failures touch the renamed code. Each has a Linux-environment
cause that predates this change. The Windows logs from 2026-09-11 show the
same tests passing there.

1. **Pipeline jobs.** The worker thread imports `transformers`. On this WSL
   checkout that import chain takes about 150 seconds (measured twice).
   The test waits 5 seconds, fails, and `tearDown` deletes the window while
   the `QThread` still runs. Qt then aborts the process, which also hides
   the failure summary.
2. **Workflow UI startup tests.** The tests patch `os.path.isfile` to hide
   `Images/example.png`. On Python 3.13+ `pathlib.Path.is_file` calls
   `os.path.isfile`, so the patch also reaches `resource_path`, which raises
   `FileNotFoundError` inside `load_last_used_image`. Python 3.12 (the
   packaged target) does not route through `os.path.isfile`, so the test
   passes there.
3. **Depth relief test and MeshTools suite.** `MeshTools/viewport_3d.py`
   imports `pygetwindow`, which raises `NotImplementedError` on Linux.

## Follow-up

- `docs/CodeMap.md` lists the renamed module. A module rename qualifies for
  a CodeMap refresh per `AGENTS.md`.
- Item 2 above is worth a separate fix: `load_last_used_image` should treat
  a missing example image as a soft condition, or the test should patch
  `resource_path` instead of `os.path.isfile`.
