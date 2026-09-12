# OpenCV startup recovery and preview stylesheet repair

Recorded: 2026-09-11 11:26 (America/Chicago).

Follow-up: the user reported that no window appeared after this repair.
The checks below established successful construction and processing, but did
not keep a native Windows event loop alive long enough to detect delayed
process termination. The subsequent diagnosis and complete startup repair are
recorded in [Startup_Windows_Runtime_Fix-2026-09-11.md](Startup_Windows_Runtime_Fix-2026-09-11.md).

## Failure and evidence

The local-source `uvx` launch reached `_refresh_mask_preview` and failed in
`cv2.cvtColor` with `Unknown C++ exception from OpenCV code`. The same launch
also emitted two Qt stylesheet parsing warnings.

The actual cached environment uses CPython 3.12.11, NumPy 2.2.6,
opencv-python 4.12.0.88 and PySide6 6.9.3. Valid synthetic images reproduced the
native exception after importing EdgeMesh. Five of five fresh EdgeMesh-import
probes failed. `cv2.getNumThreads()` and Canny could fail too, so the problem
was not limited to the mask preview or to a particular image's pixels. A
separate .venv CPython 3.12.10 test process also reproduced the native failure.

OpenCV's Windows build uses the Concurrency backend. An independent native
exception probe captured `STATUS_THREADPOOL_HANDLE_EXCEPTION` with underlying
`STATUS_OBJECT_TYPE_MISMATCH` during scheduler initialization. The underlying
Windows handle failure is not established; a DLL collision was not proven.

Seven of eight recovery probes hit the original exception and then passed both
color conversion and Canny after `cv2.setNumThreads(1)`; the eighth passed
without recovery. OpenCV's [parallel implementation](https://github.com/opencv/opencv/blob/4.12.0/modules/core/src/parallel.cpp)
and [thread configuration documentation](https://docs.opencv.org/4.12.0/db/de0/group__core__utils.html)
explain why one thread bypasses this native backend. Microsoft's
[scheduler documentation](https://learn.microsoft.com/en-us/cpp/parallel/concrt/reference/currentscheduler-class?view=msvc-170#get)
and [status definitions](https://github.com/microsoft/win32metadata/blob/main/generation/WinSDK/RecompiledIdlHeaders/shared/ntstatus.h)
support the native interpretation.

## Repair

- `edgemesh_bootstrap/runtime.py` initializes the OpenCV scheduler before any
  main-window image processing or background work. It retains a working
  configuration; a `cv2.error` triggers a logged single-thread fallback for the
  session. Other exceptions and failed fallback configuration still propagate
  to the application's existing error handling. OpenCV filters may be slower
  when fallback is needed; package versions and Torch thread settings do not change.
- `edge_mesh.py` invokes that helper after user storage is initialized and fixes
  the stylesheet's literal `default_color.name()` by interpolating the color.
- `tests/test_opencv_startup.py` injects the native failure into actual Qt
  startup, checks completed startup and visible previews, checks logged recovery,
  and verifies that a healthy configuration is not overridden.
- `tests/test_preview_stylesheet.py` captures Qt's actual stylesheet diagnostics.
- `docs/CodeMap.md` records the runtime initialization boundary.

## Validation

Both regressions failed before their fixes: startup reached the original
mask-preview traceback, and Qt emitted exactly two stylesheet parse warnings.
Both passed afterward. An independent read-only review found no actionable issues.

All 68 selected startup, workspace, workflow, UI, subject-mask, mask-editor,
image-contract and depth/edge tests passed using the user's cached uv Python
and dependencies with the repaired source. Scoped Ruff checks and Python
compilation passed. No project type-checker configuration exists in pyproject.toml.

The local-source `uvx` package was rebuilt offline with the existing dependency
profile (131 packages). All six fresh installed-package starts passed using a
temporary copy of the real saved configuration and its 2276-by-1488 image.
Two runs retained the normal 32-thread runtime; four hit the native failure,
logged recovery and completed using one OpenCV thread. Every run checked a
ready main window, visible mask preview, all preview modes, Canny output,
color conversion on a worker thread, no application errors, and no Qt
stylesheet parsing warnings. Module-path assertions confirmed that the probes
used the newly installed package rather than source-tree imports.

Rerun the original local-source uvx launch command after closing the failed
window. The normal package cache now contains this repair; project source
cache keys also include the new runtime module.

The existing checkout work, real user settings, source image and projects were
preserved. No commits, pushes, releases, package upgrades or model downloads
were performed. GPU inference, interactive rendering and frozen executable
builds are outside this repair's validation.
