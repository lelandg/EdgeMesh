# Windows startup crash and PySide6 update

Recorded: 2026-09-11 14:35 (America/Chicago).

## Result and scope

EdgeMesh already used PySide6 6.9.3. The dependency declarations and all four
Windows Qt constraint pins now select PySide6/Qt 6.11.2. The desktop launcher,
depth diagnostics and optional SDK worker also initialize a process-local
workaround for a CPython 3.12 Windows WMI race before dependency imports.

This follows the report that no window appeared after the earlier OpenCV
fallback. That fallback handled a symptom but did not prevent delayed native
process termination. The earlier construction-only validation was insufficient.

## Diagnosis

- The real Windows error record showed the Python process terminating with
  `0xc000070a` in `ntdll.dll`. Native launch probes also captured a background
  `platform._wmi_query` call reached through NumPy/SciPy imports.
- Qt-only windows worked. Forcing one OpenCV thread before imports did not
  prevent termination. Merely changing to PySide6 6.10.3 still failed in two of
  three fresh event-loop runs.
- Blocking WMI before imports made three of three native launches pass with
  the old Qt and three of three with the comparison Qt, including delayed
  worker processing. This isolates the WMI path as the startup trigger on this
  machine.
- [CPython PR 134313](https://github.com/python/cpython/pull/134313) describes
  the matching timeout race: a worker can refer to expired query data and close
  the wrong Windows handle. The correction is in the 3.13/3.14 lines, while the
  inspected 3.12 sources retain the unsafe path. The local findings are
  consistent with that upstream defect; a native memory dump was not used to
  prove the exact corrupted handle.

## Changes

`edgemesh_bootstrap/runtime.py` adds `initialize_platform_runtime`. Only on
CPython 3.12 Windows, it makes the standard library's private WMI-query hook
report unavailability so the existing registry/environment fallback is used.
It works even when `platform` was imported earlier. It changes no Windows
service, system configuration or installed Python file. Because the hook is
private, reassess it when changing the Python support baseline.

The guard runs before scientific imports in `edge_mesh.py`, at the start of
`edgemesh_bootstrap.cli.main` and direct `depth_diagnostics.main`, and before
optional SDK loading in `agent_sdk_worker.load_sdk`. The SDK runs in a separate
process and calls platform-information functions, so the parent's guard alone
would not cover it. The existing logged OpenCV serial fallback remains.

PySide6 6.10.3 reproduced an access violation in a focused test that mocks a
mask dialog's `exec` method. PySide6 6.11.2 passes that test and the entire suite.
The comparison establishes a test-runtime compatibility difference; it does
not establish a corresponding production-dialog defect in 6.10.3. Version
6.11.2 was [published on PyPI](https://pypi.org/project/PySide6/6.11.2/) more than
seven days before this change and was installed with uv's one-week age gate.

`tests/test_platform_startup.py` runs cold subprocesses through four entry
paths with an unsafe-query marker. `tests/test_native_startup.py` adds an
opt-in real Windows launcher test: load an isolated image, keep the window
visible for eight seconds, perform Canny on a worker, and exit cleanly. This
covers the delayed failure missed by construction-only tests.

## Validation

- All 332 tests passed on CPython 3.12.11 and PySide6 6.11.2 in 79.218 seconds,
  including the native Windows event-loop test. See
  [Startup_Windows_Runtime_Tests-2026-09-11.txt](Startup_Windows_Runtime_Tests-2026-09-11.txt).
- The test runner initialized the same platform guard before unittest discovery:
  discovery can import scientific libraries before it imports application entry
  points. Qt 6.11.2 was isolated over the cached uv dependency environment.
- Scoped Ruff checks, Python compilation, and CRLF-aware diff checks passed.
  The project does not configure a type checker.
- Independent read-only review found and closed the direct diagnostics and SDK
  process gaps; its final pass found no actionable issue.

The local-source uvx package was rebuilt offline and installed with the existing
131-package depth/assistants profile. Three of three fresh installed console
launches passed in isolated subprocesses outside the source checkout, using
temporary copies of the real saved configuration. Each loaded the 2276-by-1488
image, stayed visible for ten seconds, checked the mask preview and absence of
application errors, ran Canny on a worker, and exited cleanly. All three
retained OpenCV's normal 32-thread configuration. Module-path assertions proved
that the code came from the new installed wheel. See
[Startup_Windows_Installed_Tests-2026-09-11.txt](Startup_Windows_Installed_Tests-2026-09-11.txt).

Rerun the original local-source uvx command; the new package and dependency
versions are cached. No changes to the launch command are required.
The existing checkout work, real settings, source images and projects are
preserved. No commit, push, release, system-package change, provider request or
model download is part of this repair. GPU inference and frozen executable
builds were not tested.
