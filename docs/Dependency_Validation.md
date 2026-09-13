# Dependency validation boundaries

## Clean Windows CI validation — 2026-09-13

The CPU profile now contains 95 exact pins. It includes the application roots
`shapely==2.1.2` and `pyvista==0.46.3`, plus PyVista dependencies
`pooch==1.8.2` and `scooby==0.10.2`. These four additions were derived from
the application requirements and installed Windows distribution metadata.
The other 91 pins remain unchanged. `requirements-cpu-test.txt` includes this
profile. All packages were installed in a fresh CPython 3.12.10 environment
with uv's seven-day age cutoff; no system or development packages were changed.

CI now installs EdgeMesh itself with dependency resolution and build isolation
disabled before `pip check`. This makes missing application dependencies visible
to the check. Previously, only third-party metadata was installed, so `pip check`
could pass even when required application imports were absent. A packaging
regression test also checks that the profile covers every applicable application
and depth-extra requirement for Windows Python 3.12.

The import check and each test process invoke the application's Windows runtime
guard before importing scientific libraries. The guard is a no-op on Ubuntu.
A guard in a separate preflight process cannot protect later test discovery.
Faulthandler supplies native exception diagnostics. The EdgeMesh suite stops on
its first failure so a later worker failure cannot hide its unittest traceback.

Local validation in the fresh environment passed: wheel build and installation,
`pip check`, mesh/depth imports, 357 EdgeMesh tests (two opt-in native GUI tests
skipped), and 11 MeshTools tests. The hosted run then exposed a test-only path mismatch: Python used the
Windows short TEMP name `RUNNER~1`, while Qt returned `runneradmin`.
The dialog, project-store, user-state, and workflow UI tests now resolve temporary
roots with cross-platform `Path.resolve()`
so both use the same directory name. This also works on the Ubuntu EC2 reviewer;
no Windows-only path API or machine-specific path is introduced.
A new hosted run remains required after this correction. The final local test
record is in `Notes/CI_Final_Guarded_Tests-2026-09-13.txt`. All 35 project-store,
user-state, and workflow UI tests also passed under a verified Windows 8.3 alias.

## Earlier dependency snapshot

Recorded 2026-09-04 19:12, local Windows time.

The supported regression-test target is Windows x64, CPython 3.12.10, with
PyTorch CPU operations and PySide6's offscreen platform. The current `.venv` reports
PyTorch 2.8.0+cpu with no CUDA runtime. Linux, macOS, Python 3.13/3.14, CUDA,
standalone packaged builds and interactive OpenGL rendering have not been
validated by this dependency snapshot. This does not broaden platform support.

## Selected dependency closure

`requirements-cpu-test.txt` includes all 91 exact versions in
`constraints/windows-py312.txt`. These were traversed from installed distribution
metadata, evaluating Windows/Python environment markers and requested extras.
The roots are NumPy, SciPy, Trimesh, Open3D, PyTorch, torchvision, Transformers,
OpenCV, scikit-learn, matplotlib, PySide6, PyGetWindow and keyboard. No version
conflicts were found between the selected installed packages and their declared
requirements. This is a selected runtime closure, not a blanket environment dump.

These roots support CPU geometry, image and tensor tests, model-adapter contracts
with mocked providers, and offscreen PySide6 tests. The closure includes Open3D's
declared notebook/web dependencies even though the tests do not use them.
Optional SAM/model weights, GPU libraries, hardware controller integrations and
Nuitka/cx_Freeze build tools are not part of this test environment. A future
optional feature that imports another dependency must extend the relevant
profile and validate it independently.

Exact version pins make resolution reproducible on this platform. They are not
an archive of wheel bytes or a hash-locked cross-platform supply-chain manifest.
No fresh environment was installed during this work, so a clean-runner install
is still an explicit CI verification step rather than a claimed local result.
The already installed versions are retained; no packages were upgraded.

The workspace refresh also uses the already installed VTK 9.5.2 directly for
its embedded PySide6 viewport. That exact version is now included in the CPU test
profile; its matplotlib dependency was already present. Renderer initialization
remains lazy. Geometry, conversion and export tests can run without initializing
OpenGL; they do not prove that a native GPU window renders correctly.

## CI scope

`.github/workflows/tests.yml` uses a fresh virtual environment on Windows 2022
and Python 3.12.10. It installs observed packaging versions pip 25.0.1 and
setuptools 80.9.0, then the complete dependency list with dependency resolution
and build isolation disabled. The installed setuptools includes its own
`bdist_wheel` command for small source distributions such as PyGetWindow.
`pip check` catches an incomplete or incompatible closure on the clean runner.
The workflow runs root and MeshTools unittest discovery in separate steps.
Provider tests use mocks; Hugging Face offline settings guard against implicit
downloads. They do not validate model quality, downloaded checkpoint compatibility,
GPU inference, or a real OpenGL window.

There were no existing repository workflows when this file was added. The new
workflow has read-only repository permissions and no persistent checkout
credentials. It uses `pull_request`, never `pull_request_target`. Its immutable
official action revisions were verified from the corresponding upstream release
pages: [checkout v4.2.2](https://github.com/actions/checkout/releases/tag/v4.2.2)
at `11bd71901bbe5b1630ceea73d27597364c9af683` and
[setup-python v5.6.0](https://github.com/actions/setup-python/releases/tag/v5.6.0)
at `a26af69be951a213d495a4c3e4e4022e16d87065`.
The hosted workflow has not been pushed or executed as part of this local work.

## Development Open3D wheel verification

The released Python 3.12 package remains pinned to Open3D 0.19.0. When a local
wheel is explicitly selected, or a development wheel is selected for a newer
interpreter, `install_open3d.py` verifies its bytes before invoking pip.
The expected SHA-256 comes from `EDGEMESH_OPEN3D_SHA256`, or a neighboring
`<wheel filename>.sha256` file. The environment setting takes precedence. A
sidecar accepts a plain 64-character digest or a standard digest/filename line.
Missing, malformed or mismatched digests stop installation with a logged error.

The expected digest must come from a trusted recorded source. Calculating a
digest from an untrusted downloaded file and immediately trusting that value
does not authenticate its origin. The check verifies integrity against the
record, not wheel platform compatibility or upstream authorship. Tests use
temporary byte fixtures and mocked pip calls; no development wheel is installed.

## Mesh quality coverage

`mesh_health.py` accepts an Open3D mesh, a Trimesh mesh or a local mesh path.
Inspection reports vertex/face validity, zero-area and duplicate faces, boundary
and non-manifold edges, closure and winding. Edge counts use finite, valid,
nondegenerate faces; any malformed input still prevents a clean watertight result.
Closure is topological and does not establish outward orientation or printability.

Repair preview returns a copy and removes invalid, zero-area and duplicate faces
and unused vertices. Remaining colors and texture coordinates are preserved.
It does not close holes, weld vertices, flip winding, simplify surfaces, alter
dimensions, save a file or replace the active mesh. The UI must present the
preview and let the user explicitly accept or discard it. Self-intersections,
physical scale, and printer-specific constraints require separate checks.
