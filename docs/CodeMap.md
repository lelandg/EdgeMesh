# EdgeMesh Code Map

Last updated: 2026-09-11 14:32 (America/Chicago).

This map describes the current source, including the selected workflow improvements.
The application uses **PySide6**, OpenCV, PyTorch, Trimesh and Open3D. Python 3.12
is the packaging baseline for native wheels. Paths below are real repository paths; logical groupings
are not additional directories.

## Application and orchestration

| Module | Verified entry points | Responsibility |
|---|---|---|
| [edge_mesh.py](../edge_mesh.py) | `main`, `MainWindowImageProcessing` | Qt application, image controls, preview, viewport creation and explicit export. |
| [workspace_ui.py](../workspace_ui.py) | `WorkspaceMixin`, `ImagePreviewLabel` | Resizable workspace, detachable parameter panel, visible mask, Setup and History integration, layout persistence. |
| [project_workflows.py](../project_workflows.py) | `ProjectWorkflowMixin` | Project menus, visible project location, autosave and project save/open/copy orchestration. |
| [model_compliance_ui.py](../model_compliance_ui.py) | `ModelComplianceMixin`, `ModelLicenseDialog` | Model consent UI, separate selected/accepted model labels and bindings between accepted mesh provenance and export. |
| [history_panel.py](../history_panel.py) | `HistoryPanel` | Metadata-only history browser, setting comparisons and explicit snapshot restore requests. |
| [embedded_viewport.py](../embedded_viewport.py) | `EmbeddedMeshViewport`, `validated_mesh`, `mesh_to_polydata` | Lazy Qt/VTK viewer with independent display buffers and Open3D geometry for health/export. |
| [feature_workflows.py](../feature_workflows.py) | `WorkflowMixin`, `preview_image`, `pixmap` | Session/AI/model/mesh menus, immutable job inputs, cancellation, result acceptance and optional previews. |
| [generation_jobs.py](../generation_jobs.py) | `JobController`, `Cancellation`, `JobCancelled` | One cooperative `QThread` job per controller; result/progress/error/cancellation signals. |
| [data_contracts.py](../data_contracts.py) | `as_bgr`, `output_shape`, `proportional_shape`, `normalized_depth`, `foreground_mask` | Shared image, dimension, depth and foreground-mask validation. |
| [qt_extensions.py](../qt_extensions.py) | `FlowLayout`, `ExpandableLineEdit`, `state_to_bool` | Layout helpers and Qt checkbox-state conversion. |

`MainWindowImageProcessing` inherits `ProjectWorkflowMixin`, `ModelComplianceMixin`,
`WorkspaceMixin` and `WorkflowMixin`. `process_image` starts the
depth route; `generate_mesh` starts the contour route. Both delegate to
`WorkflowMixin._start_generation`. The UI remains responsible for widgets,
status, accepted settings/mask, and installing the resulting mesh in the viewport.

Workspace combines the processed image and accepted-mask display, alongside the
source preview and embedded mesh area. There is no separate Processed page.
The four main pages are Workspace, History, Setup and Assistant. Splitters resize; the
Parameters dock can move, float or close. View restores hidden panels or resets
the layout. Splitter and dock states are saved per user. Setup provides offline
and depth-model starting paths using package metadata without importing PyTorch
or loading weights. Depth and mesh pipelines import inside requested worker
operations; Open3D and VTK rendering are not initialized by the initial window.
History lists bounded
current-session snapshots without decoding every mask, compares their settings,
and advances the history cursor only after restoration succeeds. Saved sessions
preserve the current state and processing metadata; the full in-memory undo
timeline is not persisted across launches. Restoring settings or masks can require
regenerating geometry; portable project assets are described below.

## Image and mesh routes

1. `MainWindowImageProcessing.load_image` decodes an image, validates it with
   `as_bgr`, cancels obsolete work and refreshes proportional previews.
2. `_start_generation` copies the image, mask and validated settings before
   creating a worker. A worker never reads widget values or changes widgets.
3. The depth route constructs `DepthTo3D` with the shared `ModelStore` and calls
   `process_image`. Depth estimation, attenuation, smoothing and mesh generation
   report progress and check cancellation between stages.
4. The contour route calls `MeshGenerator.generate` with visualization disabled.
   `mesh_from_shapes` makes both triangles of each lateral contour wall, closes
   each contour independently and excludes zero-area projected faces. It does
   not generate top/bottom caps.
5. `_generation_succeeded` installs a current result on the UI thread through
   `update_3d_viewport`. A reported viewport failure restores prior state and
   discards the new output. Accepted results update identity/processing history,
   clear repair undo state and release the previous accepted work folder.
   Explicit Export Mesh writes the user's selected file.

| Module | Verified symbols | Role |
|---|---|---|
| [edge_detection.py](../edge_detection.py) | `detect_edges`, `detect_and_project_edges` | Canny edges, thickness and source-image overlays. |
| [depth_to_3d.py](../depth_to_3d.py) | `DepthTo3D.estimate_depth`, `process_image`, `create_3d_mesh`, `create_background_mask` | Model preprocessing/inference, normalized depth, background/subject masking, colored mesh construction and staged output. |
| [smoothing_depth_map_utils.py](../smoothing_depth_map_utils.py) | `SmoothingDepthMapUtils.apply_smoothing`, `anisotropic_diffusion` | Gaussian, bilateral, median and anisotropic depth smoothing. |
| [mesh_generator.py](../mesh_generator.py) | `MeshGenerator.generate`, `mesh_from_shapes` | Contour-based reconstruction without a depth model. |
| [depth_based3d_reconstruction.py](../depth_based3d_reconstruction.py) | `ExtrusionProjectionReconstruction.extrude`, `project` | Lower/upper vertex pairs per contour point. |
| [edge_clustering_analyzer.py](../edge_clustering_analyzer.py) | `EdgeClustering.analyze_edges` | Contours, Hough lines and DBSCAN edge clusters. |
| [shape_analyzer.py](../shape_analyzer.py) | `ShapeAnalysis.extract_geometric_primitives` | Polygon/ellipse analysis used by the contour route. |
| [surface_partitioning.py](../surface_partitioning.py) | `SurfacePartitioning.apply` | Edge-region and convex-hull analysis in the contour route. |
| [depth_cue_estimator_util.py](../depth_cue_estimator_util.py) | `DepthCueEstimator` | Light/shading cues used by contour analysis and `image_processor.py`. |

Images remain contiguous BGR `uint8` arrays internally. `as_bgr` copies input,
converts grayscale, and composites BGRA transparency over white. Conversion to
RGB happens at model/display boundaries. Shared shapes use `(height, width)`;
OpenCV resize calls convert to `(width, height)`. Normalized depth is finite 2D
`float32` in 0–255. Foreground masks are 2D boolean arrays; resizing uses nearest
neighbors. Width-based UI resolution preserves image proportions, with zero
requesting source dimensions. A depth amount of zero produces an explicitly flat
open plane using the foreground mask, rather than extruding a solid back.

`DepthTo3D.create_3d_mesh` scales relief Z using the longest X/Y span:
`normalized_depth * depth_amount * (max(height, width) - 1) / (2 * 255)`.
At depth amount 1 the available front relief is half that span, preserving
relative proportions as output resolution changes. This is relative relief
geometry, not a conversion to physical units.

## State ownership and cancellation

| Owner | State and lifecycle |
|---|---|
| Main window / `WorkflowMixin` | Source image/path, processed preview, settings, accepted `_subject_mask`, `_last_model_info`, mesh references, `_accepted_job_folder` and `_mesh_before_repair`. |
| `JobController` | Worker, thread and cancellation token until `QThread.finished`; controls return to idle only after thread cleanup. |
| Worker closure | Independent input snapshots and one `job-*` output folder under `UserPaths.work_dir`. Unfinished or discarded jobs remove their own staged folder. |
| `_source_generation` / `_job_source_generation` | Reject results from an older source, mask or edited processing settings; cancellation and pending window close also prevent acceptance. |
| `ModelStore` | Shared prepared model/processor pairs, immutable identity manifest and synchronized access. |
| `SessionHistory` | Bounded in-memory settings/mask snapshots with undo/redo; mesh repair has a separate single backup. |
| `ProjectStore` / `ProjectWorkflowMixin` | Portable source/mesh assets, project metadata and visible autosave destination; project writes are separate from image and mesh export commands. |
| `ProvenanceStore` / `ModelComplianceMixin` | Policy/consent records and accepted-mesh provenance. Changing the selected model does not rewrite the accepted result's identity. |
| `AgentRuntime` / `OwnedProcessTree` | Optional provider process, protocol output and cancellation ownership, independent of mesh-generation workers. |

Cancellation is cooperative. `_Worker.run` checks before work and before
publishing a result. `DepthTo3D.process_image` and mesh construction check between
stages; native inference/download operations may finish their current stage.
Cancelled results never become the active mesh. Closing the main window while a
generation job or retained subject-model task runs requests cancellation and
defers closing until workers finish. The accepted work folder is released at
close. `WorkflowMixin._model_work_active` guards restore/unload/register/new work
while either type of worker is active, so the GUI does not wait on a model-store
lock held by a loading worker.
The embedded VTK interactor uses Qt's event loop. Rendering initializes on the
first valid mesh, and `shutdown` finalizes its native window during application
close. A candidate mesh is validated and rendered before it replaces accepted
geometry; viewport buffers are independent from export geometry. The old
`_poll_viewport` compatibility timer is not started by the embedded workflow.

## Models and optional AI assistance

[model_store.py](../model_store.py) owns `ModelStore.get_depth`, `get_sam2`,
`pin_revision`, `register_midas`, `clear` and `identities`. The GUI passes one
shared store and defaults model downloads off. Hugging Face entries resolve to
full commit SHAs recorded in `models/manifest.json`; loading passes the selected
revision, offline policy and `trust_remote_code=False`. Cache keys include model,
revision, device and preprocessing. Local MiDaS/DPT registration records a clean
git source revision plus checkpoint SHA-256 and verifies them before loading.
The exposed GUI depth choices come from `depth_to_3d.model_names`; additional
store capabilities do not automatically create dropdown options.

`DepthTo3D.load_model` delegates to `ModelStore.get_depth` for GUI and direct
Python/CLI callers. Construction without an injected store creates a per-user
store; it does not bypass the offline/revision policy. Constructor cancellation
is passed through to preparation checks. `model_names` exposes MiDaS, DPT, Depth
Anything V2 and Depth Pro. The local MiDaS adapter uses the torchvision ResNeXt
backbone with registered checkpoint weights instead of an upstream nested moving
WSL-Images download; adapter/version details are recorded in model metadata.

[subject_mask.py](../subject_mask.py) contains `SubjectMaskDialog`, `MaskCanvas`,
`infer_subject_mask` and `MiDaSSetupDialog`. Subject selection supports positive
and negative points, a box and manual foreground/background brushes. SAM2 runs
in a retained task thread; prompt revisions reject stale results. A mask reaches
`WorkflowMixin._subject_mask` only after explicit dialog acceptance. Clearing,
accepting or restoring a mask updates source-generation state.

[parameter_suggestions.py](../parameter_suggestions.py) exposes
`suggest_parameters` and immutable `ParameterSuggestion` records. Suggestions use
local image statistics and explain each setting change. The UI shows current and
proposed previews before Apply; generating a new mesh remains a separate action.

### Model policy and accepted result provenance

[model_licensing.py](../model_licensing.py) defines `ModelPolicy`, `policy_for`,
`canonical_model_identity`, `geometry_sha256` and `ProvenanceStore`. It supplies
model-specific policy and consent records plus HMAC tamper evidence. Source and
geometry hashes bind records to the relevant input and accepted result. The
local key is protected with DPAPI on Windows; project JSON remains readable.
Foreign or unavailable keys make a record unverified locally, not forbidden by
DRM. A signature is not permission to use a model or its output.

[model_compliance_ui.py](../model_compliance_ui.py) consumes that policy layer.
`ModelLicenseDialog` presents terms and consent; `ModelComplianceMixin` keeps the
picker's selected model separate from the accepted mesh's model. Project and
export bindings carry the accepted result's provenance. Contour generation has
no learned model and must not inherit a previous depth-model restriction label.
Research-only, noncommercial and custom terms are distinct; Depth Pro's
product-development exclusion is not represented as unrestricted use.

### Optional provider connections

[assistant_panel.py](../assistant_panel.py) defines `AssistantPanel` and
`PromptEdit`. The panel calls `AgentRuntime` for connection status, authentication,
streaming output and cancellation. A context-provider callback is invoked for
an explicit context preview/inclusion request; nonsecret preferences use
`UserPaths` and `atomic_write`. Provider credentials are not project fields.

[agent_runtime.py](../agent_runtime.py) defines `RuntimeConfig`, `AgentRuntime`
and `validate_proposal`, and imports `OwnedProcessTree` from the subprocess
supervisor described below. Its provider edges are:

- Codex app-server JSON-RPC for account login and requests; an OAuth URL or API
  key is handled through the provider protocol with isolated runtime storage.
- Claude native CLI authentication and streamed responses, with an API-key
  environment for that mode.
- Antigravity account-mode CLI streaming and native OAuth sign-in. An older CLI
  can lack the capabilities required for the protected embedded connection.
- [agent_sdk_worker.py](../agent_sdk_worker.py) for Antigravity API-key mode.
  Its `load_sdk`, `build_config` and `main` use the optional Google Antigravity
  SDK/native runtime without requiring the external account-mode CLI.

[agent_process_worker.py](../agent_process_worker.py) defines `OwnedProcessTree`
and `main`. For Windows non-login provider runs, `AgentRuntime` launches this
standard-library-only supervisor with the base Python interpreter and isolated
startup. The supervisor joins a Windows Job Object before starting the selected
provider CLI or SDK worker, establishing descendant ownership before provider
code runs. Provider input/output streams pass through the supervisor.

`OwnedProcessTree` uses a Windows Job Object or a POSIX process group to own
cancellation. `validate_proposal` accepts data-only JSON proposals; receiving a
proposal does not automatically apply settings. Context, file access and host
tool approvals are not enabled by default. These source relationships do not
establish a successful live provider login or inference run.

## Projects, sessions, user storage and diagnostics

[project_store.py](../project_store.py) defines `ProjectStore` and
`ProjectSnapshot`. It layers portable project assets over `SessionDocument`,
imports source and accepted mesh assets into the project, records relative asset
references and hashes, validates asset boundaries, and rejects credential fields.
The project's JSON remains inspectable; weights and provider credentials do not
become portable project assets.

[project_workflows.py](../project_workflows.py) connects `ProjectStore`,
`SessionDocument`, `SettingsStore` and `DialogPersistence` to the main window.
`ProjectWorkflowMixin` owns project new/open/save/copy commands, the visible
project-folder controls and autosave scheduling. Project saving remains distinct
from saving an image or exporting a mesh.

[ui_persistence.py](../ui_persistence.py) defines `SettingsStore`,
`DialogPersistence` and `get_dialog_service`. It validates per-user JSON settings
and remembers purpose-specific dialog locations and eligible UI state. Mask
editor preferences include tools and display choices; masks, prompts, checkpoint
paths and download permission are not implicitly restored as those preferences.

[session_state.py](../session_state.py) defines `SessionDocument`,
`SessionHistory`, `validate_settings`, `save_session`, `load_session`,
`save_preset` and `load_preset`. Legacy versioned sessions contain source path, validated
settings, accepted mask, model metadata and bounded successful-generation history.
They do not embed source images,
weights or generated meshes. Presets contain settings only. UI restoration uses
signal-blocked widget updates and can request a relocated source image.

[user_state.py](../user_state.py) defines `UserPaths.discover`, `atomic_write`,
`migrate_config` and `export_diagnostics`. `EDGEMESH_DATA_DIR` overrides the
per-user root; otherwise Windows uses LocalAppData, macOS Application Support,
and Linux XDG state storage. Config, presets, work folders and logs live below
that root. GUI viewport settings use its `viewport.ini`.
[log_utils.py](../log_utils.py) exposes `setup_logger` and `get_logger` for
rotating per-user logs. The error dock shows UI failures; exported diagnostics
contain metadata and severity counts, excluding raw messages, paths and images.

[depth_diagnostics.py](../depth_diagnostics.py) exposes `main` and `statistics`.
The diagnostic CLI parses arguments before optional imports and defaults to
CPU/offline model loading through `ModelStore`. It follows original BGR input
through the processor, normalized/smoothed depth and a real PLY export, producing
JSON with model identity/license metadata, depth statistics, mesh extents/relief
ratio/closure and load/pipeline timings. An output directory is explicit.
[scripts/depth_diagnostics.py](../scripts/depth_diagnostics.py) is the checkout
wrapper that imports this module's `main`. Comparative trained-model measurements
belong in the run reports; this code map does not infer them from synthetic tests.

## Packaged launcher and resources

[pyproject.toml](../pyproject.toml) maps the `edgemesh` console entry point to
`edgemesh_bootstrap.cli:main` and declares package dependencies/extras.
[edgemesh_bootstrap/cli.py](../edgemesh_bootstrap/cli.py) supplies that parser and
desktop launch entry; [__main__.py](../edgemesh_bootstrap/__main__.py) supplies the
module entry. [resources.py](../edgemesh_bootstrap/resources.py) defines
`resource_path`, used by the GUI to resolve packaged resources.
[runtime.py](../edgemesh_bootstrap/runtime.py) defines `initialize_platform_runtime`,
called before scientific imports by the desktop and diagnostic launchers and
the separate SDK worker. On CPython 3.12 Windows it uses the standard library's
non-WMI platform fallback to avoid delayed native handle corruption from a
timed-out WMI query. It also defines `initialize_opencv_runtime`,
called by the main window before image loading or background work. It probes
OpenCV's native parallel scheduler and logs a single-thread fallback if that
scheduler cannot initialize; healthy runtimes retain their thread settings.
The package and Windows constraints pin PySide6/Qt 6.11.2. Cold subprocess
coverage lives in `tests/test_platform_startup.py`; `tests/test_native_startup.py`
opts into a real Windows event loop with `EDGEMESH_TEST_NATIVE_GUI=1`.
[_build_support.py](../_build_support.py) defines `application_modules` and
`BuildApplication` for the setuptools build. The launcher/resources/build layer
packages the existing app rather than introducing a second GUI implementation.

Use [Packaging.html](Packaging.html) for copyable local launch commands and
[Packaging.md](Packaging.md) for the standard launcher and packaging boundaries.
[Desktop_Workflow.html](Desktop_Workflow.html) describes the user-facing workflow
and keyboard controls. These entry points do not imply PyPI publication.

## Optional mesh health, repair and export

[mesh_health.py](../mesh_health.py) exposes `inspect_mesh` returning `MeshHealth`
and `repair_preview` returning a new mesh. Inputs may be Open3D, Trimesh or a
local mesh path. Reports cover finite geometry, invalid/degenerate/duplicate
faces, boundary/non-manifold edges, closure and winding. They do not certify
self-intersections, physical scale or printability.

`WorkflowMixin.inspect_current_mesh` displays an advisory report. The
`health_action` export check is optional and defaults off. `preview_mesh_repair`
shows before/after counts and requires Apply; `undo_mesh_repair` restores the
single previous mesh. Cleanup removes bad/duplicate/zero-area faces and unused
vertices while preserving remaining appearance. It does not fill holes, weld
vertices, simplify geometry or export automatically.

`MainWindowImageProcessing.export_mesh` chooses OBJ/STL, respects an explicitly
typed extension, and catches propagated writer failures. Binary STL export
computes normals first. Repair and pre-export checks reject active generation and
recheck mesh identity after modal dialogs, so an outdated report or repair cannot
replace a newer mesh delivered during modal events.

## Separate MeshTools repository

`MeshTools/` is a git submodule with its own history and tests. Parent and
submodule changes are separate commits; the parent tracks a submodule revision.
Do not treat edits there as ordinary parent-repository files.

| Module | Verified symbols | Responsibility |
|---|---|---|
| [MeshTools/mesh_tools.py](../MeshTools/mesh_tools.py) | `MeshTools.solidify_mesh_with_flat_back`, `add_mirror_mesh`, `_boundary_edges`, `_stitch_back`, `fix_mesh` | Trimesh operations; boundary-only back stitching, winding, duplicate cleanup. |
| [MeshTools/viewport_3d.py](../MeshTools/viewport_3d.py) | `ThreeDViewport.load_mesh`, `run`, `_export_mesh` | Open3D display, mesh loading, standalone event loop and checked OBJ/STL export. |
| [MeshTools/mesh_manipulation.py](../MeshTools/mesh_manipulation.py) | `MeshManipulation` | Interactive geometry transformations. |
| [MeshTools/measurement_grid_visualizer.py](../MeshTools/measurement_grid_visualizer.py) | `MeasurementGrid` | Depth/percentage measurement overlays. |
| [MeshTools/mesh_gradient_colorizer.py](../MeshTools/mesh_gradient_colorizer.py) | `MeshColorizer` | Depth-based vertex coloring. |

## Validation and dependency boundaries

Root `tests/` covers contracts, depth/edges, generation jobs, actual offscreen Qt
workflows, sessions/user storage, model-store and SAM2 provider contracts,
suggestions, mesh health, contour generation, smoothing, logging and installer
paths/checksums. `MeshTools/tests/test_mesh_topology.py` covers topology, rotation
rebinding and real STL round-trip export. Model-provider tests use doubles;
passing them does not establish checkpoint quality or live GPU compatibility.

[requirements-cpu-test.txt](../requirements-cpu-test.txt),
[constraints/windows-py312.txt](../constraints/windows-py312.txt) and
[.github/workflows/tests.yml](../.github/workflows/tests.yml) define the selected
Windows/Python 3.12 CPU test profile. See
[Dependency_Validation.md](Dependency_Validation.md) for exact coverage and fresh
CI-install limitations. [install_open3d.py](../install_open3d.py) verifies selected
local wheels with `verify_pinned_wheel` before installation.
