# EdgeMesh workspace refresh

Completed: 2026-09-10 16:37 America/Chicago.

## Implemented

The existing working image/depth/mesh pipeline now has a resizable central Workspace, History and Setup. The earlier uncommitted improvements and nested MeshTools checkout were preserved. No commit, push, release or package publication was requested or performed.

- Source and processed previews retain their original pixmaps and refit proportionally when panes resize. The source/mask areas and image/mesh areas have independent splitters. Parameters can move or float; saved layouts restore after all docks exist. Reset Layout recovers hidden controls and redocks detached diagnostics.
- The accepted subject mask is always visible in the workspace with Overlay, Mask and Source modes, adjustable opacity, foreground coverage, edit and clear actions. It applies to Depth Mesh. Contour Mesh continues to use its separate edge-based geometry path.
- The mask dialog adds complete-stroke undo/redo, bounded packed-mask history, mask PNG import/export at original dimensions, display modes and opacity. Manual add/remove brushes work without downloaded weights. SAM2 clicks/boxes remain available; cancellation and stale-result checks preserve accepted work.
- History lists source, model, resolution, depth and mask state, plus a saved-settings comparison. Restoring is explicit and transactional. A retention-limit revision check prevents an evicted row from silently selecting a different replacement state. The current timeline is bounded to 30 states and is not persisted across launches; session files retain the selected state and processing metadata. Restore changes inputs/settings/mask; regenerate to replace geometry.
- Setup offers the example image, source/session opening, local model guidance, download opt-in, model/license information, existing MiDaS/DPT registration and device/runtime information. Optional cloud accounts are not required for local work.
- The preview is a lazy Qt/VTK widget inside the application, with orbit/pan/zoom, standard views, fit, surface/edges/wireframe modes, axes and geometry statistics. It preserves full Open3D geometry for health checks and OBJ/STL export. Invalid input or failed rendering preserves the previous accepted mesh, and interactive errors go to the app and per-user logs. The UI no longer imports the external MeshTools viewport.
- Startup no longer eagerly imports depth/mesh inference modules. Setup reads installed package metadata without loading Torch. VTK is declared explicitly, with the already-installed 9.5.2 version recorded in the CPU validation requirements and constraints; nothing was installed.

## Validation

- Full EdgeMesh suite: **155 tests passed in 7.034 seconds**, using offscreen Qt, offline model settings and temporary user directories. [Full output](workspace-refresh-tests-2026-09-10.txt).
- Focused coverage includes 22 mask tests (10 editor and 12 existing subject-mask tests), 17 embedded-preview tests, 8 history-panel tests and 8 workspace integration tests; these are included in the total, not additional counts.
- Ruff passed for the new/changed workflow modules and their tests. The legacy main window passed focused syntax/undefined-name rules. Python compilation passed. CRLF-aware Git whitespace validation passed; existing line endings were preserved.
- A fresh-process startup probe initialized the real main window with no logged UI errors and without importing `torch`, `open3d`, `vtk` or `vtkmodules`.
- A native Windows Qt/VTK smoke test loaded a diagnostic sphere with 2,522 vertices and 5,040 triangles. The VTK framebuffer contained 21,636 distinct colors and no reported rendering/app errors. [Actual render](../docs/Workspace_3D_Render.png). This validates native rendering of a diagnostic mesh, not downloaded-model inference or GPU acceleration.
- Workspace, History and Setup captures were inspected. Native render content is shown separately because widget captures omit the native child surface and desktop capture returned black in this environment. No composite image was manufactured.
- Final independent integration review found two minor issues, both fixed: history selection after retention eviction and diagnostics recovery after floating the panel. Regression coverage passes for both.

The first full test attempt was 154/155 because the harness set `EDGEMESH_DATA_DIR` while an existing logging test intentionally checked the standard user-directory fallback. The rerun instead isolated the standard user-directory variables and passed without changing application code or that test.

No project-wide static typecheck is configured. No new model weights, paid API calls, authenticated Codex/Antigravity jobs, full 360-degree reconstruction, clean package installation, standalone build or hosted CI run was tested. Native preview tests do not establish broad graphics-driver compatibility. MeshTools files were not changed by this refresh.

## AI and packaging decisions

[Model options](../docs/Model_Options.md) verifies SAM 3/3.1, Depth Anything 3, direct image-to-3D candidates, checkpoint licenses, runtime requirements and metric/multiview data contracts. Keep SAM2 working; evaluate DA3 Small/Base/Mono-Large and optional SAM3 adapters in isolated backend environments.

[AI and product direction](../docs/AI_Product_Direction.md) verifies Astra/image-tool and Gemini image generation, Omni video, supported Codex app-server and Antigravity headless integration, relevant ImageAI interfaces, and independently proposed features inspired by Meshroom, MeshLab, COLMAP, TripoSR, Hunyuan3D and TRELLIS.2.

Prepare a Python package soon, before more heavyweight backends are added. Publish a public alpha after clean wheel installation, launch, sample workflow, session reopen and export tests on explicitly supported platforms. Keep model weights outside the wheel and heavy backends optional. Cloud generation, agent connections and new model backends remain researched proposals.

For a complete one-photo object, compare direct image-to-3D with an Omni turntable experiment. Generated hidden surfaces are plausible inventions; faithful reconstruction needs real overlapping photographs and validated camera/depth geometry.

## Files and handoff

Start with the [visual workspace guide](../docs/Workspace_Refresh.html), which includes an exact Windows launch command and a copy button. The guide links the current screenshots and research reports.

Implementation centers on `workspace_ui.py`, `embedded_viewport.py`, `history_panel.py`, `subject_mask.py` and `session_state.py`, with integration in `edge_mesh.py` and `feature_workflows.py`. The README, CodeMap, requirements/constraints and [implementation checklist](../Plans/Workspace_Refresh_Checklist.md) were updated.

