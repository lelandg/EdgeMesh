# Selected improvements implementation

Recorded 2026-09-04 19:26, local Windows time.

The ten items in `edgemesh-improvement-selections-v1.json` are implemented in the existing checkout, following the user's approval of the implementation checklist. Mesh-health checks remain optional and default off. The existing review fixes are preserved. No commit or push was made.

## Delivered behavior

- Shared image/depth/mask contracts handle BGR, grayscale, alpha compositing, rectangular sizes, finite depths and binary masks.
- Generation runs through a cooperative Qt worker with captured source/settings, stage status and cancellation. Source/settings edits invalidate pending output. Failure/cancellation retains the last accepted mesh; job staging is cleaned on discard, successful replacement and normal close.
- ModelStore reuses model/processor pairs by immutable identity/device/preprocessing. GUI and direct depth calls default offline. Optional HF downloads pin an actual SHA. Local MiDaS/DPT registration verifies clean source identity and checkpoint SHA256, and avoids the MiDaS backbone's hidden moving-branch download.
- Settings, presets, work and rotating logs use per-user paths. Legacy config is copied once. Error messages appear in an app dock. User-triggered diagnostic ZIPs omit raw logs, config values and images.
- Versioned sessions preserve source references, settings, accepted masks, model metadata and bounded processing history. Missing sources can be relocated. Presets retain only reusable settings. Undo/redo covers settings and accepted masks; geometry is regenerated explicitly.
- Local deterministic parameter suggestions have reasons, bounded values, before/after image previews and Apply/Cancel. SAM2 subject selection supports foreground/background clicks, boxes, overlays and manual brushes; accepted masks are applied in mesh generation. Neither feature uploads images.
- Optional mesh reports expose validity, degeneracy, boundaries, winding and closure. Conservative cleanup compares before/after reports and requires Apply; it can be undone and does not fill holes.
- Windows Python 3.12 CPU dependency closure and a read-only, pinned-action CI workflow were added. Development Open3D wheels require a recorded matching checksum.
- `docs/CodeMap.md` reflects the final source; `docs/Selected_Improvements.html` explains the user workflow. The original selectable report remains available.

## Integration bugs fixed during verification

Fresh startup missing settings defaults; unsupported no-smoothing selection; flat-depth zero failing to produce a plane; settings edits missing history/cancellation; invalid model identity partially restoring a session; model-cache lock waits blocking GUI actions; repair/export dialogs racing newer meshes; stale staged output retention; dialogs retaining image arrays after close; independent loggers racing file rotation; and viewport load failures not reaching the caller.

MeshTools changes make viewport settings per-user and loading transactional: validate the candidate before clearing the display, rebind controls, and restore the previous mesh on failure. MeshTools is a separate repository and must be committed separately if publication is later requested.

## Verification

- **112 EdgeMesh tests passed.** Real offscreen Qt startup, settings debounce/undo, session/preset/mask roundtrips, accepted/rejected suggestions, dialog cleanup, model lifecycle contracts, masks, worker cancellation/failure/staging and actual CPU tensor-to-mesh construction with mocked provider outputs.
- **11 MeshTools tests passed.** Topology, load/rotation rebinding, failed replacement retention, exporter failure propagation and STL roundtrip.
- Ruff passes for new runtime modules and worker integration tests. Changed legacy runtime files pass undefined-name/syntax checks; 16 runtime modules parse. No project typecheck configuration was found.
- Both working trees pass scoped whitespace checks. CodeMap modules/symbols/links and guide local links were verified.
- Command evidence is in `selected-improvements-tests.txt` and `selected-improvements-meshtools-tests.txt` beside this summary.

## Verification boundaries

No model weights were downloaded. Tests validate provider contracts and real local processing, not model prediction quality or checkpoint compatibility. Live depth/SAM2 inference, GPU execution, native OpenGL interaction, standalone builds and fresh hosted CI installation remain unverified. No packages were installed or upgraded. See `docs/Dependency_Validation.md` for the exact supported test profile and limitations.

Sessions reference images and do not embed meshes. Temporary generated output is removed when replaced or when the app closes normally; export work you want to retain. A crash may leave a job directory. Cancellation waits for an active model/download/native operation to return. No forced thread termination is used.
