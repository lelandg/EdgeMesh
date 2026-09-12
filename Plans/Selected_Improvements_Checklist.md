# Selected EdgeMesh Improvements Implementation Checklist

**Last Updated:** 2026-09-04 19:26
**Status:** Implemented; local validation passed, hardware/clean-runner validation remains
**Progress:** 10/10 selected improvements implemented

## Overview

Implement the ten improvements selected in [edgemesh-improvement-selections-v1.json](../Notes/edgemesh-improvement-selections-v1.json), preserving the existing image-to-depth-to-mesh workflow. The selected file is the scope record; this plan orders work by dependency and records acceptance criteria. Mesh-health checking is optional, as explicitly requested.

The committed starting point is EdgeMesh `29ed31e` (which includes bug fixes in `5cc5ef0`) and MeshTools `51c4adb`. Both working trees were clean when this plan was prepared. Existing branches are `codex/codebase-review-fixes` and `codex/mesh-topology-fixes`. The user approved continuing in this checkout alongside existing work.

## Implementation checklist

- [x] **1. Shared data contracts — high priority (`eng-data-contracts`).** Introduce a small module for validated BGR images, explicit `(height, width)` sizes, finite normalized depth, boolean foreground masks, and mesh coordinate conventions. Use it in depth inference, edge preview and mask application. Define alpha handling without cropping or distorting the source. Test non-square and singleton dimensions, grayscale/BGRA inputs, invalid values and shape mismatches. Complete before job/session/mask integration.
- [x] **2. Portable user state and errors — high priority (`eng-user-state`).** Centralize settings, logs, user presets, work files and support exports under platform-appropriate user paths. Copy legacy config once after successful parsing, preserving the original. Route every displayed error through the log and an in-app error surface. Export diagnostics only on user action, redacting secrets and excluding source images by default. Test read-only install directories, migration retry, malformed config, log rotation and redacted support export.
- [x] **3. Model/processor cache and revisions — high priority (`eng-model-cache`).** Reuse loaded pairs through an injected model store keyed by provider/model, immutable revision, device and preprocessing configuration. Record actual model identity; provide explicit unload and offline/download status. Avoid loading SAM2 until requested. Test reuse, invalidation, cancellation during preparation, offline cache misses and revision records. Resolve available revisions and model license requirements before downloads; do not invent commit hashes.
- [x] **4. Responsive generation with cancellation — high priority (`eng-background-jobs`).** Move depth inference and mesh construction out of GUI handlers into a worker. Snapshot settings and source data for each job, emit stage progress, and marshal results/errors back to the GUI thread. Cancel cooperatively; discard stale results after a new image, cancelled run or closed window. Keep the last accepted mesh usable. Stage files separately and adopt completed output deliberately. Test cancellation before/after each stage, repeated starts, worker failure, changing source and shutdown. A running GPU kernel may finish before cancellation takes effect.
- [x] **5. Optional mesh-health checks — high priority (`eng-mesh-validation`).** Add an opt-in check with an explicit user toggle, initially disabled. Display finite-coordinate, degeneracy, boundary, winding and watertightness results when requested. Allow intentionally open meshes; do not automatically repair or require a health report for ordinary export. Any repair must be previewable, accepted explicitly and undoable. Preserve fundamental exporter input guards. Verify exported OBJ/STL by reopening fixtures. Coordinate MeshTools edits within its own repository.
- [x] **6. Reproducible dependencies and CI — high priority (`eng-reproducibility`).** Define supported Python/platform/device combinations from tested evidence, capture direct/runtime dependency constraints, pin downloaded artifacts with identity verification, and add focused CPU/Qt smoke CI. Keep GPU results separately labeled. Enforce the seven-day package-age policy for any new installs; install no system packages. Use pull_request CI with minimal permissions, never pull_request_target. Test resolution/imports in a fresh supported environment before claiming reproduction.
- [x] **7. Undoable sessions and presets — medium priority (`eng-sessions`).** Store a versioned session with source reference, settings, selected model revision, accepted mask and processing history. Separate reusable presets from image-specific data. Support undo/redo of accepted edits without duplicating large arrays per slider event. Validate files before applying them; relocate missing source files while retaining settings. Test roundtrip, branching after undo, missing images, schema migration and cancelled jobs. Depends on data contracts, user state and job result ownership.
- [x] **8. Local parameter suggestions — medium priority (`ai-parameter-assistant`).** Start with deterministic local image statistics and bounded recommendations for edge thresholds, smoothing, background tolerance and depth scale. Give each suggestion a reason and before/after preview. Apply only after acceptance, with undo through sessions. No images leave the machine. A separate generative model is unnecessary for the first implementation. Test no state mutation before acceptance, bounds, source changes, deterministic suggestions and undo.
- [x] **9. Interactive SAM2 subject masks — medium priority (`ai-subject-mask`).** Add foreground/background clicks or box prompts to a proportionally scaled preview; map them correctly to source coordinates. Run segmentation through the worker/model store, show an overlay, allow manual refinement, and apply an accepted mask consistently to depth and mesh construction. Support undo while keeping the original image intact; color masking remains independently usable. Pin model identity and verify compatibility/license before optional weight downloads. Test coordinate transforms, masks, refinement, cancellation, acceptance and undo without requiring weights in core CI; separately validate real inference when weights/hardware are available.
- [x] **10. Refresh the code map — medium priority (`eng-codemap`).** Update docs/CodeMap.md against the final implemented source, using stable symbol names and actual directory casing. Trace load → preview → job → depth → mask → mesh → optional health → export; document state ownership, cancellation and the separate MeshTools checkout. Verify every referenced module/symbol exists.

## Delivery order and gates

Implement data contracts and portable state first. Then add model reuse and background jobs, followed by optional mesh checks and reproducibility checks. Sessions provide the acceptance/undo history used by suggestions and SAM2. Update the code map as modules settle and finish with the complete workflow walkthrough.

Each feature must include its GUI integration and regression coverage before being marked complete. Use focused lint, project type checks where configured, and relevant tests per task. Reserve the packaged build for the final branch-finishing pass; run local review before any push/PR. Follow the version-manager workflow at publication time. Do not mark model downloads, GPU inference, packaging or deployment verified from mocked tests alone.

## Scope notes

- User note for mesh health: **“Should be optional.”** The design above makes the report opt-in and keeps repair separately consented and undoable.
- Model comparison, depth-disagreement overlays and multi-view reconstruction were not selected.
- The user approved this plan with “looks good”; implementation follows the selected scope.
- No commits, dependency installs, model downloads or external publication are part of preparing this plan.
- The saved priorities are retained even where dependency ordering places one foundational item ahead of another.

## Verification and handoff

- 112 EdgeMesh tests and 11 MeshTools tests pass locally (123 total), including real offscreen QApplication workflows, real mesh construction/export with mocked model output, cancellation, stale results, staging cleanup, sessions and mask acceptance.
- Ruff passes on new runtime modules and worker integration tests. Undefined-name/syntax checks pass on changed legacy runtime files; all 16 touched/new runtime modules parse. No project typecheck configuration was found.
- Both repository diff whitespace checks pass. Code-map links and symbols were verified against source; the HTML guide's local links resolve.
- Remaining external validation: fresh Windows dependency install/hosted CI, actual downloaded depth/SAM2 inference, GPU operation, native OpenGL interaction and standalone packaging. The reproducibility profile is implemented; clean-environment reproduction is not yet claimed.
- No new dependencies installed, weights downloaded, commits created or branches pushed during implementation. MeshTools viewport changes remain in its separate working tree.
- User guide: [Selected_Improvements.html](../docs/Selected_Improvements.html). Evidence and limitations: [implementation summary](../Notes/selected-improvements-implementation-2026-09-04.md).
