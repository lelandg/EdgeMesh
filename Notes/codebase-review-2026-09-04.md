# EdgeMesh codebase review and bug fixes

Reviewed: 2026-09-04 18:31 (local machine time, America/Chicago).

The review covered the Qt/PySide6 image workflow, edge detection, model preprocessing, background masking, smoothing, depth and contour mesh generation, MeshTools solidification/viewport/export, logging and installer paths. Existing issue #1 and recent git history were checked before changing the related mesh code. Its earlier bounds/default-constructor fixes were retained; related contour completeness and write-result defects were fixed in this pass. No issue comments, closures, commits or pushes were made.

## Delivered

- Interactive report: [review-improvements.html](../docs/review-improvements.html).
- 13 selectable future proposals: eight engineering improvements and five AI enhancements, with priority, effort estimates, risks and acceptance criteria.
- Browser draft persistence, JSON download/import, optional save-location picker, and copyable selection summary. Save the JSON into this repository's Notes folder and tell Codex its path, or attach the file. Browser drafts alone are not visible to Codex. Selections do not automatically implement features.

## Verified fixes

| Area | Confirmed defect and resulting behavior |
|---|---|
| Startup | Fresh configuration assigned a list to ConfigParser and aborted initialization. It now creates a valid Settings section. |
| Image load/save | Failed decode could leave stale image/path/config combinations. Missing images crashed Save, and black images were rejected. Decode now succeeds before state changes; valid black images save and failed writes are reported. |
| Preview/edges | Grayscale preview wrote malformed temporary paths next to source files. Grayscale edge detection raised errors and BGR images used RGB luminance. Processing now stays in memory with consistent BGR output. |
| Depth models | DPT referenced an unset input tensor. Hugging Face preprocessing ignored RGB/flip data, recreated processors, and omitted device/evaluation setup. Models/processors are paired at load time, inputs follow the device, flips are undone for color alignment, and output always matches height/width. |
| Background masks | Unsigned tolerance bounds wrapped around black/white, selected RGB did not match BGR pixels, and filling external contours erased enclosed foreground. Signed bounds and border-connected component masks preserve foreground and handle exact matching. |
| Smoothing | Integer diffusion failed, opposite image borders influenced one another, and constant normalization divided by zero. Diffusion uses float arrays with nonwrapping boundaries; constants map to the requested lower bound. |
| Solidification | Back walls were added on internal edges with duplicate faces and inconsistent winding. Only directed boundary edges are stitched; collapsed seams and duplicate faces are removed while preserving colors. |
| Mesh cleanup | Trimesh unique_faces returns a boolean mask; comparing its length could never detect duplicates. Duplicate counts now use the mask's true count. |
| Viewport/export | Rotation held an old or absent mesh; export checked nonexistent .edges, preferred the dialog filter over typed extensions, attempted unsupported ASCII STL and hid write failures. Controls follow each load; exports validate state, compute normals, use binary STL and propagate logged failures. |
| Contour extrusion | One triangle per wall left half-quads and an open contour join. Every contour gets complete side walls, with distinct contours kept separate and collapsed projection triangles omitted. No top/bottom caps are implied. |
| Logging | Log directories were not created and a single global logger mixed names. UI errors now reach rotating per-user files; independent named loggers do not propagate into each other's files. |
| Installation paths | Child installer and requirements paths depended on the current directory. They now resolve from the script's own directory. Tests mocked subprocess; no installer was run. |

## Validation

- EdgeMesh unittest discovery: **33 passed**.
- MeshTools unittest discovery: **10 passed**.
- Total: **43 regression tests passed**. Tests reproduced failures before fixes; additional integration cases cover successful results.
- Real Trimesh/Open3D PLY and STL export/reload checks passed, including watertightness, winding, colors, hole rims, closed inputs and boundary seams.
- Actual Qt window creation passed offscreen with a fresh temporary configuration. This smoke test emitted existing optional-color and missing-example-image diagnostics when run outside the repository.
- Report JavaScript syntax and selection-data validation checks passed. Browser checks covered layout, category filters, recommended selection, draft restoration, JSON import (including priorities/notes), JSON download action, and clearing test choices.
- Changed Python sources parsed/compiled and whitespace checks passed with CRLF recognized as line endings. No configured linter/typechecker was found and none was installed.
- Independent local review rechecked data contracts and found a logger-propagation issue, which was corrected and regression-tested.

## Boundaries and follow-up

Pretrained model downloads, real CUDA inference, interactive GPU rendering, hardware controllers and packaged Windows builds were not exercised. Tensor tests use small stand-in models with real tensor/image operations. Watertightness tests do not guarantee arbitrary self-intersecting or non-manifold input geometry. Edge-only extrusion remains an open lateral surface; cap generation is separate. Legacy unexposed DenseDepth/LeReS integrations were not implemented. The code map is dated 2025-09-17 and its refresh is a selectable proposal. This review does not establish that the repository is free of all bugs.

The five AI options are interactive subject masking, consistent model comparisons, local parameter suggestions with user preview, depth-disagreement overlays (not calibrated confidence), and a bounded multi-view reconstruction experiment. The report links primary [SAM2 documentation](https://huggingface.co/docs/transformers/model_doc/sam2) and [Depth Anything V2 documentation](https://huggingface.co/docs/transformers/model_doc/depth_anything_v2). Hardware performance and licensing require validation before implementation.

## Working-tree state

EdgeMesh branch: `codex/codebase-review-fixes`, created from fetched `origin/main` at `0e5a11f`. MeshTools branch: `codex/mesh-topology-fixes`, created from fetched `origin/main` at `7176eca`. The nested repository contains its own uncommitted changes and tests; publishing a parent commit alone would not include those edits. All changes remain uncommitted as requested by the standing commit policy. Existing `.claude/settings.local.json` edits and the prior Notes report were left in place, following the user's explicit same-checkout choice. That choice was recorded in the standing file-disposition registry.
