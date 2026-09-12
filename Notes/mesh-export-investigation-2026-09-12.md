# Mesh export investigation

Checked: 2026-09-12 08:59 (America/Chicago).

The reported DPT/MiDaS exceptions are depth-generation failures logged at 08:21. They do not originate in the exporter. The selected-model snapshot and name mapping are correct. No export-to-generation connection was found by local source review or the independent read-only review.

Both requested Depth Anything V2 exports already exist in the remembered export directory, `C:/Users/aboog/AppData/Local/EdgeMesh/models/`:

- `DepthAnything-demo-image-via-DepthAnythingV2.obj`, written at 08:24:41, 51,180,616 bytes.
- `DepthAnything-demo-image-via-DepthAnythingV2.stl`, written at 08:25:08, 33,360,484 bytes.

Read-only Open3D loading verified 667,208 triangles in each file, matching bounds, and SHA-256 matches against both export sidecars. Each sidecar records Depth Anything V2 Large. Sidecar hash matching is not a new independent signature verification. No model registration or download is necessary to export an accepted mesh.

Added a Qt workflow regression that clicks the actual export button, exercises the save-dialog selection and actual OBJ/STL writers, and reloads output geometry. It covers DepthAnythingV2, DPT and MiDaS selections while forbidding model loading and asserting no generation is started. All six combinations passed. The test uses a synthetic accepted box and automated QFileDialog acceptance; it does not reproduce native Windows dialog input or rerun learned-model inference.

Validation: all 12 workflow UI tests passed; Python compilation and git whitespace checks passed. No production code change or inferred traceback fix was applied. Changes are local and uncommitted.

Separate observation: export suggests `mesh.obj` even when the remembered filter is STL; explicit suffix precedence can therefore keep OBJ selected. This did not affect these two exports, which have explicit matching extensions, and was left outside this investigation.
