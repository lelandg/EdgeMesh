# Depth relief diagnosis

Recorded: 2026-09-11 09:00 (local time).

The mesh pipeline had resolution-dependent relief: increasing image resolution expanded X/Y while leaving Z in a fixed 0-255 range. That defect is corrected. Real cached Depth Anything V1 and V2 runs both produced finite, varied depth and watertight meshes from the bundled neon-city illustration. These measurements establish working inference and consistent geometry proportions for this example; they do not establish photoreal reconstruction quality or explain every flat-looking viewport.

## Geometry correction and compatibility

`depth_to_3d.py` now converts normalized depth into relative front relief:

`z = (depth / 255) * depth_amount * (max(height, width) - 1) / 2`

A depth amount of 1.0 gives a maximum front relief of half the longest image side. X/Y still use pixel coordinates. Increasing resolution therefore scales all three axes together for a corresponding depth field. These are relative mesh units, not metres.

For a full normalized range without smoothing, a depth amount of 1.5 previously yielded depth/longest-side ratios of about 0.479 at 800 pixels and 0.239 at 1600 pixels. It now yields 0.75 at either resolution. Existing saved depth amounts are retained, so regenerating an older project can produce deeper relief. Already exported mesh files are unchanged. Smoothing and foreground selection can affect the measured extent.

The ratios below use a flat back at zero. The alternative mirrored back negates the front Z coordinates and can double the total thickness. A single-image depth map remains a height field: a flat or mirrored back closes the surface without reconstructing unseen sides.

The regression first failed through the real PLY export/reload path at 12 and 24 pixels because the relative thickness differed. It now passes at both resolutions. Rectangular images in both orientations and depth amounts 0, 0.5, and 2 also retain the specified proportions. The unused eager depth-pipeline import was removed from `mesh_generator.py`.

## Actual cached model measurements

Both runs used `Images/example.png` (928 x 1232 BGR), CPU with two torch threads, an explicit 112-pixel image-processor override, a 72 x 96 output mesh, depth amount 1.0, no smoothing, no subject mask, and a flat back. The processor retained aspect ratio and produced a 112 x 154 prediction. Downloads were disabled. The exported PLY files were reloaded with trimesh before geometry was measured.

| Measurement | Depth Anything V1 Large | Depth Anything V2 Large |
| --- | ---: | ---: |
| Raw depth minimum | 44.5873 | 88.7407 |
| Raw depth maximum | 96.0157 | 195.0320 |
| Raw depth standard deviation | 11.5770 | 26.9280 |
| Normalized depth standard deviation | 58.3265 | 64.7905 |
| Normalized depth range | 0-255 | 0-255.000015 |
| Mesh extents X x Y x Z | 95 x 71 x 47.5 | 95 x 71 x 47.500004 |
| Depth / longest X/Y extent | 0.5 | 0.50000004 |
| Vertices | 13,822 | 13,822 |
| Faces | 27,636 | 27,640 |
| Watertight after PLY reload | Yes | Yes |

Raw depth units differ between models. Larger variance is not an accuracy score. These runs used an illustration and a deliberately small processor size; they are a bounded pipeline diagnostic, not a model-quality ranking or a default-resolution benchmark. Depth Pro was not run.

The V1 run used [LiheYoung/depth-anything-large-hf](https://huggingface.co/LiheYoung/depth-anything-large-hf/tree/27ccb0920352c0c37b3a96441873c8d37bd52fb6), revision `27ccb0920352c0c37b3a96441873c8d37bd52fb6`, recorded as Apache 2.0. The V2 run used [depth-anything/Depth-Anything-V2-Large-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf/tree/7581137eff8d4e94f6e796d3baea0e9fa79b22d2), revision `7581137eff8d4e94f6e796d3baea0e9fa79b22d2`, recorded as CC BY-NC 4.0. Full weight SHA-256 values, license-policy snapshots, settings, and geometry evidence are in the reports.

## Saved preprocessing check

The saved settings were: use processed image enabled, project on original enabled, blend 100, edge detection disabled, grayscale disabled, color inversion disabled, sensitivity 148, and line thickness 2.

Calling the real `feature_workflows.preview_image` with these settings preserved every BGR pixel: 0 of 1,143,296 pixels changed, with zero mean and maximum channel difference. Original and prepared input SHA-256 values matched. A second actual DA2 inference on that prepared input produced exactly the same recorded raw prediction statistics and mesh extents as the original-image run. The saved preprocessing route therefore did not explain flattening for the checked illustration and settings.

## Timing and evidence

| Measured stage, seconds | V1 original image | V2 saved preview |
| --- | ---: | ---: |
| Runtime imports and setup | 48.431 | 210.685 |
| Model load | 1.645 | 1.615 |
| Depth/mesh pipeline | 1.408 | 1.187 |
| Total inside diagnostic process | 51.573 | 213.547 |

Setup was unusually slow in this environment. These are recorded run timings, not a controlled hardware comparison. Model loading and inference timings alone omit substantial startup time.

The independent comparison runner seals provenance from the actual reloaded PLY geometry. Both report envelopes were checked against another fresh PLY reload and returned `verified` under their original diagnostic profiles. This is local HMAC verification; another profile without the original key should report the evidence as unverified. The reports contain the public envelopes. Each private `model-state` directory has an ignore file covering its contents.

Evidence files:

- [Combined measurements and verification](../Notes/depth-diagnostics/comparison-summary.json)
- [V1 original-image report](../Notes/depth-diagnostics/da1-small/report.json)
- [V2 saved-preview report](../Notes/depth-diagnostics/da2-saved-preview/report.json)
- [V2 original-image baseline](../Notes/depth-diagnostics/da2-small/report.json)
- [Prepared input image](../Notes/depth-diagnostics/da2-saved-preview/prepared_input.png)

## Diagnostic CLI and checks

The importable `depth_diagnostics.main(argv=None)` and `scripts/depth_diagnostics.py` wrapper use the same offline pipeline. `--model-state` selects an application state directory containing `models/manifest.json`; omission uses the existing application registration store, allowing registered MiDaS/DPT models to be found. `--cache-dir` is the separate Hugging Face cache location. Nonfinite depth amounts and values outside 0-100 are rejected before importing the model runtime. The requested device is passed into the depth pipeline explicitly.

`scripts/compare_cached_depth.py` provides the bounded V1/V2 comparison and optional `--preview-config` evidence used above. It records input-pixel hashes, raw and normalized depth statistics, actual exported mesh measurements, timings, and the provenance envelope.

Validation completed:

- 27 focused depth-relief, depth/edge, data-contract, smoothing, and mesh-generator tests passed. These include real flat-back and mirrored-back PLY export/reload coverage.
- Four additional no-inference CLI tests passed, including five invalid-amount cases, valid 0/100 boundaries, default/explicit model-state forwarding, cache/device forwarding, and failure logging.
- Ruff passed on the new diagnostic/test files; all touched Python files compiled. Whole-file Ruff still reports eight untouched legacy findings in `depth_to_3d.py` and `mesh_generator.py` (unused imports/variables, redundant f-strings, and one-line statements).

GPU inference, default-resolution model quality, quantitative accuracy against ground truth, and the user's unsaved image/view state were outside this diagnostic. The trained-model checks establish that the cached V1/V2 paths run and the corrected mesh relief has stable proportions for the measured illustration.
