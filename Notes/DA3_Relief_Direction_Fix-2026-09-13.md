# DA3 relief direction correction

Date: 2026-09-13.

## Report and reproduction

The user reported strange mesh colors with Depth Anything 3 Small on the demo
image and supplied screenshots through the screenshot folder. The saved project
history identifies the comparison as Depth Pro. Both runs used resolution 700,
depth amount 1.0, anisotropic smoothing, no mask, no background removal,
original-image colors, and the same remaining settings.

The two saved meshes have byte-identical RGBA vertex colors and identical XY
positions. The mismatch was their depth direction. Using the exact original
reproduction regions (top-left 10 by 10 pixels of the mesh grid, and the central
10 by 10 pixels), mean front-surface Z was:

| Result | Distant corner background | Central foreground | Foreground raised |
| --- | ---: | ---: | --- |
| Original DA3 Small | 225.189 | 145.799 | No |
| Original Depth Pro | 13.945 | 221.189 | Yes |
| Corrected DA3 Small | 88.631 | 161.595 | Yes |

A broader four-corner comparison was rejected as an oracle because lower corners
include foreground in this image. The final check uses the same regions that
reproduced the original failure.

## Root cause and correction

DA3 exposes camera distance: larger values mean farther away. EdgeMesh's relief
pipeline expects inverse depth: larger values mean nearer. The initial DA3
adapter normalized distance directly and reversed the intended relief.

The installed DA3 visualization also reciprocates positive depth. The installed
Hugging Face Depth Pro model supplies raw inverse depth on EdgeMesh's existing
path. Its postprocessor converts that output to distance, but EdgeMesh does not
call that postprocessor. Depth Pro must therefore remain unchanged.

A shared normalized_inverse_depth contract validates positive finite distances,
computes min(distance)/distance to avoid overflow for tiny positive values, then
normalizes. Only the DA3 branch uses it, after restoring any horizontal processing
flip. DA3 metadata records normalized_inverse_distance. No color transformation,
image-coordinate mapping, or existing backend behavior changed.

## Validation

- The new exported-mesh regression failed in all eight combinations before the
  fix: four DA3 variants with and without a horizontal processing flip.
- All 81 targeted tests passed after the fix. Coverage includes mesh height and
  RGB positions, invalid/constant/tiny distances, DA3 adapters, existing depth
  providers, model storage, licensing, and UI workflows.
- Real offline DA3 Small inference and mesh generation used the actual project
  settings. The new mesh contains 737800 vertices and 1475596 faces. All vertex
  XY and RGBA values exactly match both original meshes.
- Project, source image, and both original mesh SHA256 values are unchanged.
- Same-camera before/after renders were inspected. The corrected foreground is
  raised and the color placement is preserved.
- Syntax checks and git diff whitespace checks passed. Wheel and source archive
  rebuilt successfully with cached isolated build tools. No release was made.

Evidence: [validation report](DA3_Relief_Direction_Validation-2026-09-13.json),
[test log](DA3_Relief_Direction_Tests-2026-09-13.txt),
[red test log](DA3_Relief_Direction_Red-2026-09-13.txt),
[build log](DA3_Relief_Direction_Build-2026-09-13.txt).

Corrected render: `.da3-validation/relief-direction-fix/corrected_da3.png` under
the checkout root. The `.da3-validation/` directory is gitignored, so the render
exists only on the machine that ran the validation.
Reproduction helper: [validate_da3_relief.py](../scripts/validate_da3_relief.py).

## Use

Restart the DA3 preview to load the changed code. Select Depth Anything 3 Small.
Generate the mesh again. Saved meshes are not modified automatically.

### Prepared local preview

On the validation machine, DA3 Small and the separate CPU runtime are already
prepared under `.da3-validation/`. The launcher
[Launch_DA3_Preview.ps1](Launch_DA3_Preview.ps1) uses separate preview settings
and the verified checkpoint cache. Run it from the checkout root in PowerShell:

```powershell
& '.\Notes\Launch_DA3_Preview.ps1'
```

Choose Depth Anything 3 Small, open an image, and click Depth Mesh. The other
three variants download their own checkpoints when downloads are enabled.
Generic setup and launch commands are on the
[setup page](../docs/Depth_Anything_3.html).

The correction remains in codex/depth-anything-3. No commit, push, merge, or
version change was requested or performed. The original main checkout and its
separate unfinished work remain intact. Real inference was validated for Small
on CPU; the shared conversion for other variants is covered by offline tests.
