# Depth Anything 3 implementation

Date: 2026-09-13.

## Delivery

The prior implementation contained DA3 research and license records only. This
change adds Small, Base, Mono-Large, and Metric-Large to the model selector,
project settings, license catalog, preparation path, and depth pipeline.

Worktree: D:\Documents\Code\GitHub\EdgeMesh\.worktrees\depth-anything-3.
Branch: codex/depth-anything-3, created from refreshed origin/main.
The original checkout and its unfinished MiDaS/DPT work were preserved.
No commits, push, version bump, or publication were performed.

DA3 runs in a separate Python 3.12 environment because its NumPy requirement
conflicts with the GUI runtime. Setup pins upstream source revision
3d835ec1a5802d64a8b8b15f817a1ab54809bfe4 and excludes packages younger than seven
days. Upstream omitted addict from its metadata; setup adds addict 2.4.0.
The installed runtime is CPU-only. xformers native acceleration is unavailable.

## Evidence

- Real DA3 Small CPU inference passed on Images/example.png. Input and output
  dimensions were 1232 by 928. All depth values were finite. The initial download,
  preparation, and inference sequence completed in 47.7 seconds.
- Checkpoint revision: e08cab65ca0ec38e7826075418411ab90cab4da3. Configuration and
  weight hashes are in Depth_Anything_3_Live_Validation-2026-09-13.json.
- A second inference and mesh generation ran offline through DepthTo3D. OBJ and
  STL exports succeeded. OBJ readback preserved 9600 vertices and 19196 faces.
- The full application suite ran 351 tests, with one skip and one error in the
  existing Assistant descendant-process cleanup test. That test passed alone
  on retest. The suite was not rerun in full.
- MeshTools: 11 tests passed. Final focused DA3, packaging, and UI run: 36 passed.
- Tests used cv2.setNumThreads(1). Default OpenCV threading also failed an
  unchanged data-contract test in this environment. No product workaround was
  added for that separate behavior.
- Syntax parsing passed for 16 edited/new Python files. Git diff whitespace
  validation passed after preserving the original mixed line endings.
- Wheel and source distribution built successfully with isolated build tools.
  Wheel inspection confirmed all DA3 modules and setup assets are included;
  model weights and local runtime files are excluded. These are unreleased
  validation artifacts retaining the existing 1.0.1 version.

## Use and boundaries

Open ../docs/Depth_Anything_3.html. Its prepared-preview command runs
Launch_DA3_Preview.ps1 with separate preview settings and the existing Small
checkpoint cache. Other variants are wired and covered offline, but their
weights and real inference were not tested. CUDA was not tested.

Metric-Large output is normalized for relief. Metres, camera calibration,
multiview reconstruction, and Gaussian splats are outside this change.

MoGe-2 is the recommended next adapter. Marigold Depth v1.1 provides a
complementary estimator. UniDepthV2 and Metric3D v2 need more Windows dependency
work; Video Depth Anything belongs with a future video workflow. See
../docs/Depth_Anything_3.md for official sources and licensing distinctions.
