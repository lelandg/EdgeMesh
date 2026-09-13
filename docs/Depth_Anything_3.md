# Depth Anything 3

EdgeMesh includes an optional Depth Anything 3 (DA3) adapter for single-image
relief meshes. Select **Depth Anything 3 Small** for the first trial. Base,
Mono-Large, and Metric-Large are also available. These four checkpoints use
Apache 2.0 terms in the [official model catalog](https://github.com/ByteDance-Seed/Depth-Anything-3).

## Setup and use

Open [the setup page](Depth_Anything_3.html) for commands with copy buttons.
The setup installs a separate Python 3.12 runtime in the EdgeMesh per-user data
directory. It uses an existing uv executable. It does not install system packages
or change the GUI environment. Package resolution excludes releases younger than
seven days. The upstream source archive is pinned to commit
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.

1. Install the separate runtime with the setup command.
2. Start EdgeMesh. Open an image and select a DA3 model.
3. Enable model downloads for the first preparation. Click **Depth Mesh**.
4. Review the result. Accept it before exporting the mesh.
5. Disable downloads for subsequent use of the prepared checkpoint.

The upstream runtime requires NumPy below 2. EdgeMesh uses NumPy 2. Separate
processes prevent this dependency conflict. Setup records the installed package
versions in the runtime directory. Setup logs are in `logs/da3-setup.log` under
the selected data directory. Inference errors are recorded in the per-user logs.

To use an existing runtime, set `EDGEMESH_DA3_PYTHON` to its Python executable
before starting EdgeMesh. `EDGEMESH_DATA_DIR` selects the data directory. Each
generation starts a worker and releases its model memory when it finishes. Model
loading on each generation adds latency. Cancel stops the worker and removes
temporary input and output files.

## Output and model identity

The adapter sends RGB pixels to DA3. It resizes proportionally, pads to the
model's patch dimensions, then removes the padding and restores the original
image dimensions. DA3 returns distance, so the adapter converts it to inverse
depth before normalization. Near surfaces then rise above far surfaces in the
relief. Image colors remain attached to their original pixel positions. The
existing pipeline applies the selected relief resolution and depth settings.
The mask remains aligned with the image.

Preparation records an immutable Hugging Face checkpoint revision and SHA-256
hashes for its configuration and weights. Generation uses only that prepared
revision and cannot download additional files. Preparation can download only
when the caller enables downloads. Modified cached files fail verification.

Metric-Large output follows the same normalized relief contract. **Exported
reliefs do not preserve metres or camera calibration.** This integration does
not reconstruct multiple photographs, export camera poses, or generate Gaussian
splats. DA3 does not automatically recover an object's unseen surfaces.

## Other model candidates

| Candidate | Benefit for EdgeMesh | Implementation assessment |
| --- | --- | --- |
| [MoGe-2](https://github.com/microsoft/MoGe) | Metric depth, point maps, valid-pixel masks, camera intrinsics, optional normals | Strongest next candidate. Pin a v2 source revision; current main contains MoGe-3. |
| [Marigold Depth v1.1](https://huggingface.co/prs-eth/marigold-depth-v1-1) | Diffusion-based relative depth provides a different estimator for comparison | Optional Diffusers runtime. Weights have Open RAIL++-M terms. Benchmark latency and memory. |
| [UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth) | Metric depth, camera estimation, and confidence | CC BY-NC 4.0; Windows dependency work is substantial. |
| [Metric3D v2](https://github.com/YvanYin/Metric3D) | Depth plus surface normals | Old PyTorch dependency stack. Evaluate official ONNX support and checkpoint terms first. |
| [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) | Depth consistency across video frames | Best suited to a future video workflow. Small is Apache 2.0; larger variants have noncommercial terms. |

These assessments come from upstream APIs and dependencies. They are not local
performance comparisons. None of these additional model families is implemented
by the DA3 change.
