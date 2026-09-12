# EdgeMesh model options

Verified: 2026-09-10 20:46 UTC

This research compares official model capabilities with EdgeMesh's existing
mask, depth and mesh interfaces. It does not claim local inference benchmarks or
validated installations of the proposed backends. The live application
environment is Python 3.12.10, as verified during the accompanying workspace
inspection. Python 3.14 compatibility remains a separate packaging concern.

## Recommended direction

1. Preserve the working SAM2 provider and make masks easier to see and edit.
2. Evaluate Depth Anything 3 Small/Base and Mono-Large against existing depth
   providers using representative EdgeMesh inputs.
3. Add SAM3 as an optional segmentation provider when text/exemplar selection
   has a clear user benefit.
4. Prototype direct image-to-3D generation separately from the existing relief
   pipeline. Prefer an isolated backend environment for large models and native
   CUDA dependencies.
5. Keep real multiview reconstruction distinct from generated unseen surfaces.

## SAM 3 and SAM 3.1

SAM 3 exists and retains positive/negative click interaction while adding text
and exemplar prompts that find matching objects across images and video.
SAM 3.1, released March 27, 2026, improves joint multi-object video tracking with
shared memory. Its principal new benefit is relevant to a future video workflow.
[Meta capability overview](https://ai.meta.com/research/sam3/),
[official repository](https://github.com/facebookresearch/sam3),
[SAM 3.1 release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).

Official SAM3 setup requires Python >=3.12, PyTorch >=2.7 and a CUDA-compatible
GPU with CUDA >=12.6. Checkpoints require authenticated Hugging Face access.
The model is under a custom SAM License, rather than the Apache-2.0 license of
EdgeMesh's current SAM2 checkpoint. Preserve the exact upstream terms in the
model catalog and download flow.
[Installation](https://github.com/facebookresearch/sam3#installation),
[license](https://github.com/facebookresearch/sam3/blob/main/LICENSE).

This is not an API drop-in replacement for `get_sam2()`. A new provider must
adapt prompts, inference and mask postprocessing while returning the same
original-image-sized boolean mask. Hugging Face also documents
[SAM3 in Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/sam3);
compatibility with the installed Transformers version still needs testing.

Keep SAM2.1 Tiny as the initial lightweight option. Text such as "the vase" or
an exemplar box could become a useful optional convenience; direct brush editing
must remain available independently of any model.

## Depth Anything 3

DA3 supports monocular and consistent multiview depth, with optional camera
parameters. License and output capabilities vary by checkpoint:

| Variant | Relevant output | Upstream weight license |
| --- | --- | --- |
| DA3 Small / Base | Relative depth; camera pose estimation and conditioning | Apache 2.0 |
| DA3 Mono-Large | Dedicated relative monocular depth | Apache 2.0 |
| DA3 Metric-Large | Metric monocular depth and sky segmentation | Apache 2.0 |
| DA3 Large / Giant | Larger any-view models; Giant also supports Gaussian output | CC BY-NC 4.0 |
| DA3 Nested Giant-Large | Combines any-view and metric models; Gaussian output | CC BY-NC 4.0 |

The official catalog also includes 1.1 variants of several large checkpoints.
Do not label the entire model family as unrestricted simply because its code
repository uses Apache 2.0.
[Official model table](https://github.com/ByteDance-Seed/Depth-Anything-3#-model-zoo).

### Runtime and dependency boundaries

The upstream package literally declares `requires-python = ">=3.9, <=3.13"`.
The current EdgeMesh Python 3.12.10 environment satisfies that declaration.
Python 3.14 does not; the literal upper bound may also exclude Python 3.13 patch
releases rather than admitting the whole 3.13 series. Verify actual package
metadata and resolver behavior before selecting a new runtime.

Its dependencies include `numpy<2`, `xformers`, Open3D and PyCOLMAP. An isolated
Python 3.12 backend is recommended to avoid forcing this dependency set into the
working GUI environment. It is a dependency-management recommendation, not a
claim that the live Python version is incompatible.
[Upstream package metadata](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/pyproject.toml).

### Data contract

DA3 returns depth, confidence and camera parameters. Its documented GLB export
contains a colored point cloud and camera visualization, not automatically a
watertight triangle mesh. Gaussian output is a separate representation and is
supported only by selected models. Metric-Large's output must be converted
using focal length; a metric-preserving path must retain raw depth, confidence,
camera intrinsics, extrinsics and scale metadata.
[DA3 API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md),
[metric-depth FAQ](https://github.com/ByteDance-Seed/Depth-Anything-3#-faq).

EdgeMesh currently normalizes each estimated depth image to 0-255. That remains
appropriate for a relative relief workflow, but cannot preserve real-world
distance or consistent scale across independent views.

## Direct image-to-3D candidates

| Candidate | Verified capability | Integration requirements and limits |
| --- | --- | --- |
| SAM 3D Objects | Image plus object mask to shape, texture and object layout; designed for natural images with clutter and occlusion | Official setup: Linux 64-bit, NVIDIA GPU with at least 32 GB VRAM, gated weights and custom SAM License |
| TRELLIS.2 | Image-to-3D with textured meshes, PBR materials and GLB export | Officially tested on Linux; at least 24 GB NVIDIA GPU memory; compiled CUDA dependencies. Code and model MIT; render dependencies have their own terms |
| Hunyuan3D 2.1 | Separate image-to-shape and PBR texture stages | README lists Windows/macOS/Linux support; tested Python 3.10 and PyTorch 2.5.1. Estimates 10 GB for shape, 21 GB for texture, 29 GB combined; custom native render components |

Sources:
[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects),
[SAM 3D setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md),
[SAM 3D license](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE),
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2),
[Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1).

Hunyuan3D 2.1's custom license expressly excludes the EU, UK and South Korea.
That makes it a poor default for a broadly distributed public package without
additional licensing arrangements.
[Hunyuan3D 2.1 license](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE).

SAM 3D Objects and TRELLIS.2 are worthwhile separate backend experiments. The
listed memory figures are upstream requirements, not measurements on this
machine. Do not promise native Windows support for backends whose official
setup is Linux-only.

## Multiview reconstruction and generated orbit video

VGGT is another candidate for observed multiview reconstruction: it estimates
cameras, depths, points and tracks, and exports COLMAP data. Its original
checkpoint is noncommercial; `facebook/VGGT-1B-Commercial` has a separate
commercial-use license and gated approval. The current official repository
also points to VGGT-Omega, which was not evaluated in this bounded research.
[Official VGGT repository](https://github.com/facebookresearch/vggt).

The product should distinguish three workflows:

- **Relief:** one image produces a depth surface. This matches the current app.
- **Reconstruction from photographs:** additional actual views provide additional
  observations of the object's surfaces.
- **Generated full object:** a generative model creates plausible unseen
  surfaces. Those surfaces are inferred, not recovered evidence.

Engineering assessment: generating an orbit video and sampling its frames is
feasible experimentation, but visual temporal consistency does not guarantee
consistent geometry or camera parameters. Generated frames do not reveal the
actual hidden side of the photographed object. A direct image-to-3D provider is
the better first one-photo experiment; genuine multiview photographs should be
the path for faithful reconstruction.

## Grounding in EdgeMesh

- `subject_mask.py` already offers a green foreground overlay, positive and
  negative points, box prompts and manual add/remove brushes. Mask visibility
  in the main workspace and editing ergonomics are the immediate improvements.
- `model_store.py` uses `facebook/sam2.1-hiera-tiny`, records model identity and
  licensing, and requires download opt-in. Extend that pattern for new models.
- `depth_to_3d.py` estimates depth from BGR input; `data_contracts.py` normalizes
  output to 0-255. Preserve that interface for relief and introduce an explicit
  raw/metric contract for multiview work.
- The current model catalog already identifies DA2 Large as CC BY-NC 4.0 and
  Depth Pro as using Apple's model-specific terms. Public packaging should
  expose model-specific availability and licenses rather than implying every
  optional checkpoint has the same terms as EdgeMesh.

No new dependencies were installed, model weights downloaded or inference
benchmarks run for this research.
