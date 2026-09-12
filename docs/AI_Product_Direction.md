# EdgeMesh AI and product direction

Research checked: 2026-09-10 15:45 America/Chicago (20:45 UTC).

This report combines current primary documentation with inspection of EdgeMesh and the adjacent ImageAI checkout. It proposes a focused product direction; it does not claim that cloud providers, account connections, or the experimental reconstruction paths below have been implemented or validated.

## Recommended order

1. Make the existing local image-to-mesh workflow understandable: editable mask, useful depth preview, integrated 3D inspection, persistent layout, and recoverable project history.
2. Prepare a normal Python package and test a clean installation before adding more heavyweight model dependencies. Keep a simple desktop launch path as well.
3. Add optional image generation/editing through dedicated provider adapters, with a small capability-aware setup panel.
4. Add an optional assistant that can explain settings and propose typed, bounded actions through an installed Codex or Antigravity runtime.
5. Compare a direct image-to-3D backend with a generated turntable reconstruction experiment before choosing a full 360-degree generation pipeline.

## Optional image generation

**Astra can coordinate image generation, but image pixels come from an image model.** GPT-6 Astra's model page lists image input and text output, without native video output. The official Responses API image guide demonstrates Astra calling the image-generation tool. Current image models include GPT Image 2.5 Sunburst for precise editing and Flare for fast generation; ImageAI's inspected OpenAI adapter currently centers on GPT Image 2. Keep the reasoning-model choice separate from the image-model choice. [Astra model](https://developers.openai.com/api/docs/models/gpt-6-astra), [image generation guide](https://developers.openai.com/api/docs/guides/image-generation), [Sunburst](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst), [Flare](https://developers.openai.com/api/docs/models/gpt-image-2.5-flare).

**Gemini is also a supported image provider.** Google's current image guide lists Nano Banana 2 (`gemini-3.1-flash-image`), Nano Banana 2 Lite (`gemini-3.1-flash-lite-image`), and Nano Banana Pro (`gemini-3-pro-image`). Nano Banana 2 is the sensible first comparison candidate for reference-image work; the guide explicitly says Lite is not optimized for multiple reference inputs or sequential editing. These are image-generation choices, distinct from Omni video generation. [Google image generation](https://ai.google.dev/gemini-api/docs/image-generation).

The useful EdgeMesh actions are narrowly focused: create a source object image, remove distracting background content, repair a selected image region, produce a texture variation, and prepare reference views. Preserve the original source and save each accepted output as a history revision. Show the provider, selected model, uploaded inputs, and available usage/cost information before the user starts a paid job. Keep local mask/depth/mesh work usable without any account.

The model registry should describe capabilities such as image editing, masks, reference-image count, output sizes, and availability. Do not treat every provider/model as interchangeable or silently replace a selected unavailable model. Newly listed model IDs still need focused capability tests; this research did not call paid APIs.

## Using the user's Codex or Antigravity authentication

**Codex embedding is officially supported.** The app-server protocol is designed to embed Codex into another product. Its default stdio transport exchanges JSON messages; managed ChatGPT login lets Codex own OAuth, token storage, and refresh. It exposes account state and rate-limit queries. The TypeScript SDK can also start and resume local Codex tasks, but requires Node.js; for a Python desktop application, a supervised stdio app-server process is the more direct integration candidate. [Codex app server](https://learn.chatgpt.com/docs/app-server), [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk).

Recommended design: an opt-in Assistant panel connects to the user's installed runtime, shows connection and usage status, and gives it documented EdgeMesh operations such as inspecting parameters or proposing a new mask/depth/mesh job. Keep approval and cancellation visible. Let the runtime manage login; do not extract tokens from another application's credential files. Codex's experimental external-token mode is intended for hosts that already own that auth lifecycle, which EdgeMesh currently does not. Do not promise that this connection grants direct Image API access or every media tool available in the Codex desktop app; validate those separately.

**Antigravity also has a supported programmatic path.** Its CLI documents headless `-p` operation, JSON/streaming JSON output, and schema-constrained results. It uses cached authentication from an interactive sign-in. The installation/auth documentation describes native Windows support and OS keyring integration. [Headless mode](https://antigravity.google/docs/cli/headless/), [installation and auth](https://antigravity.google/docs/cli/install/).

Use a supervised child process with structured output and scoped permissions for an Antigravity adapter. Its docs warn that a headless run can continue after a tool is denied; therefore success must include the requested artifact/result validation, not merely process exit code zero. Do not enable its blanket permission-bypass mode. Runtime authentication can power that runtime's documented agent capabilities; it should not be presented as a transferable credential for the separate Gemini image/video APIs.

A conversational assistant adds value when it understands the current source, mask, depth map, mesh statistics, and available actions. It is less useful as an unrestricted terminal embedded in the mesh application. Start with explanation and proposed settings, then add explicit app operations with bounded inputs and visible results.

## Gemini Omni and a complete 360-degree object

**Gemini Omni is real and available through the Gemini API.** The current detailed guide uses `gemini-omni-1.1-flash` through the Interactions API and documents image-to-video generation, first/last-frame interpolation, and conversational editing. Older announcements use earlier preview names, so the implementation must follow the current model guide rather than copying an announcement's model ID. Google's guide also notes that Omni may produce multiple shots unless instructed to keep a continuous scene. [Omni API guide](https://ai.google.dev/gemini-api/docs/omni), [Google model catalog](https://ai.google.dev/gemini-api/docs/models).

An object turntable generated from one photograph is a worthwhile experiment. It does not, by itself, establish a geometrically consistent or accurate reconstruction of unseen surfaces. That distinction follows from the task: the back of an object is absent from the source photograph, so a generated back is an inference. A visually smooth video can still change shape, material, lighting, or camera behavior between frames.

Offer two understandable workflows:

- **Reconstruct photographs:** the user supplies overlapping real photographs; the application checks coverage and preserves source/camera evidence.
- **Imagine a complete object:** a generative model supplies missing views or geometry; the result is a creative asset with generated provenance.

For the experimental video route, save the prompt, provider/model, source image, original generated clip, and chosen frames. Reject scene cuts and near-duplicate or visibly inconsistent frames, segment the subject per view, estimate/validate camera geometry, reconstruct and repair, then show reprojection or silhouette comparisons before accepting the result. Do not assume a requested rotation angle is an exact camera pose. Build the experiment around a small known-object comparison set, including an existing 3D asset rendered to a single input image, so the recovered geometry can be compared with known geometry.

Compare this route with **direct image-to-3D** first. TripoSR documents approximately 6 GB VRAM for default single-image inference and MIT coverage for both code and pretrained models. It is a reasonable first optional backend experiment, subject to Windows/Python and compiled dependency tests. TRELLIS.2 is a richer material-aware candidate, but upstream documents Linux testing and an NVIDIA GPU with at least 24 GB VRAM. Neither should become a mandatory dependency of the basic application. [TripoSR](https://github.com/VAST-AI-Research/TripoSR), [TRELLIS.2](https://github.com/microsoft/TRELLIS.2).

## Reusing ImageAI without expanding EdgeMesh into it

The useful references are:

- `../ImageAI/providers/base.py`: provider interface, normalized authentication modes, client reconfiguration, model enumeration, and generation results.
- `../ImageAI/providers/openai.py`: model capability table and separate generation/edit paths.
- `../ImageAI/providers/google.py`: provider/client initialization and model/auth capability checks.
- `../ImageAI/gui/main_window.py` (`_init_settings_tab`): provider selection, masked key inputs, and an explicit Save & Test action.
- `../ImageAI/core/config.py` and `core/security.py`: persistence and keyring abstractions.

Both current applications use PySide6. Reuse focused interfaces and interaction patterns rather than importing ImageAI's large main window. Give EdgeMesh Setup sections for Device, Local Models, Optional Providers, Storage, and Diagnostics. Show concrete readiness states and the next useful action; avoid making first use depend on filling every optional provider field.

There is one pattern to improve while adapting: ImageAI's inspected `ConfigManager.set_api_key` falls back to file storage when keyring storage fails, and its main window also maintains legacy key fields. A new EdgeMesh integration should use the OS credential store or a clearly identified session-only key, with no silent plaintext fallback. Never write keys into project history, prompts, logs, or exported bundles. No credential values were inspected for this report.

## Integrated 3D preview

The inspected baseline used `MeshTools/viewport_3d.py`, which constructs an Open3D `VisualizerWithKeyCallback` and calls `create_window`. The accompanying workspace refresh replaces that external UI dependency with `embedded_viewport.py`, a PySide6 widget using VTK's existing PySide6 interactor. MeshTools remains a separate nested repository. The new viewer preserves Open3D mesh data for health checks and export.

PyVistaQt's `QtInteractor` is another practical candidate: upstream explicitly documents embedding it inside a PySide6 main window. This implementation uses VTK's PySide6 widget directly because VTK is installed and PyVistaQt is not. Existing dependencies do not prove packaged compatibility or rendering performance. [PyVistaQt embedding example](https://qt.pyvista.org/usage.html).

The first viewer should expose orbit/pan/zoom, fit/reset, front/side/top views, perspective/orthographic projection, solid/wireframe/vertex-color display, axes, background selection, and screenshots. Later add selected-defect overlays, clipping, measurements with explicit units, material channels, and turntable output. Keep large-mesh work off the UI thread, preserve camera position across parameter changes when appropriate, and validate color/orientation by reopening exported assets.

## Open-source feature inspiration

These are independently proposed product features inspired by documented workflows; no source code was copied.

| Project | Worthwhile EdgeMesh feature | Scope judgment |
| --- | --- | --- |
| [Meshroom](https://github.com/alicevision/meshroom) | Visible Input → Mask → Depth → Mesh → Repair → Export stages with previews, errors, elapsed time, and reusable valid intermediate results. | Adopt the understandable workflow before considering a full node graph. |
| [MeshLab/PyMeshLab](https://pymeshlab.readthedocs.io/en/latest/filter_list.html) | Click a mesh-health count to highlight the affected geometry; compare a proposed repair with the accepted mesh. | Separate inspection from repair. A PyMeshLab dependency would need its own license review. |
| [COLMAP](https://colmap.github.io/gui.html) | Link selected geometry back to source-image/depth regions where the mapping exists; provide reliable clipping, projection, reset-view, and saved views. | Traceability is useful now; multi-photo reconstruction is a separate backend feature. |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR/blob/main/gradio_app.py) | Preview the exact prepared input, including mask/background and proportional foreground framing; expose a simple detail control. | Make model inputs visible and test export orientation/color against the preview. |
| [Hunyuan3D](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) | Separate geometry generation from texture generation, so texturing can be retried without replacing accepted shape. | Use the workflow idea first. Its [custom license](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE) includes geographic restrictions; it is not an assumed default dependency. |
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Inspect neutral geometry, base color, roughness/metallic/opacity, and environment lighting; expose polygon and texture export targets. | Add material controls only when the asset pipeline carries those channels end to end. |

## Package sooner, publish after installation gates

Start package preparation now. The current root has no `pyproject.toml`; `setup.py` is a short cx_Freeze executable build definition, not evidence that a normal pip installation has been verified. A package boundary will make providers, the UI, resources, and optional model backends easier to test before the feature set grows.

Prepare a minimal core package with a desktop entry point and explicit optional dependency groups. Keep weights outside the wheel and download only selected models through the model manager, recording version, source, license, and cache location. Resolve how the nested MeshTools repository is distributed or replace the UI-facing preview dependency before release. Preserve per-user storage rather than writing into the installed package.

A public alpha should follow a successful wheel build and fresh-environment install, launch, sample-image workflow, saved-project reopen, and export on supported platforms. Test the supported Python/GPU matrix explicitly; a package cannot remove CUDA or native-wheel constraints. A small first-run sample and a single recommended local-model preset will help new users more than adding several provider signup screens. Public publication is a separate action and was not performed here.

## Validation boundary

This pass inspected source and current official documentation. It did not authenticate either agent runtime, call image/video APIs, download or run new models, benchmark generated reconstruction, or validate a distributed package. Vendor model availability, account entitlement, provider terms, model licenses, and platform requirements must be checked again when implementing the corresponding adapter.
