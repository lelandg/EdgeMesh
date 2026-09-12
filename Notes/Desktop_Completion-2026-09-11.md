# Desktop completion and packaging

Recorded: 2026-09-11 09:49 (America/Chicago).

The requested desktop features are implemented in the existing working tree. Earlier workspace changes and the separate MeshTools working tree are preserved. No commit, push, release version change, or public package upload was requested or performed.

SAM3 integration, source image/video generation, and full 360-degree reconstruction remain proposed capabilities from the earlier product research. They are not implemented or included in this delivery.

## User workflow

- The workspace places the editable subject mask over the processed image. Preview sizing preserves aspect ratio and the image pipeline preserves BGR until display conversion.
- Portable projects keep a readable JSON manifest with content-addressed source and mesh assets. Changes autosave after a short debounce; projects reopen automatically, and users can change the project directory or save a portable copy. Opening an explicit project takes precedence over the remembered project.
- Window geometry, docks, splitters, camera/display settings, model/editor preferences, and dialog folders persist. Cancelled file dialogs still remember the last visited folder. Settings stores reject credential fields.
- Keyboard access covers project actions, workspace pages, mask editing, and viewer controls. Shift+Enter generates in the workspace and accepts eligible validated dialogs; it remains a newline in the assistant. The viewer provides world-axis translation and rotation with modifier-wheel and keyboard shortcuts.

The [desktop guide](../docs/Desktop_Workflow.html) has copyable shortcuts and links to the focused guides. The [project guide](../docs/Projects_and_Settings.md) and [viewer guide](../docs/Viewport_Controls.md) describe persistence and navigation.

## Models and provenance

Checkpoint-specific policies distinguish permissive, noncommercial, research-only, and unknown terms. Before a supported online model download, the application requires the relevant notice and acknowledgement, bound to the selected revision and policy. Official publisher and license links are included; commercial licensing is described only where supported by a published offer. Offline/local loading does not invent download consent or erase the visible restrictions.

Model selection, Generate actions, and the accepted mesh viewer show textual and color indicators. The viewer label follows the model that actually produced the mesh, independently of the current dropdown selection. Only models used by the generation path enter its provenance.

The project keeps readable provenance. A protected local HMAC key signs the actual geometry, model/revision/weight hashes, and parameters; exports also bind the exact file hash. Edits, absent keys, unknown imported ancestry, and signature failures are shown as unverified. This detects ordinary tampering under the local key's trust boundary; it is not DRM or a globally verifiable license certificate. The application retains its existing 0BSD license, with separate CC0 MeshTools helpers and model terms.

See [model licensing](../docs/Model_Licensing.md) and [model options](../docs/Model_Options.md), including the researched Depth Anything 3 checkpoints. DA3 is not silently downloaded or introduced as a newly validated inference backend.

## Assistants

An opt-in Assistant tab supports Codex, Claude Code, and compatible Antigravity runtimes with provider-owned account authentication or API keys. It sends a typed prompt and, only after explicit opt-in, previewable validated settings and mesh metadata. It does not attach image pixels or project files. Returned text is advice and is never executed or automatically applied.

Codex uses an isolated app-server profile and restricted requests. Claude runs with tools and MCP disabled. Antigravity API-key mode uses the pinned optional SDK with no enabled tools, MCP, hooks, or subagents. The installed older Antigravity CLI lacks the required in-app streaming capabilities and is blocked from receiving prompts; its native sign-in remains accessible. A compatible CLI is required for account-backed in-app requests.

API keys are excluded from application preferences, projects, command arguments, and logs. Windows request processes enter an owned Job Object before launching descendants, including virtual-environment launchers, so cancellation closes the correct process tree. Provider/native-runtime storage behavior is a separate boundary. See [assistant integrations](../docs/Agent_Integrations.md).

Protocol and credential/cancellation tests passed. A real supervised Codex handshake completed without inference, and the actual installed Antigravity SDK configuration probe passed. No live authenticated chat response or paid inference was exercised. Source/wheel execution is the supported delivery path; frozen Windows assistant workers are not supported by this change.

## Flat depth diagnosis

The confirmed defect was resolution-dependent relief: X/Y increased with image dimensions while Z remained in a fixed range. Depth now scales with the longest image side. With depth amount 1.0 and a flat back, the maximum front relief is half that side. Regenerating an older project with the same saved depth amount can consequently produce deeper relief. Existing exported files do not change.

Actual cached Depth Anything V1 Large and V2 Large CPU runs used the bundled neon-city illustration, a 112-pixel processor override, and a 72 by 96 output mesh. Both produced finite, varied depth and watertight PLY meshes after reload. The saved processed-image settings changed zero source pixels, so those settings did not explain flattening in this example.

| Measurement | V1 Large | V2 Large |
| --- | ---: | ---: |
| Raw depth standard deviation | 11.5770 | 26.9280 |
| Reloaded mesh extents | 95 x 71 x 47.5 | 95 x 71 x 47.500004 |
| Depth / longest X/Y extent | 0.5 | 0.50000004 |
| Vertices | 13,822 | 13,822 |
| Watertight | Yes | Yes |
| Total diagnostic process seconds | 51.573 | 213.547 |

These timings include unusually slow runtime setup. Pipeline-only timing was about 1.2 to 1.4 seconds, which must not be presented as total launch-to-result time. The runs verify the measured pipeline, not a model accuracy ranking or complete reconstruction of unseen sides. Depth Pro was not run because its research-only terms do not automatically permit application development merely because the app is noncommercial.

The [diagnosis](../docs/Depth_Diagnosis.md) links revision hashes, weight hashes, preprocessing evidence, timings, and exported geometry. The [comparison summary](depth-diagnostics/comparison-summary.json) records fresh-reload provenance verification. No additional TII integration was needed for the offline diagnostic CLI.

## Validation

- The combined application suite passed all **325 tests in 53.992 seconds**. See [complete output](Desktop_Completion_Tests-2026-09-11.txt). A preceding failure concerned a logging test fixture inheriting an intentional data-directory override; the fixture was corrected and the combined suite passed. Its output remains available in [the first-run record](Desktop_Completion_Tests-2026-09-11-first-run.txt).
- All 107 Python files in the selected application, bootstrap, MeshTools root, and test scope compiled. Scoped Ruff checks passed for the new/changed feature modules, with existing main-module exclusions. Eight untouched legacy lint findings in the depth files are documented; repository-wide lint cleanliness is not claimed.
- Native Windows VTK rendered the actual V2 relief mesh, exercised world-axis navigation, projection, changed mesh bounds, and OBJ/STL round trips with an empty application error dock. The framebuffer had varied rendered colors. The environment could not capture the entire desktop composition, so separate Qt control and native framebuffer captures were inspected.
- Independent local review covered the five assistant/runtime/project integration files and supporting context validation. Confirmed cancellation and SDK session-storage issues were corrected, retested, and reviewed. No confirmed unresolved finding remained in that bounded local review.
- The final wheel and source distribution passed archive inspection. Independent readback confirmed all 62 packaged Python sources match the checkout, including the five assistant integration modules. Both isolated installations passed dependency validation: 87 packages in the base environment and 131 with depth and the SDK. Installed help, version, and diagnostic-help commands passed outside the checkout. Both installed GUI probes passed with packaged resources, a ready desktop, the assistant panel, and empty application error logs. The installed SDK probe reported version 0.1.16 available. The full-extra uvx help invocation also exited successfully. See the [compact packaging evidence](Packaging_Validation-2026-09-11.json).

## Packaging

The package defines a base desktop profile and optional depth, assistants, and SpaceMouse extras. Resources resolve from an installed package without depending on the working directory. The launcher provides help/version without loading Qt or torch and supports opening a project or running the diagnostic CLI.

The initial isolated PEP 517 build exposed a packaging-helper import assumption. Loading that helper by its explicit source path fixed the failure. Archive inspection then caught setuptools adding root tests to the source distribution; the manifest now explicitly excludes them and preserves the exact `ReadMe.md` casing. All **nine focused packaging tests passed**, including the new isolated metadata and manifest regressions. These two follow-up packaging fixes do not change application runtime code; the 325-test combined run preceded them.

Corrected local artifacts, independently checked after build:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| [Wheel](../dist/edgemesh-0.6.7-py3-none-any.whl) | 1,689,272 | `52abf638f0038c1929bd4d939a747882fff5c7b718e79ad626a63f36462e7c86` |
| [Source distribution](../dist/edgemesh-0.6.7.tar.gz) | 1,669,598 | `d11927383f46296306b1b49da0b3bfe23c24c259c971e03173ba9ad8c73334a7` |

The archives include the intended application/bootstrap/MeshTools modules, icon, example image, and licenses. They exclude tests, Git metadata, logs, model weights, caches, credentials, and user projects. No source-distribution rebuild is needed for later changes to this external validation report.

The uvx verification ran outside the checkout with an explicit known Python 3.12 interpreter, task-owned tool/cache directories, offline resolution, no user configuration, and the public PyPI index specified. Automatic Python/tool discovery could not access protected profile folders in this sandbox. An initial two-minute harness limit expired while preparing the native dependency environment; a longer bounded run completed and printed EdgeMesh help with exit code zero. The normal `--python 3.12` launch form is documented separately from this exact tested invocation.

The [copyable launch guide](../docs/Packaging.html) and [packaging details](../docs/Packaging.md) contain the one-command uvx flow and supported profile/platform boundaries. Model weights and provider authentication remain separate from package installation.

## Remaining review and validation limits

The global instructions require an additional Claude review for authentication work. The first restricted network attempt failed; automatic approval review rejected the escalated retry because it would transmit private source files to the external Claude account without specific source-export approval. No Claude source review completed. The application work and independent local review are complete; the requested cross-provider gate remains pending permission or an explicit waiver.

The final bounded review scope is `agent_runtime.py`, `assistant_panel.py`, `agent_sdk_worker.py`, `agent_process_worker.py`, and `project_workflows.py`, with tools disabled. The earlier pending request named four files; the process supervisor was added afterward and needs inclusion in any new approval.

Live authenticated provider responses, GPU depth inference, model accuracy against ground truth, Linux/macOS execution, frozen executable builds, and public distribution have not been verified. These limits do not change the measured Windows source/wheel and cached CPU evidence above.
