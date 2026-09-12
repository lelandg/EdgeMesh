# Workspace and AI direction implementation checklist

**Last Updated:** 2026-09-10 16:37
**Status:** Complete
**Progress:** 7/7 tasks complete

Improve the working desktop workflow while evaluating newer AI models independently of the existing environment. Preserve the earlier uncommitted improvements and the nested MeshTools repository.

- [x] Inspect current UI, mask/session/job contracts, runtime and ImageAI reference seams.
- [x] Verify SAM 3, Depth Anything 3, optional image/video generation and authenticated agent integration against official sources (`docs/Model_Options.md`, `docs/AI_Product_Direction.md`).
- [x] Create a resizable workspace with detachable controls, visible accepted mask and focused setup guidance.
- [x] Improve mask editing with reversible edits and display controls (`subject_mask.py`); 22 mask tests pass.
- [x] Expose navigable settings/mask history with explicit geometry-regeneration semantics.
- [x] Add an embedded PySide6/VTK preview using existing dependencies and preserve export geometry.
- [x] Run focused and application regression checks; write a cited roadmap and implementation summary.

Model adapters, cloud generation, agent connections and package publication are evaluated proposals in this pass, not enabled services. No credentials, weight downloads, package installs or external publishing were used for the workspace changes. The installed VTK version was recorded in dependency manifests. Existing code is uncommitted; no commit or push was requested.

Validation: 155 tests passed; scoped Ruff and compilation passed; native PySide6/VTK framebuffer smoke passed. See `Notes/workspace-refresh-2026-09-10.md` for evidence and remaining boundaries.
