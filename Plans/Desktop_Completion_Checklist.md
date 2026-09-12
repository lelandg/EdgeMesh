# Desktop completion and packaging implementation checklist

**Last Updated:** 2026-09-11 09:49
**Status:** Local implementation and packaging verified; Claude review pending approval
**Progress:** 8/9 tasks complete

Build on the existing workspace without discarding prior work. Keep software distribution noncommercial in purpose while preserving the existing 0BSD source license and separately enforcing model notices. No commit, push or public package upload was requested.

- [x] Inspect checkout, user-state and provider capabilities; preserve earlier work.
- [x] Reproduce and correct flat depth output with small images and cached-model comparisons.
- [x] Implement model-specific download consent, visible NC/research-only indicators and signed geometry provenance.
- [x] Implement opt-in Codex, Antigravity and Claude Code authentication and assistant adapters.
- [x] Implement transparent portable projects, autosave/reopen, directory selection and dialog persistence.
- [x] Merge processed image and mask preview; persist workspace settings and add accessible shortcuts.
- [x] Add world-axis viewer navigation and persistent camera/display state.
- [x] Implement wheel/sdist packaging, bundled resources and one-command launch.
- [~] Integrate and verify, run independent security review, build/install smoke once, and write the handoff guide.

Validation: 325 combined tests passed, 107 selected Python files compiled, real cached V1/V2 CPU inference and native Windows VTK checks passed. Packaging fixes and all nine focused packaging tests passed; corrected wheel/sdist archives match the intended source and contents. Clean base/full installations passed dependency and installed CLI checks. Installed base/full GUIs, the SDK probe, and the full-extra uvx help invocation passed outside the checkout. Independent local assistant review has no confirmed unresolved finding. Automatic approval review rejected sending source to Claude without specific export permission; that cross-provider review remains pending. See [completion record](../Notes/Desktop_Completion-2026-09-11.md).

Notes: model provenance is locally tamper-evident, not DRM. Project JSON remains readable. Unknown/imported provenance never claims verified licensing. Auth runtimes own OAuth tokens; explicit keys are excluded from application persistence. No extra TII support was needed for the small offline diagnostic CLI. Shift+Enter uses eligible validated dialog acceptance and generates in the workspace, while remaining a newline in the assistant.
