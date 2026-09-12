### Compatibility changes

- Depth relief now scales with the longest image extent, preserving proportions when output resolution changes. Regenerating an older project with the same depth amount can produce deeper geometry. Existing exported meshes are unchanged; review the depth amount before regenerating.
- The packaged desktop targets CPython 3.12. Optional depth, assistant SDK and SpaceMouse dependencies are installed through separate extras. Experimental newer-Python source installers do not expand the packaged support matrix.
- Legacy sessions remain supported alongside portable projects. Opening a project does not import another session's mask automatically. Restoring settings or a mask requires regeneration to change the mesh.
- Model provenance is verified using the originating profile's local signing key. A copied project can display unverified provenance on another profile without that key.

### Added

- A resizable desktop workspace with proportional image previews, an editable subject mask, detachable settings, history and an embedded 3D viewport. The viewer supports standard views, projection controls and world-axis navigation.
- Portable projects with managed image and mesh assets, atomic saves, autosave, reopening of the last project and explicit save-copy workflows. Window layout, viewer preferences and dialog folders persist between launches.
- Foreground-mask import and editing, settings suggestions, cancellable generation jobs and mesh-health inspection. Mesh cleanup can be previewed, accepted or discarded, and undone.
- Offline model preparation with recorded revisions and checkpoint hashes, model-specific notices, and visible provenance tied to the mesh that was actually generated and exported.
- An optional advisory Assistant panel for Codex, Claude Code and compatible Antigravity runtimes. Sharing validated settings and mesh metadata is explicit; returned advice is not executed or applied automatically. Provider authentication and runtime compatibility remain separate requirements.
- Standard wheel and source packaging, bundled application resources and the `edgemesh` launcher. Help and version commands work without loading the desktop or depth runtimes. A depth-diagnostics command records model and mesh measurements.
- Windows CPU regression coverage for data contracts, geometry, generation lifecycle, project persistence, masks, startup, packaging and assistant protocol boundaries.

### Fixed

- Project open/save/folder actions use native operating-system dialogs where available, while retaining remembered folders, filters, suffixes and cancellation behavior.
- Image/depth processing now consistently validates BGR input, foreground masks, finite depth and proportional output dimensions. Resolution changes no longer flatten relative mesh relief.
- Generation jobs preserve the accepted mesh when work is cancelled, replaced or fails. Mesh export and cleanup retain geometry and color contracts.
- Native Windows startup initializes its platform guard before scientific imports, and OpenCV startup failures receive a logged fallback.
- Package metadata and manifests use the Git-tracked README filename consistently. Packaging tests check that Torch and torchvision pins agree with the CPU dependency constraints.
