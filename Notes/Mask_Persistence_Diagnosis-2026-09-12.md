# Mask persistence diagnosis

Recorded: 2026-09-12 06:37 (America/Chicago).

## Finding

The current AppData project has no accepted mask (`session.mask` is null).
The older `sessions/session.edgemesh.json` contains an intact 2276-by-1488
mask with 1,350,642 foreground pixels. Its referenced source and the current
project's managed source have identical SHA-256 hashes. The saved session is
therefore a valid recovery source for this project. Neither original file was
modified during diagnosis.

The inspected installed uv tool and working checkout have byte-identical mask,
workspace, session, project, and application workflow modules. The installed
application therefore includes the merged processed-image/mask preview changes.

No accepted-mask persistence or merged-preview defect reproduced. The evidence
establishes separate saved session and project states, but cannot establish the
precise earlier UI action that left the project's mask empty. Opening an image
alone does not import a previously saved session's mask.

## Expected behavior

Edit mask opens an editor initialized with the current accepted mask. Accept mask
accepts the editor result and schedules project autosave. Cancel discards editor
changes. The saved project embeds the full-resolution mask; reopening that project
or restarting the app restores it into both the merged preview and the editor.
A legacy saved session remains separately readable through Open session.

An importable lossless PNG was exported to the task's artifact directory. Its
loaded boolean pixels exactly match the older session mask. To recover only the
mask while retaining current project settings, use Edit mask, Import mask PNG, select
that recovered PNG, then Accept mask. The original session and project remain intact.

## Validation and changes

Added `test_accepted_mask_autosaves_and_restores_preview_and_editor_after_restart`
in `tests/test_workspace_ui.py`. It uses the real Qt editor acceptance and project
storage with temporary user data, verifies the debounced autosave on disk, explicitly
reopens the project, recreates the main window, checks exact merged-preview pixels,
and verifies the reopened editor mask. Canceling a changed editor mask also retains
the prior accepted mask on disk.

All 103 focused tests passed in 36.517 seconds across workspace UI, workflow UI,
project workflows/store, sessions, mask editor, subject mask, and mask settings.
The new initial regression test alone passed in 2.136 seconds. These tests use the
installed Windows CPython 3.12 tool dependencies and offscreen Qt with isolated
temporary application data. Model inference and the full application build were
outside this persistence check. No application source change, commit, push, or
installed-package modification was made.
