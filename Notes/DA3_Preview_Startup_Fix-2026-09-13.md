# DA3 Preview Startup Repair

Date: 2026-09-13 12:01 (America/Chicago).

## Report and reproduction

The preview showed a white window with its title and icon, then closed. The last console message was `State: 2`. That message comes from checkbox restoration and is not an error code.

The original preview launcher reproduced a native abort with exit code 3. Python fault handling located the failure in the Qt event loop. Windows recorded `ucrtbase.dll` with exception code `0xc0000409` for the user's launch.

## Isolation

- Forcing OpenCV to one thread did not prevent the abort.
- Disabling project restoration did not prevent it. That reproduction did not load VTK.
- Importing Qt before scientific libraries did not prevent it.
- A fresh configuration started successfully.
- Probes with copied settings and either saved geometry or saved dock state omitted started successfully.
- A probe that deferred geometry and dock state until construction completed started successfully with the existing project and accepted mesh.
- Native startup fixtures with only config.ini passed before the fix. They did not include the second layout record in ui-settings.json and were insufficient regressions.

## Competing restoration paths

The constructor restores the config.ini window state from load_ui_settings twice, from _finish_workspace, and from _finish_product_workflow. DialogPersistence then restores the main window again during Show from ui-settings.json. Several restores occur before the project toolbar exists. The default reset layout is also captured after earlier saved state has been applied.

Validation probes used the preview worktree. Successful application shutdowns refreshed its saved layout through the normal persistence path. No project or mesh files were deleted.

## Final correction

The main window now owns its layout through config.ini. It opts out of the dialog service's automatic Show-time restoration. Dialogs continue to use DialogPersistence.

Startup captures the original geometry and dock state before loading the image can save intermediate settings. It restores that snapshot once after all docks and toolbars exist. The workspace reset baseline contains the fresh default layout. Invalid saved layout data produces logged warnings.

The first correction exposed an existing layout-preservation test failure because startup saves could replace the pending dock visibility. Capturing the original settings fixed that follow-on issue.

## Validation

- Native regression before correction: exit 3, native abort and access violation. The regression includes the demo image and both competing layout records.
- Native tests after correction: 2 passed, including the real source launcher with the conflicting saved layout.
- Workspace, project, dialog, workflow, OpenCV startup and platform startup tests: 74 passed.
- Live source launcher with the preview's saved project and accepted mesh: PREVIEW_STARTUP_PASS, exit 0 after delayed verification and normal close.
- Read-only review found no actionable regression in layout ownership, dialog persistence or reset behavior.
- Syntax parsing and git diff whitespace checks passed.

The first packaging attempt could not overwrite an older dist artifact because of Windows file permissions. A fresh output directory is used for the corrected package build: dist/startup-fix. No dependency install or upgrade was needed.

## Use

Run Notes/Launch_DA3_Preview.ps1 again. It uses the corrected source in this worktree. The existing project and accepted meshes remain available. The preview is closed after the automated startup check.

No commit, push, merge, release or version change was performed.

Validation completed: 2026-09-13 12:10 (America/Chicago).

Package result: wheel and source archive built successfully in dist/startup-fix. The three corrected product modules in the wheel match the final source.
