# Native project dialogs and recovered subject mask

Recorded: 2026-09-12 07:47 (America/Chicago).

## Findings

The currently selected project is `C:\Users\aboog\AppData\Local\EdgeMesh\projects\project-DepthAnything-demo11-b8371759\project.json`. Its saved mask is null. The legacy `C:\Users\aboog\AppData\Local\EdgeMesh\sessions\session.edgemesh.json` contains a 1488-by-2276 boolean mask with 1,350,642 foreground pixels. Both source image files have identical SHA-256 hashes. Restarting correctly restores an accepted mask in the real application regression test; reopening a maskless project cannot import a separate legacy session's mask.

A separate portable recovery project was created at `C:\Users\aboog\.codex\visualizations\2026\09\12\01a09598-e353-79d2-b127-fb319a5a78c9\Recovered-DepthAnything-demo11\project.json`. It preserves the current project settings and restores the exact legacy mask. The folder also includes `recovered-mask.png` for importing only the mask into an existing project. The original project and legacy session bytes were verified unchanged. No existing application window was closed or restarted.

## Dialog change

`ui_persistence.py` no longer forces Qt widget file dialogs. Open, save, and folder selection permit the native operating-system picker, with Qt's automatic fallback where unavailable. Remembered directories, filters, suffixes, cancellation, and suggested filenames retain their existing behavior. `docs/Projects_and_Settings.md` explains native path entry and separate project/session mask state.

## Validation

- 70 focused tests passed across UI persistence, workspace UI, project storage, and project workflows (21.804 seconds).
- The accepted-mask autosave/reopen/restart regression also passed independently (2.789 seconds).
- Opening the recovered full-resolution project through the actual main window passed: foreground kept 39.9%, mask overlay pixels exactly matched the expected rendering, and the application error log was empty.
- Scoped Ruff and Python compilation passed. This project configures no type checker.
- GUI validation used offscreen Qt and temporary application data. Native Windows dialog appearance and clipboard interaction were not visually tested.
- Source-based uvx refreshes were interrupted after slow recursive cache scanning; diagnostic output also reported two unreadable provenance directories. Built a wheel directly with the existing cached setuptools runtime and verified its dialog module exactly matched the tested source. Installed only EdgeMesh into the existing isolated uv tool environment using offline, no-dependency installation (one package replaced; dependencies unchanged). The wheel required a read grant for the user account because its generated ACL initially allowed only its sandbox owner. The installed distribution now records the local wheel as its source. The older uvx archive environment was not modified.

Existing checkout work was preserved. No commit, push, version change, provider call, or model download was performed.


Final installed verification passed under the user's Windows account: the installed dialog module exactly matches the tested checkout, native dialogs are enabled, and ProjectStore successfully validates the recovered project and its full mask. Verification used isolated offscreen Qt; no existing app window was changed.
