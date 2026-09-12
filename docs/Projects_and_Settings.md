# Projects and settings

EdgeMesh projects are ordinary folders that can be moved, backed up, and opened
from another location. Each folder contains `project.json` and an `assets`
directory. The project file keeps processing settings, the subject mask, history
metadata, and model identity/licensing information in the existing validated
session format. Model metadata remains readable in plain text; compressed mask
data is a compact representation of the full-resolution boolean mask.

Only the selected source image and an explicitly supplied mesh are copied. The
app never imports the entire directory containing an image. Asset paths in the
project are relative, so moving the complete project folder preserves references.
Filenames incorporate SHA-256 checksums and the manifest records sizes and hashes.
On load, missing or changed assets produce an error rather than silently loading
different geometry. These checks detect accidental corruption; anyone able to
edit both a file and its manifest can replace the checksum. Model provenance
verification is a separate responsibility from these project integrity checks.

The default projects location is the `projects` directory inside the current
EdgeMesh user-data directory. A selected projects location changes where new
projects are created and does not move an open project. Creating a project always
allocates a new unique folder. An explicitly selected target folder must not
already exist, which prevents overwriting an unrelated directory.

Saving writes copied assets first and atomically replaces `project.json` only
after validation and serialization succeed. A failed save keeps the previous
manifest and accepted project state. A failed first save can leave its newly
created folder and assets available for recovery; automatic cleanup does not
delete user-owned folders. Existing asset files are retained on subsequent saves.
Keeping a mesh requires the UI to explicitly supply its path; otherwise a save
clears the cached mesh reference. The source and settings remain usable to
regenerate it.

The workspace can explicitly retain the last accepted mesh while settings or the
subject mask change. Its original generation provenance is preserved with that
mesh, including when reopening or autosaving the project. Restoring settings or a
mask does not regenerate geometry: the preview can still show the earlier
accepted result. Use Generate to rebuild geometry from the current inputs. The
retained mesh's provenance continues to identify the model that actually created
that result.

The existing explicit JSON session format remains readable and writable. These
legacy sessions keep their previous external-source behavior. Saving a legacy
session does not silently convert it into a project or copy its image.

The UI settings store is separate from project content. It holds named dialog
directories, window geometry, and explicitly selected UI options. Authentication
credentials belong to provider-specific secure authentication storage; they must
never be passed to either the project or UI settings store. Recognizable
credential field names are rejected on project import and save without logging
their values. This is a guard against accidental configuration copying, not a
secret scanner for arbitrary prose. Window restoration
recovers offscreen positions when monitors have changed. File dialogs retain the
directory the user last visited for that operation.

File and folder pickers use the operating system's native dialog when available,
including the Windows address bar for copying and pasting folder paths. Qt's
file picker remains the fallback on platforms without native dialog support.
In the Windows picker, use Alt+D to edit the folder path or paste a complete
file path into the File name field.

Restarting restores the mask stored in the opened project. Opening an image or
a project with no saved mask does not import a mask from a separate legacy
session. To reuse that mask, open the legacy session or import its exported mask
PNG in Edit mask, then choose Accept mask to save it into the current project.

## Integration contract

`ProjectStore(root)` exposes `create(document, name=None)`,
`save_current(document, mesh_path=None)`, `load(path)`, `set_root(path)`, and
`list_projects()`. Read-only properties expose `current_path`,
`current_document`, `current_metadata`, `current_mesh_path`, and `is_legacy`.
Every returned session document is an independent copy.

For transactional UI loading, call `inspect(path)` first. Restore its `document`
in the UI, and call `accept(snapshot)` only after that restore succeeds. A failed
inspection never switches the current project. The UI owns debounce timing,
reopening the last project, user-facing save status, and logging surfaced errors.

The implementation follows ImageAI's existing folder projects, relative media
references, configurable project roots, and saved last-project choices. It uses
EdgeMesh's validated sessions and atomic write helper instead of importing
ImageAI's runtime or configuration.
