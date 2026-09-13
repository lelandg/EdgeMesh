# Changelog

## [Unreleased]

## [1.1.0] - 2026-09-13

### Added

- Add Depth Anything 3 Small, Base, Mono-Large and Metric-Large for single-image relief meshes, with an isolated optional runtime, cancellable workers, verified checkpoint identities and offline inference.
- Add model licensing, setup guidance and validation coverage for the DA3 workflow.

### Fixed

- Convert DA3 camera distance to inverse depth so nearby surfaces rise correctly while source colors stay aligned.
- Restore the main-window layout once after construction to prevent native startup crashes from competing saved layouts and retain saved dock visibility.

## [1.0.1] - 2026-09-12

### Changed
- Every PyQt6 and Qt reference now says PySide6. The Nuitka build scripts enable the `pyside6` plugin, the conda environment file pins PySide6, and `qt_extensions.py` is renamed to `pyside6_extensions.py`. Docs, plans, docstrings, comments, and error strings use the same term.
- The agent guidance in `AGENTS.md`, `CLAUDE.md`, and `GEMINI.md` now covers project structure, environments, testing, documentation standards, review flow, and release expectations. `docs/CodeMap.md` matches the current module layout.
- Reference screenshots for the Assistant panel, Depth Anything V2 mesh generation, masking, and a PrusaSlicer import of an EdgeMesh export live in `Screenshots/`.

### Fixed
- A workflow test now exercises mesh export without loading a depth model, so the export path stays covered in offline CI.

## [1.0.0] - 2026-09-12

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

## [0.6.7] - 2026-08-07

### Changed
- Unified agent instructions: `AGENTS.md` is now the canonical guide for all AI coding CLIs, with `CLAUDE.md` and `GEMINI.md` as thin `@import` pointers.
- Added the even-numbered Python minor version rule to `AGENTS.md`, and a WSL runbook (`docs/python-3.14-wsl-runbook-2026-08-07.html`) for installing Python 3.14 alongside 3.12 — EdgeMesh stays on 3.12 until Open3D ships wheels past cp312.
- Adopted version-manager conventions: `version.py` is the canonical version location (`_version.py` is now a re-export shim), and `docs/Versions.md` was renamed to `docs/CHANGELOG.md`.
- Normalized line endings across the tree; `.venv_linux/` is now git-ignored.
- Updated MeshTools submodule and relaxed the open3d version pin.
- Added project documentation, license, and architecture map.

### Added
- `Plans/3D_Face_Reconstruction_PRD.md` — product requirements for 3D face reconstruction.

Version history (entries below 0.6.7 use the legacy format):
* 0.6.6:
  * Disable "TF_ENABLE_ONEDNN_OPTS" to get precise results from tensorflow.
  * Update main in `edge_mesh.py` to print version, author, and show '-h for help'. Not much there right now, just 'verbose'.
* 0.6.5:
  * Fixed image resizing issue. Images are no longer made square before creating a 3D mesh.
* 0.6.4:
  * Remove depth values < 0 seems to help. 
* 0.6.3:
  * Fixed the "square image" issue. 
* 0.6.2:
  * Fixed: background removal is only done from the edges. Even when you use a large tolerance! Finally! :) 
* 0.6.1:
  * Major upgrade to PyQt6 for UI elements. This is mostly so we can build with nuitka.
  * Updated support of image types to everything OpenCV handles, which are:
    .BMP .DIB .JPEG .JPG .JPE .JP2 .PNG .WEBP .PBM .PGM .PPM .PXM .PNM .SR .RAS .TIFF .TIF .EXR .HDR .PIC ... 
    Cool. :-)
  * Tweaks for background removal. Still not perfect. Most things work great if it's a solid background, but details close to that color can cause jaggies along the edges. 
* 0.5.18:
  * I just heard about "Depth Pro" during lunch. I got it implemented before 13:00. (I'll have to play with it.)
  * Combo box settings are now saved and restored. (Depth and smoothing methods.)
* 0.5.17:
  * Added SpaceMouse support. It's slow, but that's because it removes and re-adds the mesh each time. But it works. :-) 
* 0.5.16:
  * Restored functionality to Generate Mesh button. (It was broken in 0.5.15 and earlier.)
  * Added automatic saving of this file. Right now it's very flat. (I've had it better, so can probably fix it.)
  * It's a little buggy right now, too. Need to eliminate "mostly horizontal" lines?
  * Updates to viewport_3d.py: 
    * Refactored to use open3d.geometry.TriangleMesh() instead of trimesh.TriangleMesh().
    * Fixes for multiple file loading. (One at a time.)
    * If custom labels are not provided, they are generated using mesh depth values. These are displayed at the ends of the grid lines, after 'G' and 'D' keystrokes.
  * Fixes for spinner.py.
  * Minor refactorings and fixings. :-)
* 0.5.15:
  * Added blend amount slider option to blend between original and processed image.
  * Updated image processing to blend two images by a given percentage. Images can be mixed formats. (Grayscale, color, etc.)
  * Tweaks for UI layout.
* 0.5.14:
  * Tweaks to depth map generation and uses. Now discards values < 0. I think this helped....
  * Prints info about this as depth map is processed. (Just in some other model comes out all < 0, or something! Ha.)
  * Added tooltips to combo box items.
* 0.5.13:
  * Big UI update. Everything has tooltips now. Labels stick to associated widgets.
  * Fixed image scaling issues at startup.
* 0.5.12:
  * Add hotkeys to buttons.
  * Fix for direction of faces for flat back mesh.
  * Add Images/example.png for testing. Set as default image if config.ini is missing.
  * Add run.bat and run.sh to easily run the program from a command line. The .sh assumes python3 is in your path.
* 0.5.11:
  * Much improved "flat back" functionality.
* 0.5.10:
  * Added "pick color" to let you pick the background color from the image. (Or from anything on your desktop!)
  * After you pick it, the background will not be detected until you click "Clear Color"
  * Improved UI layout. Saves a few settings I missed before.
  * Fixes issues with windows size, position and state saving/loading.
* 0.5.9:
  * Started reworking UI. Window now can be sized smaller and still access all widgets.
  * Saves/restores window size and position in config.ini.
  * Fixed window name of 3D viewport to include the file name if given.
* 0.5.8:
  * Added _version to store the version so the main file is not always updated.
  * Removed __last_updated__ from the main file since I didn't update it.
* 0.5.7:
  * Added rotate left and right with arrow keys in 3D viewport. It's slow, but it works. Mouse is responsive.
  * Note the new MeshManipulation.move() function is untested, but it's not used, yet. :)
* 0.5.6:
  * Fix bug I just created in depth_to_3d.py
* 0.5.5:
  * Added save/load configuration for all user inputs.
  * Added "Reset Defaults" button to reset all user inputs to default values.
  * Yet more color fixes. (I hope this is the last time I have to say that.) <-- Suggested by CoPilot! (The part in parentheses was! Haha!)
* 0.5.4:
  * Added a 'D' toggle to the 'G' grid toggle. It toggles between percentages and depth values. 
    Sides of mesh are coming out colored based on the front/back portion. Need to look into that, but sides are flat.
* 0.5.3:
  * Move history to ReadMe.md
  * Add mesh_gradient_colorizer to shade the mesh based on depth, given a range of colors. Press 'C' for "Colorize".
  * Window can now be sized much smaller. (Still need to rework UI to organize widgets better!)
  * Added the FlowLayout class to implement this. (It's a start.)
  * Disabled tensor OneDNN option. It may be slower, but it's more accurate. And it's not much slower.
* 0.5.2:
  * Updates: Grid lines are now rainbow colored and numbered at each end. 
  * Added text_3d.py for 3D text generation in color, at position.
* 0.5.1:
  * Adds: G key now toggles a "measuring grid" on and off. It's just color lines on the edge of the mesh, right now.
    This should come in handy to figure out what percentage depth drop to use!
* 0.5.0:
  * Fixes:
    * Grayscale mode works without edge detection.
    * Only flip the image before processing when DepthAnythingV2 is used. (Not sure why it reverses depth map?)
    * ThreeDViewport class no longer binds some keys, mostly so W for wireframe works.
  * Adds:
    * Tweaks to depth map to allow modifications.
* 0.4.3:
  * UI improvements. Added tooltips to all input fields and buttons.
  * Fixed the depth removal by percentage. 
* 0.4.2:
  * Corrected input validation for depth to allow float.
* 0.4.1:
  * Implements depth amount. This is just a simple multiplier for the depth map. (Still does not use "max depth".)
  * Changed default depth amount to 1.0.
  * Fixes a bug where the processed image was not displayed correctly in the preview at startup.
  * Fix problems when using grayscale mode.
  * Tweak output file name to include more information.
* 0.4.0
  * Fixes several issues listed as "Known Issues" in the previous commit. Good checkpoint.
* 0.3.7 
  * Fixes: 
    * Color issues (again). 
  * Adds: 
    * "Project on Original" option. Works with and without grayscale enabled.
  * Known Issues:
    * Preview window may not update correctly. Change options a time or two to "fix it". 
    * Preview image scaling is not perfect at startup. Resize the window to fix it. (For now.)
* 0.3.6 Adds:
  * Add method to mirror and stitch the generated mesh. Not perfect, but it's a start. 
  * You can still use the old method (no mirrored back) by checking "Dynamic Depth". That's better for 3D printing.
  * Adds spinner.py for fun.
* 0.3.5 Adds: 
  * Includes "date_time" at the end output file name, so you'll always get a new file! 
                  *** Clean your <output folder> as needed. ***
  * Use Processed Image" option to create 3D mesh from current results, i.e., the image on the current right in GUI.
  * Dynamic Depth" option to make the mesh dynamically shaped on the back, approximating the front.
  * Edge Detection" option to enable or disable edge detection.
  * Grayscale option to enable or disable grayscale mode.
  * Notes: "Invert Colors" is intended for edge detection. Should it just be automatic? IDK, because it's fun
              to have the option. It's like a filter. If you enable it, colors become their opposite, or complementary
              color. This may be useful sometimes for a grayscale image. Or, if you have a picture of a negative! 
              So I think it's fun _and_ useful.
        ***Warning*** Use your new inverted colors with care.  
  *Fixes:
    * Background color for 3D viewport.
    * Supports 0 for resolution input. When you use this, the maximum of the image's width or height is used.
  * Updated:
    * ReadMe.md    
* 0.3.4 Adds background color as background color for 3D viewport. For more expected user experience.
              Note this background color is not saved in the exported mesh. It's simply carried over from
              the original image to the Open3D viewport.
* 0.3.3 Adds automatic background removal. This only happens when all four corners are a solid color.
* 0.3.2 Reopens the 3D viewport when closed, and clears existing geometry before loading a new mesh. 
* 0.3.1 Adds full color .PLY export support and a new depth map smoothing method.
              viewport_3d.py now has export_mesh_as_obj and export_mesh_as_stl methods, and a SUPPORTED_EXTENSIONS list.
                             It takes a list of mesh files as arguments and opens a viewport for each valid file. 
                             (One at a time.) Be careful with many files! *Especially* if they're large. 
                             Large files take a few seconds to minutes to load, depending on your system specs.
              Fixes 3D viewport update issue. (Now controlled by mouse.)
* 0.3.0 adds depth map smoothing options and anisotropic diffusion.
* 0.2.0 adds depth map generation from shading and light cues.
* 0.1.0 adds edge detection and 3D mesh generation from edges.


*   **WIP 20250213** *Leland Green, Springfield, MO, USA*   
*   *This is a work in progress.*
*   **Please report any issues via GitHub/lelandg/EdgeMesh/issues.**