# 3D viewer camera controls

All controls navigate the camera. They leave mesh vertices, topology, colors, and exported geometry unchanged. Axis actions use fixed world X, Y, and Z, including after freehand orbit. The small axes marker shows that coordinate frame.

| Action | Mouse | Keyboard with focus in the viewer |
| --- | --- | --- |
| Zoom | Wheel | `+` / `=` in; `-` out |
| Translate camera and target on world X | Shift + wheel | Ctrl + Left / Right |
| Translate camera and target on world Y | Ctrl + wheel | Ctrl + Down / Up |
| Translate camera and target on world Z | Alt + wheel | Ctrl + Page Down / Page Up |
| Orbit about world X through the camera target | Ctrl + Shift + wheel | Alt + Down / Up |
| Orbit about world Y through the camera target | Alt + Shift + wheel | Alt + Left / Right |
| Orbit about world Z through the camera target | Ctrl + Alt + wheel | Alt + Page Down / Page Up |
| Freehand orbit | Left drag | Use the axis orbit buttons or shortcuts |
| Pan in the current screen plane | Shift + left drag or middle drag | Use the axis translation buttons or shortcuts |
| Fit the whole mesh | Fit button | F |
| Toggle perspective / orthographic | Projection selector | O |
| Cycle surface / edges / wireframe | Display selector | W |
| Toggle orientation axes | Axes checkbox | A |
| Isometric / front / back / left / right / top / bottom | View selector | 1 / 2 / 3 / 4 / 5 / 6 / 7 |

Positive wheel motion moves toward the positive world axis or rotates by the right-hand rule. Each wheel notch or keypress rotates five degrees. Translation uses 10% of the current vertical half-span, so the step remains useful at different zoom levels. Wheel motion without modifiers zooms by a factor of 1.1 per notch. Fine touchpad motion is proportional. Unsupported modifier combinations are consumed instead of unexpectedly zooming. These shortcuts work only while the viewer canvas or its buttons have focus. The selectors retain standard keyboard typing and Alt + Down to open their menus; other application fields retain their own keys. Every axis action is also a focusable button with an explicit accessible name and a shortcut tooltip.

## Saved views

`save_view_state()` returns a versioned JSON object containing camera position, target, up direction, projection, parallel scale, view angle, standard-view selection, display mode, axes visibility, and background color. `restore_view_state(data)` validates the whole object before changing the viewer and returns `False` for invalid or incompatible data. A saved camera can be restored before the first mesh is loaded without starting VTK or creating an OpenGL window. Camera clipping ranges are recalculated for the current mesh.

Replacing a successfully rendered mesh preserves the camera orientation and keeps pan and zoom relative to the mesh bounds, so resolution changes retain framing. Identical bounds retain the exact camera values. An explicitly saved camera takes precedence over relative framing. Fit or a standard-view selection explicitly reframes it. A failed mesh load retains the previous geometry and camera. `view_state_changed` signals successful control actions and the end of native drag interactions, allowing the application to save changes without repeated writes during a drag. Restoring state does not emit the signal.

## Model provenance in the viewer

The visible badge describes the **accepted mesh**, not whichever model is currently selected for a future generation. Non-commercial and unknown licensing use amber, with explicit text so color is never the only indication. The badge separately says whether provenance was verified. `set_provenance_status(text, usage_class="unknown", verified=False)` only presents the status supplied by the application; verification belongs to mesh metadata handling.

Camera preferences intentionally do not contain provenance. Loading a saved layout cannot replace the model/license record associated with a mesh. An NC badge describes license restrictions, not legal advice or tamper-proof enforcement. Inspect the model-specific terms and licensing links shown elsewhere in the application before distributing or using its output.
