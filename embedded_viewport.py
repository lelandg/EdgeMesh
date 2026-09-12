"""An in-process PySide6 mesh viewport with lazy VTK rendering.

The Open3D mesh is the source of truth for export and mesh-health analysis.
VTK receives a copy for display, so camera and display changes cannot modify
the generated geometry. Importing this module never creates an OpenGL window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

STANDARD_VIEWS = ("Isometric", "Front", "Back", "Left", "Right", "Top", "Bottom")
DISPLAY_MODES = ("Surface", "Surface + edges", "Wireframe")
PROJECTIONS = ("Perspective", "Orthographic")


def wheel_navigation(modifiers):
    """Map exact PySide6 modifiers to a camera operation on fixed world axes."""
    shift, control, alt = (
        Qt.KeyboardModifier.ShiftModifier,
        Qt.KeyboardModifier.ControlModifier,
        Qt.KeyboardModifier.AltModifier,
    )
    return {
        Qt.KeyboardModifier.NoModifier: ("zoom", None),
        shift: ("translate", "X"),
        control: ("translate", "Y"),
        alt: ("translate", "Z"),
        control | shift: ("orbit", "X"),
        alt | shift: ("orbit", "Y"),
        control | alt: ("orbit", "Z"),
    }.get(modifiers)


def validated_view_state(data):
    """Return fresh JSON-safe state, rejecting unsafe or degenerate cameras."""
    if (
        not isinstance(data, dict)
        or type(data.get("version")) is not int
        or data["version"] != 1
    ):
        raise ValueError("Unsupported mesh view state version.")
    view, mode, projection = (
        data.get("view"),
        data.get("display_mode"),
        data.get("projection"),
    )
    if (
        view not in STANDARD_VIEWS
        or mode not in DISPLAY_MODES
        or projection not in PROJECTIONS
    ):
        raise ValueError("Unknown mesh view, display mode, or projection.")
    if type(data.get("axes_visible")) is not bool:
        raise ValueError("Mesh view axes visibility must be a boolean.")
    background = normalized_background(data.get("background"))
    result = {
        "version": 1,
        "view": view,
        "display_mode": mode,
        "projection": projection,
        "axes_visible": data["axes_visible"],
        "background": list(background),
        "camera": None,
    }
    camera = data.get("camera")
    if camera is not None:
        if not isinstance(camera, dict):
            raise ValueError("Mesh view camera must be an object.")
        clean = {}
        for name in ("position", "focal_point", "view_up"):
            value = camera.get(name)
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise ValueError(f"Camera {name} must contain three numbers.")
            if any(type(part) not in (int, float) for part in value):
                raise ValueError(f"Camera {name} must contain three numbers.")
            vector = np.asarray(value, dtype=float)
            if not np.isfinite(vector).all() or np.any(np.abs(vector) > 1e12):
                raise ValueError(f"Camera {name} is outside the supported range.")
            clean[name] = vector.tolist()
        direction = np.subtract(clean["position"], clean["focal_point"])
        direction_length = np.linalg.norm(direction)
        up = np.asarray(clean["view_up"])
        up_length = np.linalg.norm(up)
        if direction_length < 1e-9 or up_length < 1e-9:
            raise ValueError(
                "Camera position, target, and up direction are degenerate."
            )
        if (
            np.linalg.norm(np.cross(direction / direction_length, up / up_length))
            < 1e-6
        ):
            raise ValueError(
                "Camera up direction is parallel to its viewing direction."
            )
        for name, lower, upper in (
            ("parallel_scale", 1e-12, 1e12),
            ("view_angle", 0.01, 179.0),
        ):
            value = camera.get(name)
            if (
                type(value) not in (int, float)
                or not np.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError(f"Camera {name} is outside the supported range.")
            clean[name] = float(value)
        result["camera"] = clean
    return result


class _WheelNavigationFilter(QObject):
    """Intercept wheel input, leaving VTK's native drag controls intact."""

    def eventFilter(self, watched, event):
        if isinstance(watched, QComboBox):
            if event.type() == QEvent.Type.ShortcutOverride:
                sequence = QKeySequence(event.keyCombination())
                if any(
                    sequence == shortcut.key() for shortcut in self.parent()._shortcuts
                ):
                    # Preserve selector typing and Alt+Down without blocking
                    # unrelated application shortcuts such as Ctrl+S.
                    event.accept()
                    return True
            return False
        if event.type() != QEvent.Type.Wheel:
            return False
        mapping = wheel_navigation(event.modifiers())
        delta = event.angleDelta().y() / 120.0
        if not delta:
            delta = event.pixelDelta().y() / 60.0
        if mapping is not None and delta:
            self.parent()._invoke(
                "navigate_camera", *mapping, max(-10.0, min(10.0, delta))
            )
        # Do not let unsupported combinations fall through to VTK's zoom.
        event.accept()
        return True


def _logger():
    # Initialize the application's per-user rotating log only when needed.
    from log_utils import get_logger

    return get_logger(__name__)


def validated_mesh(mesh_or_path):
    """Read/copy a triangle mesh and reject geometry unsafe to send to VTK.

    Topology defects such as zero-area faces remain available to the separate
    health report; validation here only enforces the renderer's data contract.
    """
    import open3d as o3d

    if isinstance(mesh_or_path, (str, Path)):
        path = Path(mesh_or_path)
        if not path.is_file():
            raise FileNotFoundError(f"Mesh file does not exist: {path}")
        candidate = o3d.io.read_triangle_mesh(str(path))
    elif isinstance(mesh_or_path, o3d.geometry.TriangleMesh):
        candidate = o3d.geometry.TriangleMesh(mesh_or_path)
    else:
        raise TypeError("Expected an Open3D triangle mesh or a mesh file path.")

    vertices = np.asarray(candidate.vertices)
    triangles = np.asarray(candidate.triangles)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
        raise ValueError("The mesh has no vertices.")
    if triangles.ndim != 2 or triangles.shape[1:] != (3,) or not len(triangles):
        raise ValueError("The mesh has no triangle faces.")
    if not np.isfinite(vertices).all():
        raise ValueError("The mesh contains non-finite vertex coordinates.")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("The mesh contains triangle indices outside its vertices.")
    colors = np.asarray(candidate.vertex_colors)
    if colors.size and (
        colors.shape != vertices.shape or not np.isfinite(colors).all()
    ):
        raise ValueError("The mesh contains invalid vertex colors.")
    candidate.compute_vertex_normals()
    return candidate


def mesh_to_polydata(mesh):
    """Convert validated Open3D geometry to independent VTK display buffers."""
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.vertices), deep=True))
    triangles = np.asarray(mesh.triangles)
    cells = np.empty((len(triangles), 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = triangles
    polygons = vtkCellArray()
    polygons.SetCells(len(triangles), numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polygons)
    if mesh.has_vertex_normals():
        normals = numpy_to_vtk(np.asarray(mesh.vertex_normals), deep=True)
        normals.SetName("Normals")
        polydata.GetPointData().SetNormals(normals)
    if mesh.has_vertex_colors():
        colors = np.rint(np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(
            np.uint8
        )
        scalars = numpy_to_vtk(colors, deep=True)
        scalars.SetName("Colors")
        polydata.GetPointData().SetScalars(scalars)
    return polydata


def normalized_background(color):
    """Accept QColor, normalized RGB, or 0-255 RGB as a normalized tuple."""
    if hasattr(color, "getRgbF"):
        color = color.getRgbF()[:3]
    rgb = np.asarray(color, dtype=float)
    if (
        rgb.shape != (3,)
        or not np.isfinite(rgb).all()
        or np.any(rgb < 0)
        or np.any(rgb > 255)
    ):
        raise ValueError("Background must contain three RGB values from 0 to 255.")
    if np.any(rgb > 1):
        rgb = rgb / 255.0
    return tuple(float(value) for value in rgb)


class _VTKRenderer:
    """Native rendering adapter, constructed only when geometry is loaded."""

    def __init__(self, parent):
        # Register VTK's native implementations before constructing the widget.
        import vtkmodules.vtkInteractionStyle  # noqa: F401
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
        from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
        from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
        from vtkmodules.vtkRenderingCore import vtkRenderer

        self.widget = QVTKRenderWindowInteractor(parent)
        self.widget.setMinimumSize(220, 180)
        self.widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.window = self.widget.GetRenderWindow()
        self.window.SetMultiSamples(0)
        self.renderer = vtkRenderer()
        self.window.AddRenderer(self.renderer)
        self.interactor = self.window.GetInteractor()
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        self.actor = None
        self.mode = "Surface"
        self._axes_enabled = True
        self._initialized = False
        self._render_failed = False
        self.axes = vtkOrientationMarkerWidget()
        self.axes.SetOrientationMarker(vtkAxesActor())
        self.axes.SetInteractor(self.interactor)
        self.axes.SetViewport(0.0, 0.0, 0.2, 0.2)
        self._error_observer = self.window.AddObserver(
            "ErrorEvent", self._log_vtk_error
        )
        self._interaction_observer = self.interactor.AddObserver(
            "EndInteractionEvent",
            lambda _caller, _event: parent.view_state_changed.emit(),
        )

    def _log_vtk_error(self, _caller, event):
        self._render_failed = True
        _logger().error("VTK render window reported %s", event)

    def _render(self):
        self._render_failed = False
        self.window.Render()
        if self._render_failed:
            raise RuntimeError(
                "The graphics driver could not render the mesh. See the application log."
            )

    def set_mesh(
        self,
        mesh,
        *,
        view="Isometric",
        mode="Surface",
        projection="Perspective",
        camera_state=None,
        preserve_camera=True,
    ):
        from vtkmodules.vtkRenderingCore import vtkActor, vtkCamera, vtkPolyDataMapper

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(mesh_to_polydata(mesh))
        if mesh.has_vertex_colors():
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()
        else:
            mapper.ScalarVisibilityOff()
        candidate = vtkActor()
        candidate.SetMapper(mapper)
        candidate.GetProperty().SetColor(0.70, 0.83, 0.90)
        candidate.GetProperty().SetAmbient(0.22)
        candidate.GetProperty().SetDiffuse(0.78)
        candidate.GetProperty().SetSpecular(0.15)
        candidate.GetProperty().SetSpecularPower(20)
        previous = self.actor
        previous_mode = self.mode
        previous_camera = vtkCamera()
        previous_camera.DeepCopy(self.renderer.GetActiveCamera())
        try:
            self.renderer.AddActor(candidate)
            if previous is not None:
                self.renderer.RemoveActor(previous)
            self.actor = candidate
            self.mode = mode
            self._apply_mode()
            if camera_state is not None:
                self._restore_camera(camera_state)
            elif previous is None or not preserve_camera:
                self.set_view(view, render=False)
            else:
                self._preserve_relative_framing(previous, candidate)
            self.renderer.GetActiveCamera().SetParallelProjection(
                projection == "Orthographic"
            )
            self.renderer.ResetCameraClippingRange()
            if not self._initialized:
                self._render_failed = False
                self.interactor.Initialize()
                if self._render_failed:
                    raise RuntimeError(
                        "The graphics driver could not initialize the mesh preview."
                    )
                self.axes.SetEnabled(self._axes_enabled)
                if self._axes_enabled:
                    self.axes.InteractiveOff()
                self._initialized = True
            self._render()
        except Exception:
            self.renderer.RemoveActor(candidate)
            self.actor = previous
            self.mode = previous_mode
            if previous is not None:
                self.renderer.AddActor(previous)
            self.renderer.GetActiveCamera().DeepCopy(previous_camera)
            try:
                self._render()
            except Exception:
                _logger().exception(
                    "Could not redraw the previous mesh after a failed load"
                )
            raise

    def _apply_mode(self):
        if self.actor is None:
            return
        prop = self.actor.GetProperty()
        if self.mode == "Wireframe":
            prop.SetRepresentationToWireframe()
            prop.EdgeVisibilityOff()
        else:
            prop.SetRepresentationToSurface()
            prop.SetEdgeVisibility(self.mode == "Surface + edges")
        prop.SetEdgeColor(0.12, 0.16, 0.20)

    def set_display_mode(self, mode):
        self.mode = mode
        self._apply_mode()
        self._render()

    def set_view(self, view, *, render=True):
        directions = {
            "Isometric": ((1, -1, 1), (0, 0, 1)),
            "Front": ((0, 0, 1), (0, 1, 0)),
            "Back": ((0, 0, -1), (0, 1, 0)),
            "Left": ((-1, 0, 0), (0, 0, 1)),
            "Right": ((1, 0, 0), (0, 0, 1)),
            "Top": ((0, 1, 0), (0, 0, -1)),
            "Bottom": ((0, -1, 0), (0, 0, 1)),
        }
        direction, up = directions[view]
        camera = self.renderer.GetActiveCamera()
        center = self.actor.GetCenter() if self.actor is not None else (0, 0, 0)
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center[index] + direction[index] for index in range(3)))
        camera.SetViewUp(*up)
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        if render:
            self._render()

    def _preserve_relative_framing(self, previous, candidate):
        """Keep inspection orientation, pan, and zoom relative to mesh bounds."""
        old_bounds, new_bounds = previous.GetBounds(), candidate.GetBounds()
        if old_bounds is None or new_bounds is None or old_bounds == new_bounds:
            return
        old_bounds = np.asarray(old_bounds, dtype=float).reshape(3, 2)
        new_bounds = np.asarray(new_bounds, dtype=float).reshape(3, 2)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            old_extent = old_bounds[:, 1] - old_bounds[:, 0]
            new_extent = new_bounds[:, 1] - new_bounds[:, 0]
            if (old_extent < 0).any() or (new_extent < 0).any():
                return
            old_radius = np.hypot.reduce(old_extent) / 2
            new_radius = np.hypot.reduce(new_extent) / 2
            if (
                not np.isfinite([old_radius, new_radius]).all()
                or min(old_radius, new_radius) <= 1e-12
            ):
                return
            ratio = new_radius / old_radius
            old_center = old_bounds[:, 0] / 2 + old_bounds[:, 1] / 2
            new_center = new_bounds[:, 0] / 2 + new_bounds[:, 1] / 2
            camera = self.renderer.GetActiveCamera()
            position = (
                new_center + (np.asarray(camera.GetPosition()) - old_center) * ratio
            )
            focal = (
                new_center + (np.asarray(camera.GetFocalPoint()) - old_center) * ratio
            )
            parallel_scale = camera.GetParallelScale() * ratio
        if (
            not np.isfinite(position).all()
            or not np.isfinite(focal).all()
            or not np.isfinite(parallel_scale)
        ):
            return
        if (
            np.any(np.abs(position) > 1e12)
            or np.any(np.abs(focal) > 1e12)
            or not 1e-12 <= parallel_scale <= 1e12
        ):
            return
        if np.linalg.norm(position - focal) < 1e-9:
            return
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal)
        camera.SetParallelScale(parallel_scale)

    def camera_state(self):
        camera = self.renderer.GetActiveCamera()
        return {
            "position": list(camera.GetPosition()),
            "focal_point": list(camera.GetFocalPoint()),
            "view_up": list(camera.GetViewUp()),
            "parallel_scale": camera.GetParallelScale(),
            "view_angle": camera.GetViewAngle(),
        }

    def _restore_camera(self, state):
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*state["position"])
        camera.SetFocalPoint(*state["focal_point"])
        camera.SetViewUp(*state["view_up"])
        camera.SetParallelScale(state["parallel_scale"])
        camera.SetViewAngle(state["view_angle"])
        self.renderer.ResetCameraClippingRange()

    def set_projection(self, projection):
        self.renderer.GetActiveCamera().SetParallelProjection(
            projection == "Orthographic"
        )
        self.renderer.ResetCameraClippingRange()
        self._render()

    def apply_view_state(self, state):
        """Apply an already validated state, rolling back failed rendering."""
        from vtkmodules.vtkRenderingCore import vtkCamera

        previous_camera = vtkCamera()
        previous_camera.DeepCopy(self.renderer.GetActiveCamera())
        previous_mode, previous_axes = self.mode, self._axes_enabled
        previous_background = self.renderer.GetBackground()
        try:
            self.mode = state["display_mode"]
            self._apply_mode()
            self.renderer.SetBackground(*state["background"])
            self._axes_enabled = state["axes_visible"]
            if self._initialized:
                self.axes.SetEnabled(self._axes_enabled)
                if self._axes_enabled:
                    self.axes.InteractiveOff()
            if state["camera"] is None:
                self.set_view(state["view"], render=False)
            else:
                self._restore_camera(state["camera"])
            self.renderer.GetActiveCamera().SetParallelProjection(
                state["projection"] == "Orthographic"
            )
            self.renderer.ResetCameraClippingRange()
            self._render()
        except Exception:
            self.renderer.GetActiveCamera().DeepCopy(previous_camera)
            self.mode, self._axes_enabled = previous_mode, previous_axes
            self._apply_mode()
            self.renderer.SetBackground(*previous_background)
            if self._initialized:
                self.axes.SetEnabled(previous_axes)
                if previous_axes:
                    self.axes.InteractiveOff()
            try:
                self._render()
            except Exception:
                _logger().exception(
                    "Could not redraw the previous camera after a failed restore"
                )
            raise

    def navigate_camera(self, operation, axis, steps):
        """Move/orbit the camera along world axes without touching geometry."""
        if operation not in ("zoom", "translate", "orbit") or not np.isfinite(steps):
            raise ValueError("Invalid camera navigation operation.")
        if operation != "zoom" and axis not in ("X", "Y", "Z"):
            raise ValueError("Camera navigation requires a world X, Y, or Z axis.")
        steps = max(-10.0, min(10.0, float(steps)))
        camera = self.renderer.GetActiveCamera()
        position, focal = (
            np.asarray(camera.GetPosition()),
            np.asarray(camera.GetFocalPoint()),
        )
        if operation == "zoom":
            factor = 1.1**steps
            if camera.GetParallelProjection():
                camera.SetParallelScale(
                    max(1e-9, min(1e12, camera.GetParallelScale() / factor))
                )
            else:
                offset = position - focal
                distance = np.linalg.norm(offset)
                target_distance = max(1e-9, min(1e12, distance / factor))
                if distance > 0:
                    camera.SetPosition(*(focal + offset * (target_distance / distance)))
        elif operation == "translate":
            if camera.GetParallelProjection():
                half_height = camera.GetParallelScale()
            else:
                half_height = np.linalg.norm(position - focal) * np.tan(
                    np.radians(camera.GetViewAngle()) / 2
                )
            delta = np.zeros(3)
            delta[("X", "Y", "Z").index(axis)] = max(1e-9, half_height) * 0.1 * steps
            camera.SetPosition(*(position + delta))
            camera.SetFocalPoint(*(focal + delta))
        else:
            # Rodrigues rotation, with a fixed world axis through the focal point.
            unit = np.eye(3)[("X", "Y", "Z").index(axis)]
            radians = np.radians(5.0 * steps)
            cosine, sine = np.cos(radians), np.sin(radians)

            def rotate(vector):
                return (
                    vector * cosine
                    + np.cross(unit, vector) * sine
                    + unit * np.dot(unit, vector) * (1 - cosine)
                )

            camera.SetPosition(*(focal + rotate(position - focal)))
            camera.SetViewUp(*rotate(np.asarray(camera.GetViewUp())))
            camera.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()
        self._render()

    def reset_camera(self):
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self._render()

    def set_axes_visible(self, visible):
        self._axes_enabled = visible
        if self._initialized:
            self.axes.SetEnabled(visible)
            if visible:
                self.axes.InteractiveOff()
            self._render()

    def set_background_color(self, color):
        self.renderer.SetBackground(*color)
        if self._initialized:
            self._render()

    def shutdown(self):
        self.widget.setUpdatesEnabled(False)
        self.widget.hide()
        if self._initialized:
            self.axes.SetEnabled(False)
        self.axes.SetInteractor(None)
        self.interactor.RemoveObserver(self._interaction_observer)
        self.interactor.Disable()
        self.window.Finalize()
        self.window.RemoveObserver(self._error_observer)


class EmbeddedMeshViewport(QWidget):
    """PySide6-hosted preview exposing the existing viewport's mesh/export API."""

    error_occurred = Signal(str)
    view_state_changed = Signal()

    def __init__(self, parent=None, *, renderer_factory=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("3D mesh viewer")
        self.mesh = None
        self.mesh_file = None
        self.custom_labels = None
        self._renderer_factory = renderer_factory or _VTKRenderer
        self._backend = None
        self._background = (0.10, 0.13, 0.17)
        self._pending_camera = None
        self._wheel_filter = _WheelNavigationFilter(self)
        self._shortcuts = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        controls = QHBoxLayout()
        self.view_combo = QComboBox()
        self.view_combo.addItems(STANDARD_VIEWS)
        self.view_combo.setAccessibleName("Standard camera view")
        self.view_combo.setToolTip("Standard view (keys 1-7 in the viewer)")
        self.view_combo.textActivated.connect(self._select_view)
        controls.addWidget(self.view_combo)
        self.fit_button = QPushButton("Fit")
        self.fit_button.setAccessibleName("Fit mesh to camera view")
        self.fit_button.setToolTip("Fit the whole mesh in the current view (F)")
        self.fit_button.clicked.connect(self._fit_camera)
        controls.addWidget(self.fit_button)
        self.display_combo = QComboBox()
        self.display_combo.addItems(DISPLAY_MODES)
        self.display_combo.setAccessibleName("Mesh display mode")
        self.display_combo.setToolTip(
            "Surface, surface with edges, or wireframe (W cycles)"
        )
        self.display_combo.currentTextChanged.connect(
            lambda mode: self._invoke("set_display_mode", mode)
        )
        controls.addWidget(self.display_combo)
        secondary_controls = QHBoxLayout()
        self.projection_combo = QComboBox()
        self.projection_combo.addItems(PROJECTIONS)
        self.projection_combo.setAccessibleName("Camera projection")
        self.projection_combo.setToolTip(
            "Perspective or orthographic projection (O toggles)"
        )
        self.projection_combo.currentTextChanged.connect(
            lambda value: self._invoke("set_projection", value)
        )
        secondary_controls.addWidget(self.projection_combo)
        self.axes_checkbox = QCheckBox("Axes")
        self.axes_checkbox.setChecked(True)
        self.axes_checkbox.setToolTip("Show world-axis orientation (A toggles)")
        self.axes_checkbox.toggled.connect(
            lambda visible: self._invoke("set_axes_visible", visible)
        )
        secondary_controls.addWidget(self.axes_checkbox)
        controls.addStretch()
        layout.addLayout(controls)
        secondary_controls.addStretch()
        layout.addLayout(secondary_controls)
        for selector in (self.view_combo, self.display_combo, self.projection_combo):
            selector.installEventFilter(self._wheel_filter)
        self.navigation_buttons = {}
        keyboard_pairs = {
            ("translate", "X"): ("Ctrl+Left", "Ctrl+Right"),
            ("translate", "Y"): ("Ctrl+Down", "Ctrl+Up"),
            ("translate", "Z"): ("Ctrl+PgDown", "Ctrl+PgUp"),
            ("orbit", "X"): ("Alt+Down", "Alt+Up"),
            ("orbit", "Y"): ("Alt+Left", "Alt+Right"),
            ("orbit", "Z"): ("Alt+PgDown", "Alt+PgUp"),
        }
        for operation, label in (
            ("translate", "Move camera"),
            ("orbit", "Orbit camera"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            for axis in ("X", "Y", "Z"):
                for index, step in enumerate((-1, 1)):
                    key = keyboard_pairs[(operation, axis)][index]
                    direction = "negative" if step < 0 else "positive"
                    button = QPushButton(f"{axis}{'-' if step < 0 else '+'}")
                    button.setMinimumWidth(34)
                    button.setMaximumWidth(56)
                    button.setAccessibleName(
                        f"{operation.title()} camera on {direction} world {axis} axis"
                    )
                    button.setToolTip(
                        f"{operation.title()} camera on {direction} world {axis} axis ({key})"
                    )

                    def callback(op=operation, ax=axis, amount=step):
                        return self._invoke("navigate_camera", op, ax, amount)

                    button.clicked.connect(
                        lambda _checked=False, action=callback: action()
                    )
                    self._add_shortcut(key, callback)
                    self.navigation_buttons[(operation, axis, step)] = button
                    row.addWidget(button)
            row.addStretch()
            layout.addLayout(row)
        self.provenance_label = QLabel()
        self.provenance_label.setWordWrap(True)
        self.provenance_label.setTextFormat(Qt.TextFormat.PlainText)
        self.provenance_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.provenance_label.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.set_provenance_status("No model provenance loaded")
        layout.addWidget(self.provenance_label)
        self._render_layout = QStackedLayout()
        self.placeholder = QLabel("Generate or open a mesh to preview it here.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setMinimumSize(220, 180)
        self._render_layout.addWidget(self.placeholder)
        layout.addLayout(self._render_layout, 1)
        help_text = QLabel(
            "Drag: orbit · Shift + drag / middle drag: pan · Wheel: zoom\n"
            "Shift / Ctrl / Alt + wheel: move world X / Y / Z\n"
            "Ctrl+Shift / Alt+Shift / Ctrl+Alt + wheel: orbit world X / Y / Z"
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.stats_label = QLabel("No mesh loaded")
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.stats_label)
        self._add_shortcut("F", self._fit_camera)
        self._add_shortcut(
            "O",
            lambda: self.projection_combo.setCurrentIndex(
                1 - self.projection_combo.currentIndex()
            ),
        )
        self._add_shortcut(
            "W",
            lambda: self.display_combo.setCurrentIndex(
                (self.display_combo.currentIndex() + 1) % len(DISPLAY_MODES)
            ),
        )
        self._add_shortcut("A", self.axes_checkbox.toggle)
        self._add_shortcut(
            "+", lambda: self._invoke("navigate_camera", "zoom", None, 1)
        )
        self._add_shortcut(
            "=", lambda: self._invoke("navigate_camera", "zoom", None, 1)
        )
        self._add_shortcut(
            "-", lambda: self._invoke("navigate_camera", "zoom", None, -1)
        )
        for index, view in enumerate(STANDARD_VIEWS, 1):
            self._add_shortcut(
                str(index), lambda selected=view: self._select_view(selected)
            )

    def _add_shortcut(self, key, callback):
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(callback)
        self._shortcuts.append(shortcut)

    def _select_view(self, view):
        self._pending_camera = None
        self.view_combo.setCurrentText(view)
        self._invoke("set_view", view)

    def _fit_camera(self):
        self._pending_camera = None
        self._invoke("reset_camera")

    def _invoke(self, method, *args):
        try:
            if self._backend is not None:
                getattr(self._backend, method)(*args)
        except Exception as error:
            _logger().exception("Mesh preview action failed: %s", method)
            self.error_occurred.emit(f"Could not update the mesh preview: {error}")
            return False
        self.view_state_changed.emit()
        return True

    def save_view_state(self):
        """Return JSON camera/preferences; provenance is owned by mesh metadata."""
        camera = (
            self._backend.camera_state()
            if self._backend is not None
            else self._pending_camera
        )
        return validated_view_state(
            {
                "version": 1,
                "view": self.view_combo.currentText(),
                "display_mode": self.display_combo.currentText(),
                "projection": self.projection_combo.currentText(),
                "axes_visible": self.axes_checkbox.isChecked(),
                "background": list(self._background),
                "camera": camera,
            }
        )

    def restore_view_state(self, data):
        """Restore atomically, or retain a validated camera until first render.

        Invalid persisted data is logged and returns False without changing the
        viewer. A successful restore deliberately emits no view_state_changed.
        """
        try:
            state = validated_view_state(data)
            if self._backend is not None:
                self._backend.apply_view_state(state)
        except Exception:
            _logger().exception("Could not restore the saved mesh view")
            return False
        blockers = [
            QSignalBlocker(control)
            for control in (
                self.view_combo,
                self.display_combo,
                self.projection_combo,
                self.axes_checkbox,
            )
        ]
        self.view_combo.setCurrentText(state["view"])
        self.display_combo.setCurrentText(state["display_mode"])
        self.projection_combo.setCurrentText(state["projection"])
        self.axes_checkbox.setChecked(state["axes_visible"])
        del blockers
        self._background = tuple(state["background"])
        self._pending_camera = state["camera"] if self._backend is None else None
        return True

    def set_provenance_status(self, text, usage_class="unknown", verified=False):
        """Show the accepted mesh's license/provenance, never selected-model UI.

        The caller verifies provenance. This presentation-only method neither
        verifies signatures nor persists a mutable substitute for mesh metadata.
        """
        usage = (
            str(usage_class).lower().replace("-", "").replace("_", "").replace(" ", "")
        )
        if usage in ("nc", "noncommercial", "researchonly"):
            label, background, foreground = (
                "NC - Non-commercial use",
                "#fff3cd",
                "#553c00",
            )
        elif usage in (
            "commercial",
            "commercialpermitted",
            "commercialallowed",
            "permissive",
        ):
            label, background, foreground = (
                "Commercial use permitted by model license",
                "#e7f1ff",
                "#153f67",
            )
        else:
            label, background, foreground = (
                (
                    "Custom model license - review terms"
                    if usage == "custom"
                    else "Model license unknown - review terms"
                ),
                "#fff3cd",
                "#553c00",
            )
        verification = (
            "Provenance verified" if verified is True else "Provenance unverified"
        )
        message = f"{label} · {verification}"
        if text:
            message += f"\n{text}"
        self.provenance_label.setText(message)
        self.provenance_label.setAccessibleName(message)
        self.provenance_label.setToolTip(message)
        self.provenance_label.setStyleSheet(
            f"QLabel {{ background: {background}; color: {foreground}; border-radius: 4px; padding: 6px; }}"
        )

    def load_mesh(self, mesh_or_path, custom_labels=None, *, depth_labels=None):
        """Replace the mesh only after validation and rendering both succeed."""
        backend = self._backend
        created = False
        try:
            candidate = validated_mesh(mesh_or_path)
            if backend is None:
                backend = self._renderer_factory(self)
                created = True
                self._render_layout.addWidget(backend.widget)
                self._render_layout.setCurrentWidget(backend.widget)
                backend.widget.setAccessibleName("Interactive 3D mesh camera")
                backend.widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                backend.widget.installEventFilter(self._wheel_filter)
                backend.widget.show()
                self.layout().activate()
                backend.set_background_color(self._background)
                backend.set_axes_visible(self.axes_checkbox.isChecked())
            backend.set_mesh(
                candidate,
                view=self.view_combo.currentText(),
                mode=self.display_combo.currentText(),
                projection=self.projection_combo.currentText(),
                camera_state=self._pending_camera,
                preserve_camera=not created,
            )
        except Exception:
            _logger().exception("Could not load the mesh into the embedded preview")
            if created:
                try:
                    backend.shutdown()
                except Exception:
                    _logger().exception("Could not release a failed mesh preview")
                self._render_layout.removeWidget(backend.widget)
                self._render_layout.setCurrentWidget(self.placeholder)
                backend.widget.deleteLater()
            raise
        self._backend = backend
        self._pending_camera = None
        self.mesh = candidate
        self.mesh_file = (
            str(mesh_or_path) if isinstance(mesh_or_path, (str, Path)) else None
        )
        self.custom_labels = (
            custom_labels if custom_labels is not None else depth_labels
        )
        extent = np.ptp(np.asarray(candidate.vertices), axis=0)
        self.stats_label.setText(
            f"{len(candidate.vertices):,} vertices  ·  {len(candidate.triangles):,} triangles  ·  "
            f"Size: {extent[0]:.3g} × {extent[1]:.3g} × {extent[2]:.3g} model units"
        )
        return True

    def set_background_color(self, color):
        try:
            self._background = normalized_background(color)
        except Exception:
            _logger().exception("Invalid mesh preview background color")
            raise
        self._invoke("set_background_color", self._background)

    def _export_mesh(self, output_path):
        import open3d as o3d

        try:
            if self.mesh is None:
                raise ValueError("Generate or open a mesh before exporting.")
            candidate = validated_mesh(self.mesh)
            if not o3d.io.write_triangle_mesh(
                str(output_path), candidate, write_ascii=False
            ):
                raise OSError(f"Could not write the mesh to {output_path}")
        except Exception:
            _logger().exception("Mesh export failed: %s", output_path)
            raise
        return True

    def export_mesh_as_obj(self, output_path):
        return self._export_mesh(output_path)

    def export_mesh_as_stl(self, output_path):
        return self._export_mesh(output_path)

    def shutdown(self):
        if self._backend is not None:
            self._pending_camera = self._backend.camera_state()
        backend, self._backend = self._backend, None
        if backend is not None:
            try:
                backend.shutdown()
            except Exception:
                _logger().exception("Could not release the embedded mesh preview")
            self._render_layout.removeWidget(backend.widget)
            backend.widget.deleteLater()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
