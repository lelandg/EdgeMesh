"""Geometry/Qt checks that do not create an OpenGL context."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import open3d as o3d
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget
from vtkmodules.util.numpy_support import vtk_to_numpy

from embedded_viewport import (
    EmbeddedMeshViewport,
    _VTKRenderer,
    mesh_to_polydata,
    normalized_background,
    validated_mesh,
    validated_view_state,
    wheel_navigation,
)


class FakeRenderer:
    def __init__(self, parent):
        self.widget = QWidget(parent)
        self.mesh = None
        self.fail_next = False
        self.stopped = False
        self.mode = None
        self.view = None
        self.projection = "Perspective"
        self.navigation = []
        self.camera = {
            "position": [4.0, -4.0, 4.0],
            "focal_point": [0.0, 0.0, 0.0],
            "view_up": [0.0, 0.0, 1.0],
            "parallel_scale": 1.0,
            "view_angle": 30.0,
        }

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
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("renderer rejected the candidate")
        self.mesh = mesh
        self.view = view
        self.mode = mode
        self.projection = projection
        self.preserve_camera = preserve_camera
        if camera_state is not None:
            self.camera = deepcopy(camera_state)

    def camera_state(self):
        return deepcopy(self.camera)

    def set_projection(self, projection):
        self.projection = projection

    def navigate_camera(self, operation, axis, steps):
        self.navigation.append((operation, axis, steps))

    def apply_view_state(self, state):
        self.view = state["view"]
        self.mode = state["display_mode"]
        self.projection = state["projection"]
        self.axes = state["axes_visible"]
        self.background = tuple(state["background"])
        if state["camera"] is not None:
            self.camera = deepcopy(state["camera"])

    def set_background_color(self, color):
        self.background = color

    def set_axes_visible(self, visible):
        self.axes = visible

    def set_display_mode(self, mode):
        self.mode = mode

    def set_view(self, view):
        self.view = view

    def reset_camera(self):
        pass

    def shutdown(self):
        self.stopped = True


class _IsolatedLogsTestCase(unittest.TestCase):
    def setUp(self):
        # Negative-path tests should not create files in the user's log folder.
        logger_patch = patch("embedded_viewport._logger")
        self.logged_errors = logger_patch.start().return_value
        self.addCleanup(logger_patch.stop)


class TestMeshGeometry(_IsolatedLogsTestCase):
    def test_validation_copies_geometry_without_changing_source(self):
        source = o3d.geometry.TriangleMesh.create_box()
        source.paint_uniform_color([0.2, 0.4, 0.6])
        candidate = validated_mesh(source)
        self.assertIsNot(source, candidate)
        self.assertFalse(source.has_vertex_normals())
        self.assertTrue(candidate.has_vertex_normals())
        np.testing.assert_array_equal(source.vertices, candidate.vertices)
        np.testing.assert_array_equal(source.triangles, candidate.triangles)
        candidate.translate([10, 0, 0])
        self.assertNotEqual(
            np.asarray(source.vertices)[0, 0], np.asarray(candidate.vertices)[0, 0]
        )

    def test_rejects_empty_nonfinite_and_invalid_indices(self):
        with self.assertRaisesRegex(ValueError, "vertices"):
            validated_mesh(o3d.geometry.TriangleMesh())
        source = o3d.geometry.TriangleMesh.create_box()
        np.asarray(source.vertices)[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validated_mesh(source)
        source = o3d.geometry.TriangleMesh.create_box()
        np.asarray(source.triangles)[0, 0] = len(source.vertices)
        with self.assertRaisesRegex(ValueError, "indices"):
            validated_mesh(source)
        np.asarray(source.triangles)[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "indices"):
            validated_mesh(source)

    def test_topology_defects_are_preserved_for_health_report(self):
        source = o3d.geometry.TriangleMesh.create_box()
        triangles = np.asarray(source.triangles).copy()
        triangles[0] = [0, 0, 0]
        source.triangles = o3d.utility.Vector3iVector(triangles)
        candidate = validated_mesh(source)
        np.testing.assert_array_equal(candidate.triangles, source.triangles)

    def test_vtk_conversion_preserves_all_faces_and_copies_buffers(self):
        source = validated_mesh(o3d.geometry.TriangleMesh.create_sphere(resolution=12))
        source.paint_uniform_color([1.0, 0.5, 0.0])
        polydata = mesh_to_polydata(source)
        self.assertEqual(polydata.GetNumberOfPoints(), len(source.vertices))
        self.assertEqual(polydata.GetNumberOfPolys(), len(source.triangles))
        points = vtk_to_numpy(polydata.GetPoints().GetData())
        np.testing.assert_array_equal(points, source.vertices)
        cells = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)
        np.testing.assert_array_equal(cells[:, 1:], source.triangles)
        colors = vtk_to_numpy(polydata.GetPointData().GetScalars())
        np.testing.assert_array_equal(colors[0], [255, 128, 0])
        before = points.copy()
        source.translate([1, 2, 3])
        np.testing.assert_array_equal(points, before)

    def test_background_accepts_both_rgb_ranges(self):
        self.assertEqual(normalized_background([255, 128, 0]), (1.0, 128 / 255.0, 0.0))
        self.assertEqual(normalized_background([0.1, 0.2, 0.3]), (0.1, 0.2, 0.3))
        for invalid in ([1, 2], [-1, 0, 0], [0, np.nan, 0], [256, 0, 0]):
            with self.subTest(color=invalid), self.assertRaises(ValueError):
                normalized_background(invalid)

    def test_native_adapter_rolls_back_actor_on_vtk_error_event(self):
        from vtkmodules.vtkRenderingCore import vtkActor, vtkRenderer

        backend = _VTKRenderer.__new__(_VTKRenderer)
        backend.renderer = vtkRenderer()
        previous = vtkActor()
        backend.actor = previous
        backend.renderer.AddActor(previous)
        backend.mode = "Wireframe"
        backend._initialized = True
        backend._render_failed = False
        original_position = backend.renderer.GetActiveCamera().GetPosition()

        class FailingWindow:
            attempts = 0

            def Render(self):
                self.attempts += 1
                if self.attempts == 1:
                    backend._log_vtk_error(self, "ErrorEvent")

        backend.window = FailingWindow()
        with self.assertRaisesRegex(RuntimeError, "graphics driver"):
            backend.set_mesh(validated_mesh(o3d.geometry.TriangleMesh.create_box()))
        self.assertIs(backend.actor, previous)
        self.assertEqual(backend.mode, "Wireframe")
        self.assertEqual(backend.renderer.GetActors().GetNumberOfItems(), 1)
        self.assertEqual(
            backend.renderer.GetActiveCamera().GetPosition(), original_position
        )
        self.assertEqual(backend.window.attempts, 2)

    def test_disabling_axes_does_not_configure_a_disabled_marker(self):
        backend = _VTKRenderer.__new__(_VTKRenderer)
        backend._initialized = True
        backend._render = lambda: None

        class Axes:
            enabled = True
            configured = 0

            def SetEnabled(self, enabled):
                self.enabled = enabled

            def InteractiveOff(self):
                if not self.enabled:
                    raise RuntimeError("Cannot configure a disabled orientation marker")
                self.configured += 1

        backend.axes = Axes()
        backend.set_axes_visible(False)
        self.assertFalse(backend.axes.enabled)
        self.assertEqual(backend.axes.configured, 0)
        backend.set_axes_visible(True)
        self.assertTrue(backend.axes.enabled)
        self.assertEqual(backend.axes.configured, 1)


class TestAxisNavigation(_IsolatedLogsTestCase):
    def make_backend(self):
        from vtkmodules.vtkRenderingCore import vtkRenderer

        backend = _VTKRenderer.__new__(_VTKRenderer)
        backend.renderer = vtkRenderer()
        backend.actor = None
        backend.mode = "Surface"
        backend._axes_enabled = True
        backend._initialized = False
        backend._render = lambda: None
        camera = backend.renderer.GetActiveCamera()
        camera.SetPosition(10, -8, 6)
        camera.SetFocalPoint(1, 2, 3)
        camera.SetViewUp(0, 0, 1)
        return backend

    def test_wheel_modifier_combinations_are_exact_and_unambiguous(self):
        shift, ctrl, alt = Qt.ShiftModifier, Qt.ControlModifier, Qt.AltModifier
        for modifiers, expected in (
            (Qt.NoModifier, ("zoom", None)),
            (shift, ("translate", "X")),
            (ctrl, ("translate", "Y")),
            (alt, ("translate", "Z")),
            (ctrl | shift, ("orbit", "X")),
            (alt | shift, ("orbit", "Y")),
            (ctrl | alt, ("orbit", "Z")),
            (ctrl | shift | alt, None),
            (Qt.MetaModifier, None),
        ):
            with self.subTest(modifiers=modifiers):
                self.assertEqual(wheel_navigation(modifiers), expected)

    def test_translation_stays_on_world_axes_after_orbit(self):
        backend = self.make_backend()
        backend.navigate_camera("orbit", "X", 3)
        backend.navigate_camera("orbit", "Z", -2)
        camera = backend.renderer.GetActiveCamera()
        for axis in ("X", "Y", "Z"):
            with self.subTest(axis=axis):
                position = np.asarray(camera.GetPosition())
                focal = np.asarray(camera.GetFocalPoint())
                before_up = camera.GetViewUp()
                backend.navigate_camera("translate", axis, 1)
                delta = np.asarray(camera.GetPosition()) - position
                expected = np.zeros(3)
                expected[("X", "Y", "Z").index(axis)] = delta[
                    ("X", "Y", "Z").index(axis)
                ]
                self.assertGreater(np.linalg.norm(delta), 0)
                np.testing.assert_allclose(delta, expected, atol=1e-12)
                np.testing.assert_allclose(
                    np.asarray(camera.GetFocalPoint()) - focal, delta, atol=1e-12
                )
                np.testing.assert_allclose(camera.GetViewUp(), before_up, atol=1e-12)

    def test_orbit_rotates_about_world_axis_through_focal_point(self):
        backend = self.make_backend()
        camera = backend.renderer.GetActiveCamera()
        camera.SetPosition(11, 2, 3)
        camera.SetFocalPoint(1, 2, 3)
        backend.navigate_camera("orbit", "Z", 1)
        radians = np.radians(5)
        np.testing.assert_allclose(
            camera.GetPosition(),
            [1 + 10 * np.cos(radians), 2 + 10 * np.sin(radians), 3],
        )
        np.testing.assert_allclose(camera.GetFocalPoint(), [1, 2, 3])
        np.testing.assert_allclose(camera.GetViewUp(), [0, 0, 1])
        self.assertAlmostEqual(camera.GetDistance(), 10)

    def test_zoom_uses_distance_or_parallel_scale_as_appropriate(self):
        backend = self.make_backend()
        camera = backend.renderer.GetActiveCamera()
        target = camera.GetFocalPoint()
        distance = camera.GetDistance()
        backend.navigate_camera("zoom", None, 1)
        self.assertAlmostEqual(camera.GetDistance(), distance / 1.1)
        self.assertEqual(camera.GetFocalPoint(), target)
        camera.SetParallelProjection(True)
        position = camera.GetPosition()
        scale = camera.GetParallelScale()
        backend.navigate_camera("zoom", None, -1)
        self.assertAlmostEqual(camera.GetParallelScale(), scale * 1.1)
        self.assertEqual(camera.GetPosition(), position)

    def test_camera_snapshot_round_trip_matches_native_camera(self):
        backend = self.make_backend()
        backend.navigate_camera("orbit", "Z", 3)
        backend.navigate_camera("translate", "X", -2)
        expected = backend.camera_state()
        backend.navigate_camera("orbit", "Y", 2)
        backend._restore_camera(expected)
        for key, value in expected.items():
            np.testing.assert_allclose(backend.camera_state()[key], value, atol=1e-12)

    def test_view_state_render_failure_restores_camera_and_preferences(self):
        backend = self.make_backend()
        original = backend.camera_state()
        original_background = backend.renderer.GetBackground()
        candidate = {
            "version": 1,
            "view": "Top",
            "display_mode": "Wireframe",
            "projection": "Orthographic",
            "axes_visible": False,
            "background": [0.8, 0.7, 0.6],
            "camera": deepcopy(original),
        }
        candidate["camera"]["position"] = [20, 30, 40]
        attempts = []

        def render():
            attempts.append(True)
            if len(attempts) == 1:
                raise RuntimeError("driver unavailable")

        backend._render = render
        with self.assertRaisesRegex(RuntimeError, "driver unavailable"):
            backend.apply_view_state(candidate)
        self.assertEqual(backend.camera_state(), original)
        self.assertEqual(backend.renderer.GetBackground(), original_background)
        self.assertEqual(backend.mode, "Surface")
        self.assertTrue(backend._axes_enabled)
        self.assertFalse(backend.renderer.GetActiveCamera().GetParallelProjection())
        self.assertEqual(len(attempts), 2)

    def test_same_bounds_replacement_preserves_exact_camera_and_projection(self):
        backend = self.make_backend()
        backend._initialized = True
        mesh = validated_mesh(o3d.geometry.TriangleMesh.create_box())
        backend.set_mesh(mesh)
        backend.navigate_camera("orbit", "Y", 4)
        backend.navigate_camera("translate", "Z", 1)
        before = backend.camera_state()
        refined = validated_mesh(mesh.subdivide_midpoint(number_of_iterations=1))
        with patch.object(
            backend, "set_view", side_effect=AssertionError("Unexpected camera fit")
        ):
            backend.set_mesh(refined, projection="Orthographic")
        self.assertEqual(backend.camera_state(), before)
        self.assertTrue(backend.renderer.GetActiveCamera().GetParallelProjection())

    def test_scaled_translated_mesh_preserves_relative_framing_and_clipping(self):
        for projection in ("Perspective", "Orthographic"):
            with self.subTest(projection=projection):
                backend = self.make_backend()
                backend._initialized = True
                backend.set_mesh(
                    validated_mesh(o3d.geometry.TriangleMesh.create_box()),
                    projection=projection,
                )
                backend.navigate_camera("orbit", "Y", 4)
                backend.navigate_camera("translate", "X", 2)
                backend.navigate_camera("zoom", None, 2)
                before = backend.camera_state()
                old_clipping = np.asarray(
                    backend.renderer.GetActiveCamera().GetClippingRange()
                )
                offset = np.array([10.0, -3.0, 5.0])
                refined = o3d.geometry.TriangleMesh.create_box(
                    width=2, height=2, depth=2
                )
                refined.translate(offset)
                with patch.object(
                    backend,
                    "set_view",
                    side_effect=AssertionError("Unexpected camera fit"),
                ):
                    backend.set_mesh(validated_mesh(refined), projection=projection)
                after = backend.camera_state()
                np.testing.assert_allclose(
                    after["position"], 2 * np.asarray(before["position"]) + offset
                )
                np.testing.assert_allclose(
                    after["focal_point"], 2 * np.asarray(before["focal_point"]) + offset
                )
                self.assertEqual(after["view_up"], before["view_up"])
                self.assertEqual(after["view_angle"], before["view_angle"])
                self.assertAlmostEqual(
                    after["parallel_scale"], 2 * before["parallel_scale"]
                )
                clipping = backend.renderer.GetActiveCamera().GetClippingRange()
                np.testing.assert_allclose(clipping, 2 * old_clipping)
                self.assertGreater(clipping[0], 0)
                self.assertGreater(clipping[1], clipping[0])
                self.assertEqual(
                    bool(backend.renderer.GetActiveCamera().GetParallelProjection()),
                    projection == "Orthographic",
                )

    def test_explicit_recorded_camera_overrides_relative_framing(self):
        backend = self.make_backend()
        backend._initialized = True
        backend.set_mesh(validated_mesh(o3d.geometry.TriangleMesh.create_box()))
        backend.navigate_camera("orbit", "Y", 3)
        recorded = backend.camera_state()
        recorded["position"] = (3 * np.asarray(recorded["position"])).tolist()
        recorded["focal_point"] = (3 * np.asarray(recorded["focal_point"])).tolist()
        recorded["parallel_scale"] = 7.0
        recorded["view_angle"] = 45.0
        larger = validated_mesh(
            o3d.geometry.TriangleMesh.create_box(width=2, height=2, depth=2)
        )
        with patch.object(
            backend,
            "_preserve_relative_framing",
            side_effect=AssertionError("Recorded camera must win"),
        ):
            backend.set_mesh(larger, camera_state=recorded, projection="Orthographic")
        for key, value in recorded.items():
            np.testing.assert_allclose(backend.camera_state()[key], value, atol=1e-12)

    def test_explicit_fit_uses_selected_view_on_replacement(self):
        backend = self.make_backend()
        backend._initialized = True
        mesh = validated_mesh(o3d.geometry.TriangleMesh.create_box())
        with patch.object(backend, "set_view", wraps=backend.set_view) as fit:
            backend.set_mesh(mesh, view="Top")
            fit.assert_called_once_with("Top", render=False)
        backend.navigate_camera("orbit", "Y", 3)
        with patch.object(backend, "set_view", wraps=backend.set_view) as fit:
            backend.set_mesh(mesh, view="Front", preserve_camera=False)
            fit.assert_called_once_with("Front", render=False)

    def test_unusable_bounds_or_camera_transform_retains_whole_camera(self):
        backend = self.make_backend()
        before = backend.camera_state()
        valid = (0, 1, 0, 1, 0, 1)
        cases = [
            (None, valid),
            ((1, -1, 1, -1, 1, -1), valid),
            ((0, 0, 0, 0, 0, 0), valid),
            (valid, (0, float("nan"), 0, 1, 0, 1)),
            (valid, (0, 1e-11, 0, 1e-11, 0, 1e-11)),
            (valid, (0, 1e14, 0, 1e14, 0, 1e14)),
        ]
        for old_bounds, new_bounds in cases:
            with self.subTest(old=old_bounds, new=new_bounds):
                previous, candidate = Mock(), Mock()
                previous.GetBounds.return_value = old_bounds
                candidate.GetBounds.return_value = new_bounds
                backend._preserve_relative_framing(previous, candidate)
                self.assertEqual(backend.camera_state(), before)

    def test_failed_relative_reframe_restores_camera_actor_and_clipping(self):
        backend = self.make_backend()
        backend._initialized = True
        backend.set_mesh(validated_mesh(o3d.geometry.TriangleMesh.create_box()))
        backend.navigate_camera("orbit", "Y", 4)
        backend.navigate_camera("translate", "Z", 1)
        previous = backend.actor
        before = backend.camera_state()
        clipping = backend.renderer.GetActiveCamera().GetClippingRange()
        larger = validated_mesh(
            o3d.geometry.TriangleMesh.create_box(width=2, height=2, depth=2)
        )
        with patch.object(
            backend, "_render", side_effect=[RuntimeError("driver unavailable"), None]
        ):
            with self.assertRaisesRegex(RuntimeError, "driver unavailable"):
                backend.set_mesh(larger, projection="Orthographic")
        self.assertIs(backend.actor, previous)
        self.assertEqual(backend.camera_state(), before)
        self.assertEqual(
            backend.renderer.GetActiveCamera().GetClippingRange(), clipping
        )
        self.assertFalse(backend.renderer.GetActiveCamera().GetParallelProjection())
        backend.set_mesh(larger)
        np.testing.assert_allclose(
            backend.camera_state()["position"], 2 * np.asarray(before["position"])
        )


class TestEmbeddedViewport(_IsolatedLogsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        self.created = []

        def factory(parent):
            backend = FakeRenderer(parent)
            self.created.append(backend)
            return backend

        self.viewport = EmbeddedMeshViewport(renderer_factory=factory)

    def tearDown(self):
        self.viewport.shutdown()
        self.viewport.close()
        self.viewport.deleteLater()

    def test_renderer_is_lazy_and_invalid_mesh_never_initializes_it(self):
        self.assertFalse(self.created)
        with self.assertRaises(ValueError):
            self.viewport.load_mesh(o3d.geometry.TriangleMesh())
        self.assertFalse(self.created)
        self.assertTrue(self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box()))
        self.assertEqual(len(self.created), 1)

    def test_failed_validation_preserves_mesh_labels_and_stats(self):
        self.viewport.load_mesh(
            o3d.geometry.TriangleMesh.create_box(), custom_labels=[1, 2]
        )
        previous = self.viewport.mesh
        stats = self.viewport.stats_label.text()
        with self.assertRaises(FileNotFoundError):
            self.viewport.load_mesh(Path("not-a-real-mesh-file.obj"), custom_labels=[9])
        self.assertIs(self.viewport.mesh, previous)
        self.assertIs(self.created[0].mesh, previous)
        self.assertEqual(self.viewport.custom_labels, [1, 2])
        self.assertEqual(self.viewport.stats_label.text(), stats)

    def test_initial_render_uses_the_chosen_view_and_display_mode(self):
        self.viewport.view_combo.setCurrentText("Top")
        self.viewport.display_combo.setCurrentText("Wireframe")
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        self.assertEqual(self.created[0].view, "Top")
        self.assertEqual(self.created[0].mode, "Wireframe")

    def test_failed_render_preserves_previous_mesh(self):
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        previous = self.viewport.mesh
        self.created[0].fail_next = True
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_sphere())
        self.assertIs(self.viewport.mesh, previous)
        self.assertIs(self.created[0].mesh, previous)

    def test_failed_first_render_is_released_and_can_retry(self):
        def failing_factory(parent):
            backend = FakeRenderer(parent)
            backend.fail_next = True
            self.created.append(backend)
            return backend

        self.viewport._renderer_factory = failing_factory
        with self.assertRaises(RuntimeError):
            self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        self.assertTrue(self.created[0].stopped)
        self.assertIsNone(self.viewport.mesh)
        self.assertIsNone(self.viewport._backend)
        self.assertFalse(self.viewport.placeholder.isHidden())
        self.viewport._renderer_factory = FakeRenderer
        self.assertTrue(self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box()))

    def test_display_controls_do_not_mutate_mesh(self):
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        before = np.asarray(self.viewport.mesh.vertices).copy()
        self.viewport.display_combo.setCurrentText("Wireframe")
        self.viewport.view_combo.setCurrentText("Top")
        self.viewport.view_combo.textActivated.emit("Top")
        self.viewport.axes_checkbox.setChecked(False)
        self.viewport.set_background_color([255, 128, 0])
        backend = self.created[0]
        self.assertEqual(backend.mode, "Wireframe")
        self.assertEqual(backend.view, "Top")
        self.assertFalse(backend.axes)
        self.assertEqual(backend.background, (1.0, 128 / 255.0, 0.0))
        np.testing.assert_array_equal(self.viewport.mesh.vertices, before)

    def test_interactive_render_failure_is_logged_and_reported(self):
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        previous = self.viewport.mesh
        errors = []
        self.viewport.error_occurred.connect(errors.append)
        with patch.object(
            self.created[0], "set_view", side_effect=RuntimeError("driver unavailable")
        ):
            self.viewport.view_combo.textActivated.emit("Top")
        self.assertEqual(
            errors, ["Could not update the mesh preview: driver unavailable"]
        )
        self.logged_errors.exception.assert_called_once()
        self.assertIs(self.viewport.mesh, previous)

    def test_obj_and_stl_export_full_mesh_after_display_changes(self):
        mesh = o3d.geometry.TriangleMesh.create_sphere(resolution=12)
        self.viewport.load_mesh(mesh)
        self.viewport.display_combo.setCurrentText("Wireframe")
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("obj", "stl"):
                with self.subTest(format=suffix):
                    path = Path(directory) / f"mesh.{suffix}"
                    export = getattr(self.viewport, f"export_mesh_as_{suffix}")
                    self.assertTrue(export(path))
                    reloaded = o3d.io.read_triangle_mesh(str(path))
                    self.assertEqual(len(reloaded.triangles), len(mesh.triangles))
                    np.testing.assert_allclose(
                        reloaded.get_min_bound(), mesh.get_min_bound(), atol=1e-5
                    )
                    np.testing.assert_allclose(
                        reloaded.get_max_bound(), mesh.get_max_bound(), atol=1e-5
                    )

    def test_saved_camera_is_retained_before_lazy_renderer_exists(self):
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        self.created[0].camera["position"] = [20.0, -30.0, 40.0]
        self.viewport.display_combo.setCurrentText("Wireframe")
        self.viewport.projection_combo.setCurrentText("Orthographic")
        self.viewport.axes_checkbox.setChecked(False)
        state = json.loads(json.dumps(self.viewport.save_view_state()))
        self.viewport.shutdown()
        changes = []
        self.viewport.view_state_changed.connect(lambda: changes.append(True))
        self.assertTrue(self.viewport.restore_view_state(state))
        self.assertIsNone(self.viewport._backend)
        self.assertEqual(self.viewport.save_view_state(), state)
        self.assertEqual(changes, [])
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_sphere())
        self.assertEqual(self.created[-1].camera, state["camera"])
        self.assertEqual(self.created[-1].projection, "Orthographic")
        self.assertFalse(self.created[-1].axes)
        self.assertEqual(self.viewport.save_view_state(), state)

    def test_view_state_restores_to_existing_backend_without_change_signal(self):
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        state = self.viewport.save_view_state()
        state["camera"]["position"] = [40.0, -60.0, 20.0]
        state["background"] = [0.2, 0.3, 0.4]
        state["projection"] = "Orthographic"
        changes = []
        self.viewport.view_state_changed.connect(lambda: changes.append(True))
        self.assertTrue(self.viewport.restore_view_state(state))
        self.assertEqual(self.viewport.save_view_state(), state)
        self.assertEqual(changes, [])

    def test_invalid_saved_state_does_not_partly_change_controls_or_camera(self):
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        original = self.viewport.save_view_state()
        for key, value in (
            ("position", [0, 0, 0]),
            ("position", [float("nan"), 0, 0]),
            ("view_up", [0, 0, 0]),
            ("view_up", [4, -4, 4]),
            ("view_angle", 180),
            ("parallel_scale", 0),
            ("position", [True, 2, 3]),
        ):
            with self.subTest(key=key, value=value):
                invalid = deepcopy(original)
                invalid["view"] = "Top"
                invalid["camera"][key] = value
                self.assertFalse(self.viewport.restore_view_state(invalid))
                self.assertEqual(self.viewport.save_view_state(), original)
        for invalid in (
            None,
            [],
            {},
            {**original, "version": 999},
            {**original, "axes_visible": "false"},
        ):
            with self.subTest(state=invalid):
                self.assertFalse(self.viewport.restore_view_state(invalid))
                self.assertEqual(self.viewport.save_view_state(), original)
        self.assertTrue(self.logged_errors.exception.called)

    def test_saved_state_is_independent_of_the_source_dictionary(self):
        state = self.viewport.save_view_state()
        self.assertTrue(self.viewport.restore_view_state(state))
        state["background"][0] = 0.9
        self.assertNotEqual(
            self.viewport.save_view_state()["background"], state["background"]
        )
        self.assertEqual(
            validated_view_state(self.viewport.save_view_state()),
            self.viewport.save_view_state(),
        )

    def test_failed_restore_preserves_controls(self):
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        original = self.viewport.save_view_state()
        candidate = deepcopy(original)
        candidate["display_mode"] = "Wireframe"
        with patch.object(
            self.created[0],
            "apply_view_state",
            side_effect=RuntimeError("driver unavailable"),
        ):
            self.assertFalse(self.viewport.restore_view_state(candidate))
        self.assertEqual(self.viewport.save_view_state(), original)

    def test_wheel_filter_routes_fractional_steps_and_consumes_unknown_modifiers(self):
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        changes = []
        self.viewport.view_state_changed.connect(lambda: changes.append(True))
        for modifier, delta in ((Qt.ShiftModifier, 60), (Qt.MetaModifier, 120)):
            event = QWheelEvent(
                QPointF(10, 10),
                QPointF(10, 10),
                QPoint(),
                QPoint(0, delta),
                Qt.NoButton,
                modifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
            QApplication.sendEvent(self.created[0].widget, event)
            self.assertTrue(event.isAccepted())
        self.assertEqual(self.created[0].navigation, [("translate", "X", 0.5)])
        self.assertEqual(changes, [True])

    def test_navigation_buttons_emit_changes_and_preserve_mesh(self):
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        before = np.asarray(self.viewport.mesh.vertices).copy()
        changes = []
        self.viewport.view_state_changed.connect(lambda: changes.append(True))
        button = self.viewport.navigation_buttons[("orbit", "Y", -1)]
        self.assertIn("negative world Y", button.accessibleName())
        button.click()
        self.assertEqual(self.created[0].navigation, [("orbit", "Y", -1)])
        self.assertEqual(changes, [True])
        np.testing.assert_array_equal(self.viewport.mesh.vertices, before)

    def test_keyboard_navigation_is_scoped_to_viewer_focus(self):
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        self.viewport.show()
        self.viewport.activateWindow()
        self.created[0].widget.setFocus()
        QApplication.processEvents()
        QTest.keyClick(self.created[0].widget, Qt.Key_Right, Qt.ControlModifier)
        QTest.keyClick(self.created[0].widget, Qt.Key_Up, Qt.AltModifier)
        QTest.keyClick(self.created[0].widget, Qt.Key_O)
        self.assertEqual(
            self.created[0].navigation, [("translate", "X", 1), ("orbit", "X", 1)]
        )
        self.assertEqual(self.viewport.projection_combo.currentText(), "Orthographic")
        self.assertTrue(
            all(
                shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
                for shortcut in self.viewport._shortcuts
            )
        )

    def test_selector_keeps_native_alt_down_instead_of_orbiting_camera(self):
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        self.viewport.show()
        self.viewport.activateWindow()
        self.viewport.view_combo.setFocus()
        QApplication.processEvents()
        try:
            QTest.keyClick(self.viewport.view_combo, Qt.Key_Down, Qt.AltModifier)
            self.assertEqual(self.created[0].navigation, [])
            self.assertTrue(self.viewport.view_combo.view().isVisible())
        finally:
            self.viewport.view_combo.hidePopup()

    def test_provenance_badge_is_accessible_plain_text_and_not_camera_settings(self):
        self.viewport.set_provenance_status(
            "DA model <b>source</b>", "non-commercial", verified=True
        )
        label = self.viewport.provenance_label
        self.assertIn("NC - Non-commercial use", label.text())
        self.assertIn("Provenance verified", label.text())
        self.assertIn("DA model <b>source</b>", label.text())
        self.assertEqual(label.textFormat(), Qt.TextFormat.PlainText)
        self.assertEqual(label.accessibleName(), label.text())
        self.assertIn("#fff3cd", label.styleSheet())
        self.assertNotIn("provenance", self.viewport.save_view_state())
        self.viewport.set_provenance_status("Custom terms", "custom", verified=True)
        self.assertIn("Custom model license", label.text())
        self.assertIn("#fff3cd", label.styleSheet())
        self.viewport.set_provenance_status("Imported mesh", "unknown", verified=False)
        self.assertIn("license unknown", label.text())
        self.assertIn("Provenance unverified", label.text())
        self.assertNotIn("NC -", label.text())

    def test_export_errors_are_propagated(self):
        with self.assertRaisesRegex(ValueError, "before exporting"):
            self.viewport.export_mesh_as_obj("unused.obj")
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        with patch("open3d.io.write_triangle_mesh", return_value=False):
            with self.assertRaises(OSError):
                self.viewport.export_mesh_as_stl("unused.stl")

    def test_shutdown_releases_backend_without_discarding_geometry(self):
        self.viewport.load_mesh(o3d.geometry.TriangleMesh.create_box())
        previous = self.viewport.mesh
        self.viewport.shutdown()
        self.viewport.shutdown()
        self.assertTrue(self.created[0].stopped)
        self.assertIs(self.viewport.mesh, previous)
        self.assertIsNone(self.viewport._backend)


if __name__ == "__main__":
    unittest.main()
