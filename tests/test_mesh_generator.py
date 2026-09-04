"""Actual contour extrusion geometry, without model downloads or GUI windows."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np

from depth_based3d_reconstruction import ExtrusionProjectionReconstruction
from mesh_generator import MeshGenerator


def contour(offset=0):
    return np.array([[[offset, 0]], [[offset + 1, 0]],
                     [[offset + 1, 1]], [[offset, 1]]], dtype=np.int32)


class MeshGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.generator = MeshGenerator({'visualize_partitioning': False})

    def test_closed_contour_has_every_complete_wall(self):
        shapes = ExtrusionProjectionReconstruction([contour()], depth=2).extrude()
        mesh = self.generator.mesh_from_shapes(shapes)
        self.assertEqual(len(mesh.triangles), 8)
        self.assertAlmostEqual(mesh.get_surface_area(), 8)
        triangles = np.asarray(mesh.triangles)
        for start in range(4):
            wall = {2 * start, 2 * start + 1,
                    2 * ((start + 1) % 4), 2 * ((start + 1) % 4) + 1}
            self.assertEqual(sum(set(face).issubset(wall) for face in triangles), 2)

    def test_separate_contours_are_not_joined(self):
        shapes = ExtrusionProjectionReconstruction([contour(), contour(5)]).extrude()
        mesh = self.generator.mesh_from_shapes(shapes)
        self.assertEqual(len(mesh.triangles), 16)
        for face in np.asarray(mesh.triangles):
            self.assertTrue(np.all(face < 8) or np.all(face >= 8))

    def test_two_point_contour_is_one_quad(self):
        line = np.array([[[0, 0]], [[1, 0]]], dtype=np.int32)
        shapes = ExtrusionProjectionReconstruction([line], depth=2).extrude()
        mesh = self.generator.mesh_from_shapes(shapes)
        self.assertEqual(len(mesh.triangles), 2)
        self.assertAlmostEqual(mesh.get_surface_area(), 2)

    def test_projected_apex_does_not_create_zero_area_faces(self):
        shapes = ExtrusionProjectionReconstruction(
            [contour()], depth=2, vanishing_point=(0.5, 0.5)).project()
        mesh = self.generator.mesh_from_shapes(shapes)
        self.assertEqual(len(mesh.triangles), 4)
        vertices = np.asarray(mesh.vertices)
        for face in np.asarray(mesh.triangles):
            a, b, c = vertices[face]
            self.assertGreater(np.linalg.norm(np.cross(b - a, c - a)), 0)

    def test_empty_and_single_point_have_no_faces(self):
        for contours in ([], [np.array([[[0, 0]]])]):
            with self.subTest(contours=contours):
                mesh = self.generator.mesh_from_shapes(
                    ExtrusionProjectionReconstruction(contours).extrude())
                self.assertEqual(len(mesh.triangles), 0)

    def test_failed_mesh_write_is_logged_and_raised(self):
        with ExitStack() as stack:
            clustering = stack.enter_context(patch('mesh_generator.EdgeClustering'))
            clustering.return_value.analyze_edges.return_value = ({}, [contour()], [])
            for name in ('ShapeAnalysis', 'DepthCueEstimator', 'SurfacePartitioning'):
                stack.enter_context(patch('mesh_generator.' + name))
            stack.enter_context(patch('mesh_generator.o3d.io.write_triangle_mesh', return_value=False))
            printed = stack.enter_context(patch('builtins.print'))
            with self.assertRaisesRegex(OSError, 'Could not write mesh'), self.assertLogs('mesh_generator', level='ERROR'):
                self.generator.generate(np.zeros((8, 8, 3), dtype=np.uint8), 'source.png')
            self.assertFalse(any('Saved mesh' in str(call) for call in printed.call_args_list))


if __name__ == '__main__':
    unittest.main()
