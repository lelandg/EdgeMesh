import tempfile
from pathlib import Path
import unittest

import numpy as np
import open3d as o3d
import trimesh

from mesh_health import inspect_mesh, repair_preview


class MeshHealthTests(unittest.TestCase):
    def test_closed_mesh_and_json_scalars(self):
        report = inspect_mesh(trimesh.creation.box())
        self.assertTrue(report.is_watertight)
        self.assertTrue(report.is_winding_consistent)
        self.assertEqual(report.boundary_edge_count, 0)
        self.assertEqual(report.face_count, 12)
        import json
        self.assertEqual(json.loads(json.dumps(report.to_dict()))['vertex_count'], 8)

    def test_open_hole_not_filled_by_preview(self):
        mesh = trimesh.creation.box()
        mesh.update_faces(np.arange(len(mesh.faces)) != 0)
        before = mesh.faces.copy()
        report = inspect_mesh(mesh)
        self.assertEqual(report.boundary_edge_count, 3)
        self.assertFalse(report.is_watertight)
        preview = repair_preview(mesh)
        self.assertIsNot(preview, mesh)
        self.assertEqual(inspect_mesh(preview).boundary_edge_count, 3)
        np.testing.assert_array_equal(mesh.faces, before)

    def test_cleanup_invalid_degenerate_duplicates_and_colors(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [np.nan, 0, 0], [9, 9, 9]]
        faces = [[0, 1, 2], [2, 1, 0], [0, 0, 1], [0, 1, 3], [0, 1, 99]]
        colors = [[255, 0, 0, 255]] * 5
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
        report = inspect_mesh(mesh)
        self.assertEqual(report.nonfinite_vertex_count, 1)
        self.assertEqual(report.invalid_face_count, 2)
        self.assertEqual(report.degenerate_face_count, 1)
        self.assertEqual(report.duplicate_face_count, 1)
        repaired = repair_preview(mesh)
        self.assertEqual(len(repaired.vertices), 3)
        self.assertEqual(len(repaired.faces), 1)
        self.assertEqual(len(mesh.faces), 5)
        np.testing.assert_array_equal(repaired.visual.vertex_colors, colors[:3])

    def test_winding_and_nonmanifold_are_reported(self):
        mesh = trimesh.creation.box()
        mesh.faces[0] = mesh.faces[0][::-1]
        self.assertFalse(inspect_mesh(mesh).is_winding_consistent)
        mesh.faces = np.vstack([mesh.faces, mesh.faces[0]])
        self.assertEqual(inspect_mesh(mesh).nonmanifold_edge_count, 3)

    def test_open3d_type_preserved_and_source_unmodified(self):
        mesh = o3d.geometry.TriangleMesh.create_box()
        mesh.paint_uniform_color([0.1, 0.2, 0.3])
        mesh.triangles = o3d.utility.Vector3iVector(np.vstack([mesh.triangles, np.asarray(mesh.triangles)[0]]))
        repaired = repair_preview(mesh)
        self.assertIsInstance(repaired, o3d.geometry.TriangleMesh)
        self.assertEqual(len(mesh.triangles), 13)
        self.assertEqual(len(repaired.triangles), 12)
        np.testing.assert_allclose(repaired.vertex_colors, mesh.vertex_colors)
        self.assertTrue(inspect_mesh(repaired).is_watertight)

    def test_texture_coordinates_and_face_material_mapping_survive_cleanup(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9]],
                               faces=[[0, 1, 2], [0, 1, 2]], process=False)
        uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, face_materials=[0, 0])
        preview = repair_preview(mesh)
        np.testing.assert_array_equal(preview.visual.uv, uv[:3])
        np.testing.assert_array_equal(preview.visual.face_materials, [0])
        np.testing.assert_array_equal(mesh.visual.face_materials, [0, 0])

    def test_path_input_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'box.ply'
            trimesh.creation.box().export(path)
            self.assertTrue(inspect_mesh(path).is_watertight)
            self.assertIsInstance(repair_preview(path), trimesh.Trimesh)
            with self.assertLogs('mesh_health', level='ERROR'), self.assertRaises(ValueError):
                inspect_mesh(Path(directory) / 'absent.ply')

    def test_empty_mesh(self):
        mesh = trimesh.Trimesh()
        report = inspect_mesh(mesh)
        self.assertFalse(report.is_watertight)
        self.assertEqual(report.finite_vertex_count, 0)
        self.assertEqual(len(repair_preview(mesh).faces), 0)


if __name__ == '__main__':
    unittest.main()
