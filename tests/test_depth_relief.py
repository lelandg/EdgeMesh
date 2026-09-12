"""Geometry regressions through the real depth-to-mesh export pipeline."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from depth_to_3d import DepthTo3D


class DepthReliefTests(unittest.TestCase):
    def test_relative_relief_does_not_flatten_at_larger_resolution(self):
        pipeline = DepthTo3D.__new__(DepthTo3D)
        pipeline.model_type = "depth_anything_v2"
        pipeline.verbose = False
        relief_ratios = []
        with tempfile.TemporaryDirectory() as folder:
            for size in (12, 24):
                image = np.full((size, size, 3), 128, dtype=np.uint8)
                depth = np.broadcast_to(np.linspace(1, 255, size), (size, size)).copy()
                output, _ = pipeline.create_3d_mesh(
                    image, depth, str(Path(folder) / f"ramp-{size}.png"),
                    None, (size, size), True, False, False)
                mesh = trimesh.load_mesh(output)
                relief_ratios.append(mesh.extents[2] / max(mesh.extents[:2]))
        self.assertAlmostEqual(relief_ratios[0], relief_ratios[1], places=6)
        self.assertAlmostEqual(relief_ratios[0], 0.5, places=6)

    def test_relief_scales_with_amount_and_longest_rectangular_side(self):
        pipeline = DepthTo3D.__new__(DepthTo3D)
        pipeline.model_type = "depth_anything_v2"
        pipeline.verbose = False
        with tempfile.TemporaryDirectory() as folder:
            for shape in ((8, 15), (15, 8)):
                for amount in (0.0, 0.5, 2.0):
                    image = np.full((*shape, 3), 128, dtype=np.uint8)
                    depth = np.broadcast_to(np.linspace(1, 255, shape[1]), shape).copy()
                    output, _ = pipeline.create_3d_mesh(
                        image, depth, str(Path(folder) / f"rectangle-{shape}-{amount}.png"),
                        None, shape, True, False, False, depth_amount=amount)
                    mesh = trimesh.load_mesh(output)
                    ratio = mesh.extents[2] / max(mesh.extents[:2])
                    self.assertAlmostEqual(ratio, 0.5 * amount, places=6)
                    if amount:
                        self.assertTrue(mesh.is_watertight)


if __name__ == "__main__":
    unittest.main()
