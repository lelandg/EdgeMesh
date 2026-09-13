"""Distance predictions must raise near surfaces without moving their colors."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import trimesh

from da3_backend import DA3_MODELS
from depth_to_3d import DepthTo3D


class DA3ReliefDirectionTests(unittest.TestCase):
    def test_near_feature_is_raised_and_rgb_stays_at_its_source_pixel(self):
        # A blue near object inside a red distant background. One farther pixel
        # retains a nonzero background height, exercising real mesh generation.
        image = np.full((5, 5, 3), (0, 0, 255), np.uint8)
        image[1:4, 1:4] = (255, 0, 0)
        distance = np.full((5, 5), 4, np.float32)
        distance[1:4, 1:4] = 1
        distance[0, 0] = 5
        for model_type in DA3_MODELS:
            for flip in (False, True):
                with self.subTest(model=model_type, flip=flip), tempfile.TemporaryDirectory() as directory:
                    pipeline = DepthTo3D.__new__(DepthTo3D)
                    pipeline.model_type = model_type
                    pipeline.model_info = {'backend': 'depth_anything_3'}
                    pipeline.cancelled = None
                    pipeline.verbose = False
                    pipeline.model = Mock()
                    pipeline.model.predict_depth.return_value = np.fliplr(distance) if flip else distance.copy()
                    depth = pipeline.estimate_depth(image, image.shape[:2], flip=flip)
                    self.assertGreater(depth[2, 2], depth[0, 2],
                                       'Near DA3 features must rise above the distant background.')
                    path, _ = pipeline.create_3d_mesh(
                        image, depth, str(Path(directory) / 'relief.png'), None,
                        image.shape[:2], False, False, False,
                    )
                    mesh = trimesh.load(path, process=False)
                    vertices = np.asarray(mesh.vertices)
                    x = np.rint(vertices[:, 0]).astype(int)
                    y = np.rint(-vertices[:, 1]).astype(int)
                    expected_rgb = image[y, x, ::-1]
                    np.testing.assert_array_equal(mesh.visual.vertex_colors[:, :3], expected_rgb)
                    near_z = vertices[(x == 2) & (y == 2), 2].max()
                    far_z = vertices[(x == 2) & (y == 0), 2].max()
                    self.assertGreater(near_z, far_z)


if __name__ == '__main__':
    unittest.main()
