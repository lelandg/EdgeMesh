import unittest
import warnings

import numpy as np

from smoothing_depth_map_utils import SmoothingDepthMapUtils


class SmoothingTests(unittest.TestCase):
    def test_integer_depth_diffusion_stays_finite_and_preserves_input(self):
        depth = np.array([[0, 20], [40, 80]], dtype=np.uint8)
        original = depth.copy()
        output = SmoothingDepthMapUtils.anisotropic_diffusion(depth, iterations=2)
        self.assertTrue(np.isfinite(output).all())
        self.assertGreaterEqual(output.min(), original.min())
        self.assertLessEqual(output.max(), original.max())
        np.testing.assert_array_equal(depth, original)

    def test_diffusion_does_not_wrap_opposite_borders(self):
        depth = np.zeros((4, 4), dtype=np.float32)
        depth[0] = 1
        output = SmoothingDepthMapUtils.anisotropic_diffusion(depth, iterations=1)
        np.testing.assert_array_equal(output[-1], np.zeros(4))
        self.assertGreater(output[1, 0], 0)

    def test_constant_normalization_returns_lower_bound_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            output = SmoothingDepthMapUtils.normalize_depth_map(np.full((2, 2), 5), (20, 100))
        np.testing.assert_array_equal(output, np.full((2, 2), 20, dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
