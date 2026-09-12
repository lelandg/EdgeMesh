import unittest
import numpy as np
from data_contracts import as_bgr, foreground_mask, normalized_depth, output_shape, proportional_shape


class ContractTests(unittest.TestCase):
    def test_gray_and_transparent_images(self):
        self.assertEqual(as_bgr(np.zeros((2, 3), np.uint8)).shape, (2, 3, 3))
        np.testing.assert_array_equal(as_bgr(np.zeros((2, 3, 4), np.uint8)), 255)

    def test_nonfinite_and_empty_depth_rejected(self):
        for value in (np.array([[np.nan]]), np.array([[np.inf]]), np.zeros((0, 2))):
            with self.assertRaises(ValueError):
                normalized_depth(value)

    def test_constant_singleton_depth_and_proportions(self):
        np.testing.assert_array_equal(normalized_depth([[7]], (1, 4)), 0)
        self.assertEqual(proportional_shape((1, 100), 2), (1, 2))
        self.assertEqual(output_shape((7, 9, 3), (0, 0)), (7, 9))
        with self.assertRaises(ValueError):
            output_shape((2, 2), (0, 2))

    def test_masks_remain_binary_and_do_not_mutate(self):
        original = np.array([[0, 255]], np.uint8)
        result = foreground_mask(original, (2, 4))
        self.assertEqual(result.dtype, bool)
        np.testing.assert_array_equal(result, [[False, False, True, True]] * 2)
        with self.assertRaises(ValueError):
            foreground_mask([[0, 0.2]], (1, 2))
