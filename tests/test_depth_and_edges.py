"""Regression tests using real image/tensor operations without model downloads."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
import trimesh
from PySide6.QtGui import QColor

from depth_to_3d import DepthTo3D
from edge_detection import detect_edges


class EdgeTests(unittest.TestCase):
    def test_grayscale_works_on_canvas_and_overlay(self):
        gray = np.zeros((8, 8), dtype=np.uint8)
        gray[:, 4:] = 255
        for overlay in (False, True):
            with self.subTest(overlay=overlay):
                result = detect_edges(gray, 50, 150, project_on_original=overlay)
                self.assertEqual(result.shape, (8, 8, 3))
                self.assertGreater(np.count_nonzero(result), 0)

    def test_bgr_channels_use_opencv_luminance(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, 4:] = (255, 0, 0)
        expected = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)
        result = detect_edges(image, 100, 200)
        np.testing.assert_array_equal(result[:, :, 0], expected)


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = DepthTo3D.__new__(DepthTo3D)
        self.pipeline.verbose = False
        self.pipeline.device = torch.device('cpu')

    def test_background_tolerance_does_not_wrap_black_or_white(self):
        for value in (0, 255):
            image = np.full((5, 5, 3), value, np.uint8)
            result = self.pipeline.create_background_mask(
                image, [value] * 3, True, 10)
            np.testing.assert_array_equal(result, 255)

    def test_qcolor_and_rgb_tuple_match_bgr_pixels(self):
        image = np.full((5, 5, 3), (0, 0, 255), np.uint8)
        for color in (QColor(255, 0, 0), [255, 0, 0]):
            result = self.pipeline.create_background_mask(image, color, True, 0)
            np.testing.assert_array_equal(result, 255)

    def test_background_mask_preserves_enclosed_foreground(self):
        image = np.full((7, 7, 3), 128, np.uint8)
        image[2:5, 2:5] = 0
        mask = self.pipeline.create_background_mask(image, [128] * 3, True, 0)
        self.assertEqual(mask[0, 0], 255)
        self.assertEqual(mask[3, 3], 0)

    def test_exact_automatic_background_and_isolated_pixel(self):
        image = np.full((7, 7, 3), 128, np.uint8)
        image[1:6, 1:6] = 0
        image[3, 3] = 128
        mask = self.pipeline.create_background_mask(image, background_removal=True)
        self.assertEqual(mask[0, 0], 255)
        self.assertEqual(mask[3, 3], 0)

    def test_missing_image_has_actionable_error(self):
        self.pipeline.model_type = 'midas'
        with patch('depth_to_3d.cv2.imread', return_value=None):
            with self.assertRaisesRegex(ValueError, 'Image not found'):
                self.pipeline.process_image('missing.png')

    def test_dpt_and_midas_restore_original_dimensions(self):
        for model_type in ('dpt', 'midas'):
            with self.subTest(model_type=model_type):
                self.pipeline.model_type = model_type
                self.pipeline.transform = Mock(return_value=torch.zeros(1, 3, 32, 32))
                self.pipeline.model = Mock(return_value=torch.arange(6.).reshape(1, 2, 3))
                result = self.pipeline.estimate_depth(np.zeros((7, 11, 3), np.uint8), (0, 0))
                self.assertEqual(result.shape, (7, 11))

    def test_hf_models_use_rgb_flip_and_requested_shape(self):
        image = np.zeros((7, 11, 3), np.uint8)
        image[:, :3] = (5, 10, 200)
        for model_type in ('depth_anything_v2', 'depth_pro'):
            with self.subTest(model_type=model_type):
                self.pipeline.model_type = model_type
                self.pipeline.transform = Mock(return_value={'pixel_values': torch.zeros(1, 3, 7, 11)})
                self.pipeline.model = Mock(return_value=SimpleNamespace(predicted_depth=torch.arange(6.).reshape(1, 2, 3)))
                with patch('depth_to_3d.AutoImageProcessor.from_pretrained') as loader:
                    loader.return_value = self.pipeline.transform
                    result = self.pipeline.estimate_depth(image, (5, 9), flip=True)
                actual = np.asarray(self.pipeline.transform.call_args.kwargs['images'])
                np.testing.assert_array_equal(actual, cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB))
                self.assertEqual(result.shape, (5, 9))
                loader.assert_not_called()

    def test_depth_loading_delegates_identity_download_and_cancellation_to_store(self):
        for model_type in ('depth_anything_v2', 'depth_pro'):
            self.pipeline.model_type = model_type
            model, processor = Mock(), Mock()
            store = Mock()
            metadata = {'revision': 'a' * 40, 'backend': 'huggingface'}
            store.get_depth.return_value = model, processor, metadata
            self.pipeline.model_store = store
            self.pipeline.allow_download = False
            self.pipeline.cancelled = Mock(return_value=False)
            actual_model, actual_processor = self.pipeline.load_model()
            store.get_depth.assert_called_once_with(
                model_type, self.pipeline.device, allow_download=False,
                cancelled=self.pipeline.cancelled)
            self.assertIs(actual_model, model)
            self.assertIs(actual_processor, processor)
            self.assertEqual(self.pipeline.model_info, metadata)

    def test_prediction_flip_is_undone_for_color_alignment(self):
        self.pipeline.model_type = 'midas'
        self.pipeline.transform = Mock(return_value=torch.zeros(1, 3, 32, 32))
        self.pipeline.model = Mock(return_value=torch.tensor([[[0., 1., 2.], [0., 1., 2.]]]))
        result = self.pipeline.estimate_depth(np.zeros((2, 3, 3), np.uint8), (2, 3), flip=True)
        np.testing.assert_array_equal(result, [[255, 127.5, 0], [255, 127.5, 0]])

    def test_real_mesh_pipeline_exports_watertight_flat_and_mirror(self):
        self.pipeline.model_type = 'midas'
        image = np.full((3, 4, 3), (0, 0, 255), np.uint8)
        depth = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
        with tempfile.TemporaryDirectory() as folder:
            for flat in (False, True):
                path, background = self.pipeline.create_3d_mesh(
                    image, depth.copy(), str(Path(folder) / 'input.png'),
                    'gaussian', (3, 4), flat, False, False)
                mesh = trimesh.load_mesh(path)
                self.assertTrue(mesh.is_watertight)
                self.assertTrue(mesh.is_winding_consistent)
                np.testing.assert_array_equal(mesh.visual.vertex_colors[:, :3], np.tile([255, 0, 0], (len(mesh.vertices), 1)))


if __name__ == '__main__':
    unittest.main()
