"""DA3 catalog, pipeline and persistence contracts without model downloads."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from da3_backend import DA3_ALIASES, DA3_MODELS
from model_licensing import policy_for
from model_store import HF_MODELS, ModelPreparationCancelled, ModelStore
from session_state import SessionDocument, load_session, save_session, validate_settings


class DA3IntegrationTests(unittest.TestCase):
    def test_all_variants_roundtrip_and_resolve_permissive_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            for display_name, key in DA3_ALIASES.items():
                with self.subTest(model=key):
                    settings = validate_settings({'model': display_name})
                    path = Path(directory) / 'session.json'
                    save_session(path, SessionDocument('image.png', settings))
                    self.assertEqual(load_session(path).settings, settings)
                    policy = policy_for(display_name)
                    self.assertEqual(policy.model_id, DA3_MODELS[key][0])
                    self.assertEqual(policy.license, 'apache-2.0')
                    self.assertFalse(policy.requires_acceptance)
                    self.assertNotIn(key, HF_MODELS)

    def test_store_dispatches_all_aliases_without_transformers_or_proxy_cache(self):
        cancelled = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as directory:
            store = ModelStore(directory, cache_dir=Path(directory) / 'cache')
            for display_name, key in DA3_ALIASES.items():
                metadata = {'backend': 'depth_anything_3', 'model_type': key}
                pair = (Mock(), None, metadata)
                with self.subTest(model=key), patch('da3_backend.load_da3', return_value=pair) as loader, patch.object(store, '_get_hf') as hf:
                    self.assertEqual(store.get_depth(display_name, 'cpu', True, cancelled=cancelled), pair)
                    loader.assert_called_once_with(key, 'cpu', root=Path(directory), cache_dir=str(Path(directory) / 'cache'), allow_download=True, cancelled=cancelled)
                    hf.assert_not_called()
                    self.assertFalse(store.consent_request(display_name)['consent_needed'])
            self.assertEqual(store._pairs, {})

    def test_store_cancellation_prevents_da3_preparation(self):
        with tempfile.TemporaryDirectory() as directory, patch('da3_backend.load_da3') as loader:
            with self.assertRaises(ModelPreparationCancelled):
                ModelStore(directory).get_depth('Depth Anything 3 Small', 'cpu', cancelled=lambda: True)
            loader.assert_not_called()


class DA3DepthPipelineTests(unittest.TestCase):
    def setUp(self):
        from depth_to_3d import DepthTo3D
        self.pipeline = DepthTo3D.__new__(DepthTo3D)
        self.pipeline.model_type = 'depth_anything_3_small'
        self.pipeline.model_info = {'backend': 'depth_anything_3'}
        self.pipeline.cancelled = Mock(return_value=False)
        self.pipeline.model = Mock()

    def test_rgb_processing_flip_is_undone_and_depth_normalized(self):
        bgr = np.array([[[1, 2, 30], [4, 5, 60], [7, 8, 90]]], dtype=np.uint8)
        self.pipeline.model.predict_depth.return_value = np.array([[1, 2, 4]], dtype=np.float32)
        depth = self.pipeline.estimate_depth(bgr, (0, 0), flip=True)
        call = self.pipeline.model.predict_depth.call_args
        np.testing.assert_array_equal(call.args[0], bgr[:, ::-1, ::-1])
        self.assertIs(call.kwargs['cancelled'], self.pipeline.cancelled)
        np.testing.assert_allclose(depth, [[0, 85, 255]], atol=1e-5)
        self.assertEqual(depth.dtype, np.float32)

    def test_shape_constant_depth_and_nonfinite_contract(self):
        self.pipeline.model.predict_depth.return_value = np.ones((1, 2), np.float32)
        result = self.pipeline.estimate_depth(np.zeros((3, 5, 3), np.uint8), (6, 10))
        self.assertEqual(result.shape, (6, 10))
        np.testing.assert_array_equal(result, 0)
        self.pipeline.model.predict_depth.return_value = np.array([[np.nan]], np.float32)
        with self.assertRaises(ValueError):
            self.pipeline.estimate_depth(np.zeros((3, 5, 3), np.uint8))

    def test_pipeline_model_names_include_every_variant(self):
        from depth_to_3d import model_names
        for label, key in DA3_ALIASES.items():
            self.assertEqual(model_names[label], key)
