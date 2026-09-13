"""Offline checks for the isolated DA3 runtime and image contract."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import da3_backend
import da3_worker
from model_store import ModelPreparationCancelled, ModelSetupError


class SnapshotTests(unittest.TestCase):
    def test_offline_snapshot_records_and_verifies_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text('{}')
            (snapshot / "model.safetensors").write_bytes(b"checkpoint")
            hub = types.ModuleType("huggingface_hub")
            calls = []
            hub.HfApi = lambda: self.fail("Offline preparation contacted the Hub")
            hub.try_to_load_from_cache = lambda *a, **kw: None
            def download(*args, **kwargs):
                calls.append(kwargs)
                return str(snapshot)
            hub.snapshot_download = download
            request = {"root": directory, "model_id": "depth-anything/DA3-SMALL",
                       "revision": "a" * 40, "allow_download": False}
            with patch.dict(sys.modules, huggingface_hub=hub):
                _, manifest, metadata = da3_worker.prepare_snapshot(request)
                da3_worker.write_json(manifest, metadata)
                self.assertEqual(calls[0]["revision"], "a" * 40)
                self.assertTrue(calls[0]["local_files_only"])
                self.assertEqual(set(metadata["weight_files_sha256"]),
                                 {"config.json", "model.safetensors"})
                (snapshot / "model.safetensors").write_bytes(b"modified")
                with self.assertRaisesRegex(ValueError, "recorded hashes"):
                    da3_worker.prepare_snapshot(request)

    def test_offline_without_revision_does_not_query_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = types.ModuleType("huggingface_hub")
            hub.HfApi = lambda: self.fail("Offline preparation contacted the Hub")
            hub.try_to_load_from_cache = lambda *a, **kw: None
            hub.snapshot_download = lambda *a, **kw: self.fail("Missing pin used")
            with patch.dict(sys.modules, huggingface_hub=hub):
                with self.assertRaisesRegex(RuntimeError, "not cached"):
                    da3_worker.prepare_snapshot({"root": directory,
                                                 "model_id": "depth-anything/DA3-SMALL"})


class AdapterTests(unittest.TestCase):
    def test_missing_runtime_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, EDGEMESH_DA3_PYTHON=""):
            with self.assertRaisesRegex(ModelSetupError, "separate Python runtime"):
                da3_backend.load_da3("depth_anything_3_small", "cpu", root=directory)

    def test_cancel_before_start_does_not_launch(self):
        with patch("da3_backend.subprocess.Popen") as launch:
            with self.assertRaises(ModelPreparationCancelled):
                da3_backend.load_da3("depth_anything_3_small", "cpu", root="unused", cancelled=lambda: True)
            launch.assert_not_called()

    def test_inference_uses_prepared_revision_and_forbids_download(self):
        depth = np.array([[1, 3]], dtype=np.float32)
        request = {"revision": "b" * 40, "allow_download": True}
        model = da3_backend.DA3DepthModel(request, ".")
        with patch("da3_backend._run", return_value=({}, depth)) as run:
            actual = model.predict_depth(np.zeros((1, 2, 3), dtype=np.uint8))
        np.testing.assert_array_equal(actual, depth)
        self.assertFalse(run.call_args.args[0]["allow_download"])
        self.assertEqual(run.call_args.args[0]["revision"], "b" * 40)

    def test_cancellation_terminates_worker_and_removes_temporary_images(self):
        with tempfile.TemporaryDirectory() as directory:
            checks = iter([False, True])
            with patch.dict(os.environ, EDGEMESH_DA3_PYTHON=sys.executable), \
                 patch("da3_backend.subprocess.Popen") as launch:
                process = launch.return_value
                process.poll.return_value = None
                with self.assertRaises(ModelPreparationCancelled):
                    da3_backend._run({"action": "infer"}, directory,
                                     lambda: next(checks), np.zeros((2, 3, 3), dtype=np.uint8))
                process.terminate.assert_called_once()
                self.assertFalse(list((Path(directory) / "work").iterdir()))

    def test_portrait_landscape_and_singleton_images_are_padded_without_crop(self):
        for shape in ((45, 99), (99, 45), (1, 99), (99, 1)):
            image = np.full((*shape, 3), [12, 89, 234], dtype=np.uint8)
            padded, (height, width) = da3_worker.padded_image(image)
            self.assertEqual(padded.shape[0] % 14, 0)
            self.assertEqual(padded.shape[1] % 14, 0)
            self.assertEqual(max(padded.shape[:2]), 504)
            self.assertLess(padded.shape[0] - height, 14)
            self.assertLess(padded.shape[1] - width, 14)
            np.testing.assert_array_equal(padded[0, 0], image[0, 0])
            np.testing.assert_array_equal(padded[-1, -1], image[0, 0])


if __name__ == "__main__":
    unittest.main()
