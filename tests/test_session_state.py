import base64
import json
from pathlib import Path
import tempfile
import unittest
import zlib

import numpy as np

from session_state import (SessionDocument, SessionHistory, load_preset, load_session,
                           save_preset, save_session, validate_settings)


class SessionTests(unittest.TestCase):
    def test_roundtrip_missing_source_mask_and_model_identity(self):
        mask = np.array([[True, False, True], [False, True, False]])
        doc = SessionDocument("missing/source.png", {"resolution": 700, "depth_amount": 1.0},
                              {"revision": "recorded-revision"}, mask, [{"action": "accepted mask"}])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.json"
            save_session(path, doc)
            loaded = load_session(path)
        self.assertEqual(loaded.source_path, doc.source_path)
        self.assertEqual(loaded.settings, doc.settings)
        self.assertEqual(loaded.model_info, doc.model_info)
        np.testing.assert_array_equal(loaded.mask, mask)
        self.assertEqual(loaded.history, doc.history)

    def test_history_is_bounded_deep_copied_and_branches_after_undo(self):
        history = SessionHistory(limit=3)
        doc = SessionDocument("image.png", {"resolution": 100}, mask=np.ones((2, 2), bool))
        history.record(doc)
        doc.settings["resolution"] = 200
        doc.mask[0, 0] = False
        self.assertEqual(history.current.settings["resolution"], 100)
        self.assertTrue(history.current.mask[0, 0])
        history.record(doc)
        self.assertEqual(history.undo().settings["resolution"], 100)
        self.assertEqual(history.redo().settings["resolution"], 200)
        history.undo()
        history.record(SessionDocument("image.png", {"resolution": 300}))
        self.assertFalse(history.can_redo)
        for value in (400, 500, 600):
            history.record(SessionDocument("image.png", {"resolution": value}))
        self.assertEqual(history.undo().settings["resolution"], 500)
        self.assertEqual(history.undo().settings["resolution"], 400)
        self.assertIsNone(history.undo())
        # Only metadata and compressed strings are retained, never numpy arrays.
        json.dumps(history._snapshots)

    def test_presets_never_include_source_or_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "preset.json"
            save_preset(path, {"depth_amount": 0.7, "grayscale_enabled": True})
            self.assertEqual(load_preset(path), {"depth_amount": 0.7, "grayscale_enabled": True})
            self.assertEqual(set(json.loads(path.read_text())), {"format", "version", "settings"})

    def test_invalid_settings_and_nonfinite_metadata_rejected(self):
        for settings in ({"resolution": -1}, {"depth_amount": float("nan")},
                         {"grayscale_enabled": "true"}, {"unexpected": 1}, {"line_thickness": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_settings(settings)
        with self.assertRaises(ValueError):
            SessionHistory().record(SessionDocument("x", {}, {"x": np.ones(10)}))

    def test_ui_supported_values_are_normalized_and_unknown_model_rejected(self):
        self.assertEqual(validate_settings({"smoothing_method": "none", "depth_amount": 0}),
                         {"smoothing_method": "(none)", "depth_amount": 0})
        self.assertEqual(validate_settings({"model": "DPT"}), {"model": "DPT"})
        with self.assertRaises(ValueError):
            validate_settings({"model": "unrecognized-model"})

    def test_old_schema_migration_and_future_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.json"
            payload = {"format": "edgemesh-session", "version": 0, "source_path": "old.png", "settings": {}}
            path.write_text(json.dumps(payload))
            self.assertEqual(load_session(path).model_info, {})
            payload["version"] = 9000
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_session(path)

    def test_decompression_size_mismatch_rejected(self):
        payload = {"format": "edgemesh-session", "version": 1, "source_path": "x", "settings": {},
            "mask": {"shape": [1, 1], "encoding": "zlib-packbits-little",
                "data": base64.b64encode(zlib.compress(b"x" * 100_000)).decode()}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_session(path)


if __name__ == "__main__":
    unittest.main()
