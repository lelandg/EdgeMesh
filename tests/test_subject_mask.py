"""Original-size mask, prompt, acceptance, and cooperative worker contracts."""

import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtTest import QTest

from model_store import ModelPreparationCancelled
from ui_persistence import SettingsStore
from subject_mask import (
    MaskCanvas,
    SubjectMaskDialog,
    cancel_active_jobs,
    has_active_jobs,
    image_rect,
    infer_subject_mask,
    paint_mask,
    source_point,
    validated_mask,
)


class SubjectMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.image = np.zeros((40, 80, 3), dtype=np.uint8)
        self.image[:, :, 0] = 200
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        preferences = SettingsStore(Path(temporary.name) / "ui-settings.json")
        settings_patch = patch(
            "subject_mask._preference_store", return_value=preferences
        )
        settings_patch.start()
        self.addCleanup(settings_patch.stop)

    def test_non_square_preview_preserves_proportion_and_rejects_letterbox(self):
        self.assertEqual(image_rect(self.image.shape, 400, 400), (0, 100, 400, 200))
        self.assertEqual(
            source_point((200, 200), self.image.shape, (400, 400)), (40, 20)
        )
        self.assertEqual(
            source_point((399, 299), self.image.shape, (400, 400)), (79, 39)
        )
        self.assertIsNone(source_point((20, 20), self.image.shape, (400, 400)))
        self.assertIsNone(source_point((400, 200), self.image.shape, (400, 400)))
        self.assertEqual(source_point((0, 0), (1, 1, 3), (1, 1)), (0, 0))

    def test_mask_validation_rejects_shape_and_nonfinite_values_and_copies(self):
        mask = np.ones((40, 80), dtype=bool)
        copy = validated_mask(mask, self.image.shape)
        copy[0, 0] = False
        self.assertTrue(mask[0, 0])
        with self.assertRaises(ValueError):
            validated_mask(np.zeros((80, 40)), self.image.shape)
        with self.assertRaises(ValueError):
            validated_mask(np.full((40, 80), np.nan), self.image.shape)

    def test_brush_refinement_is_clipped_additive_and_reversible(self):
        mask = np.zeros((40, 80), dtype=bool)
        paint_mask(mask, (0, 0), 2, True)
        self.assertTrue(mask[0, 0])
        self.assertTrue(mask[0, 2])
        self.assertFalse(mask[2, 2])
        self.assertFalse(mask[-1, -1])
        paint_mask(mask, (0, 0), 2, False)
        self.assertFalse(mask.any())

    def test_accept_returns_copy_and_never_changes_source_or_initial_mask(self):
        initial = np.ones((40, 80), dtype=bool)
        original = self.image.copy()
        dialog = SubjectMaskDialog(self.image, Mock(), initial_mask=initial)
        paint_mask(dialog.canvas.mask, (3, 3), 2, False)
        self.assertTrue(initial.all())
        self.assertIsNone(dialog.accepted_mask)
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(dialog.accepted_mask.shape, (40, 80))
        self.assertEqual(dialog.accepted_mask.dtype, np.bool_)
        dialog.canvas.mask[:] = False
        self.assertTrue(dialog.accepted_mask.any())
        np.testing.assert_array_equal(self.image, original)
        dialog.deleteLater()

    def test_reject_does_not_apply_manual_edits(self):
        dialog = SubjectMaskDialog(self.image, Mock())
        paint_mask(dialog.canvas.mask, (20, 20), 5, True)
        dialog.canvas.has_mask = True
        dialog.reject()
        self.assertIsNone(dialog.accepted_mask)
        dialog.deleteLater()

    def test_empty_mask_cannot_be_accepted(self):
        dialog = SubjectMaskDialog(self.image, Mock())
        with patch("subject_mask._logger", return_value=Mock()):
            dialog.accept()
        self.assertIsNone(dialog.accepted_mask)
        self.assertIn("Create a mask", dialog.status.text())
        dialog.deleteLater()

    def test_stale_or_cancelled_result_never_replaces_current_mask(self):
        dialog = SubjectMaskDialog(self.image, Mock())
        dialog._edited()
        dialog._received((np.ones((40, 80), bool), {}), 0)
        self.assertFalse(dialog.canvas.has_mask)
        dialog.cancel_inference()
        dialog._received((np.ones((40, 80), bool), {}), 1)
        self.assertFalse(dialog.canvas.has_mask)
        dialog.deleteLater()

    def test_no_inference_or_model_load_on_dialog_open(self):
        store = Mock()
        dialog = SubjectMaskDialog(self.image, store)
        store.get_sam2.assert_not_called()
        self.assertFalse(dialog.download.isChecked())
        dialog.reject()
        dialog.deleteLater()

    def test_inference_uses_bgr_conversion_and_exact_prompt_dimensions(self):
        import torch

        model = Mock(
            return_value=SimpleNamespace(pred_masks=torch.ones((1, 1, 1, 40, 80)))
        )
        processor = Mock(
            return_value={
                "pixel_values": torch.zeros((1, 3, 40, 80)),
                "original_sizes": torch.tensor([[40, 80]]),
            }
        )
        processor.post_process_masks.return_value = [
            torch.ones((1, 1, 40, 80), dtype=torch.bool)
        ]
        store = Mock()
        store.get_sam2.return_value = model, processor, {"revision": "a" * 40}
        mask, metadata = infer_subject_mask(
            self.image, [(3, 4), (10, 12)], [1, 0], (0, 0, 70, 30), store, device="cpu"
        )
        args = processor.call_args.kwargs
        self.assertEqual(args["input_points"], [[[(3, 4), (10, 12)]]])
        self.assertEqual(args["input_labels"], [[[1, 0]]])
        self.assertEqual(args["input_boxes"], [[[0, 0, 70, 30]]])
        self.assertEqual(args["images"][0, 0].tolist(), [0, 0, 200])
        self.assertEqual(mask.shape, self.image.shape[:2])
        self.assertTrue(mask.all())
        self.assertEqual(metadata["revision"], "a" * 40)
        self.assertFalse(model.call_args.kwargs["multimask_output"])

    def test_inference_cancellation_prevents_model_acquisition(self):
        store = Mock()
        with self.assertRaises(ModelPreparationCancelled):
            infer_subject_mask(
                self.image, [(3, 4)], [1], None, store, cancelled=lambda: True
            )
        store.get_sam2.assert_not_called()

    def test_worker_cancel_is_nonblocking_and_retains_thread_until_finished(self):
        started, release = threading.Event(), threading.Event()

        def inference(*args, **kwargs):
            started.set()
            release.wait(5)
            return np.ones((40, 80), dtype=bool), {}

        dialog = SubjectMaskDialog(self.image, Mock())
        dialog.canvas.points.append((10, 10))
        dialog.canvas.labels.append(1)
        try:
            with patch("subject_mask.infer_subject_mask", side_effect=inference):
                dialog.run_segmentation()
                self.assertTrue(started.wait(2))
                self.assertTrue(has_active_jobs())
                dialog.reject()
                self.assertTrue(has_active_jobs())
                cancel_active_jobs()
                release.set()
                for _ in range(100):
                    self.app.processEvents()
                    if not has_active_jobs():
                        break
                    QTest.qWait(10)
                self.assertFalse(has_active_jobs())
                self.assertIsNone(dialog.accepted_mask)
                self.assertFalse(dialog.canvas.has_mask)
        finally:
            release.set()
            dialog.deleteLater()

    def test_canvas_initial_mask_has_original_dimensions(self):
        canvas = MaskCanvas(self.image, np.ones((40, 80), dtype=bool))
        canvas.resize(400, 400)
        self.assertEqual(canvas.mask.shape, (40, 80))
        rendered = canvas.grab().toImage()
        self.assertFalse(rendered.isNull())
        self.assertEqual(rendered.pixelColor(10, 10).name(), "#202829")
        self.assertEqual(rendered.pixelColor(200, 200).getRgb()[:3], (12, 82, 170))
        canvas.deleteLater()


if __name__ == "__main__":
    unittest.main()
