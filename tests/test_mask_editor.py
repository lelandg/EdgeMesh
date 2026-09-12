"""Mask editor transactions, preview-only controls, and original-size PNG I/O."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtWidgets import QApplication, QWidget

from subject_mask import (
    MaskCanvas,
    SubjectMaskDialog,
    load_mask_png,
    mask_preview_bgr,
    save_mask_png,
)


from ui_persistence import DialogPersistence, SettingsStore


class MaskEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.image = np.full((40, 80, 3), [200, 80, 20], dtype=np.uint8)
        self.widgets = []
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = QWidget()
        self.parent.ui_settings = SettingsStore(
            Path(self.temp.name) / "ui-settings.json"
        )
        self.parent.dialogs = DialogPersistence(self.parent.ui_settings, self.parent)

    def tearDown(self):
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        self.app.removeEventFilter(self.parent.dialogs)
        self.parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def canvas(self, initial=None):
        canvas = MaskCanvas(self.image, initial)
        canvas.resize(320, 200)
        self.widgets.append(canvas)
        return canvas

    def dialog(self, initial=None):
        dialog = SubjectMaskDialog(
            self.image, Mock(), parent=self.parent, initial_mask=initial
        )
        self.widgets.append(dialog)
        return dialog

    @staticmethod
    def event(x, y):
        # Source coordinates through the real 320 × 200 letterboxed preview.
        return SimpleNamespace(
            position=lambda: QPointF(x * 4 + 2, y * 4 + 22),
            button=lambda: Qt.MouseButton.LeftButton,
            buttons=lambda: Qt.MouseButton.LeftButton,
        )

    def click(self, canvas, x, y):
        event = self.event(x, y)
        canvas.mousePressEvent(event)
        canvas.mouseReleaseEvent(event)

    def test_brush_drag_is_one_undoable_transaction_and_redo_restores_it(self):
        canvas = self.canvas()
        canvas.mode, canvas.radius = "Brush add", 1
        canvas.mousePressEvent(self.event(5, 15))
        for x in (12, 22, 35):
            canvas.mouseMoveEvent(self.event(x, 15))
        canvas.mouseReleaseEvent(self.event(35, 15))
        completed = canvas.mask.copy()
        self.assertTrue(completed[15, 5:36].all())
        self.assertEqual(len(canvas._undo), 1)
        canvas.undo()
        self.assertFalse(canvas.has_mask)
        self.assertFalse(canvas.mask.any())
        canvas.redo()
        self.assertTrue(canvas.has_mask)
        np.testing.assert_array_equal(canvas.mask, completed)

    def test_prompts_boxes_and_clear_restore_together(self):
        initial = np.ones(self.image.shape[:2], bool)
        canvas = self.canvas(initial)
        self.click(canvas, 8, 9)
        canvas.mode = "Background point"
        self.click(canvas, 22, 11)
        canvas.mode = "Box"
        canvas.mousePressEvent(self.event(40, 30))
        canvas.mouseMoveEvent(self.event(4, 5))
        canvas.mouseReleaseEvent(self.event(4, 5))
        self.assertEqual(canvas.box, (4, 5, 40, 30))
        canvas.clear()
        self.assertFalse(canvas.has_mask)
        self.assertFalse(canvas.points)
        canvas.undo()
        self.assertEqual(canvas.points, [(8, 9), (22, 11)])
        self.assertEqual(canvas.labels, [1, 0])
        self.assertEqual(canvas.box, (4, 5, 40, 30))
        np.testing.assert_array_equal(canvas.mask, initial)
        canvas.undo()
        self.assertIsNone(canvas.box)
        canvas.undo()
        self.assertEqual(canvas.labels, [1])

    def test_new_edit_discards_redo_and_noop_does_not_consume_history(self):
        canvas = self.canvas()
        canvas.clear()
        self.assertFalse(canvas.can_undo)
        self.click(canvas, 8, 9)
        self.click(canvas, 22, 11)
        canvas.undo()
        self.assertTrue(canvas.can_redo)
        self.click(canvas, 30, 20)
        self.assertFalse(canvas.can_redo)
        mask = np.ones(self.image.shape[:2], bool)
        canvas.set_mask(mask)
        count = len(canvas._undo)
        canvas.set_mask(mask)
        self.assertEqual(len(canvas._undo), count)

    def test_history_has_both_count_and_packed_pixel_memory_limits(self):
        canvas = self.canvas()
        canvas.HISTORY_LIMIT = 3
        for x in range(5):
            self.click(canvas, x, 10)
        self.assertEqual(len(canvas._undo), 3)
        for _ in range(5):
            canvas.undo()
        self.assertEqual(len(canvas.points), 2)
        canvas.HISTORY_BYTES = len(np.packbits(canvas.mask)) * 2
        for x in range(5, 10):
            self.click(canvas, x, 10)
        self.assertEqual(len(canvas._undo), 2)

    def test_preview_modes_preserve_source_mask_dimensions_and_bgr_order(self):
        mask = np.zeros(self.image.shape[:2], bool)
        mask[3:8, 12:20] = True
        original, original_mask = self.image.copy(), mask.copy()
        preview = mask_preview_bgr(self.image, mask)
        self.assertEqual(preview[4, 14].tolist(), [170, 130, 24])
        self.assertEqual(preview[0, 0].tolist(), [200, 80, 20])
        np.testing.assert_array_equal(
            mask_preview_bgr(self.image, mask, opacity=0), original
        )
        np.testing.assert_array_equal(
            mask_preview_bgr(self.image, mask, mode="Source"), original
        )
        binary = mask_preview_bgr(self.image, mask, mode="Mask")
        self.assertEqual(binary.shape, self.image.shape)
        self.assertEqual(binary[4, 14].tolist(), [255, 255, 255])
        self.assertEqual(binary[0, 0].tolist(), [0, 0, 0])
        np.testing.assert_array_equal(self.image, original)
        np.testing.assert_array_equal(mask, original_mask)

    def test_view_controls_leave_inference_revision_and_edit_history_unchanged(self):
        dialog = self.dialog()
        revision = dialog._revision
        dialog.display_mode.setCurrentText("Mask")
        dialog.opacity.setValue(75)
        self.assertEqual(dialog.canvas.display_mode, "Mask")
        self.assertEqual(dialog.canvas.overlay_opacity, 0.75)
        self.assertEqual(dialog._revision, revision)
        self.assertFalse(dialog.canvas.can_undo)
        self.assertFalse(dialog.opacity.isEnabled())
        dialog.display_mode.setCurrentText("Overlay")
        self.assertTrue(dialog.opacity.isEnabled())

    def test_inference_result_is_undoable_and_undo_invalidates_stale_results(self):
        dialog = self.dialog()
        dialog.canvas.points = [(12, 8)]
        dialog.canvas.labels = [1]
        dialog._edited()
        mask = np.ones(self.image.shape[:2], bool)
        dialog._received((mask, {"revision": "test-model"}), dialog._revision)
        self.assertTrue(dialog.undo_button.isEnabled())
        self.assertEqual(dialog.model_metadata, {"revision": "test-model"})
        result_revision = dialog._revision
        dialog.canvas.undo()
        self.assertFalse(dialog.canvas.has_mask)
        self.assertEqual(dialog.canvas.points, [(12, 8)])
        self.assertIsNone(dialog.model_metadata)
        dialog._received((mask, {}), result_revision)
        self.assertFalse(dialog.canvas.has_mask)
        self.assertIn("stale result", dialog.status.text())

    def test_png_roundtrip_handles_unicode_paths_and_unaligned_original_width(self):
        path = Path(self.temp.name) / "subject-水.png"
        mask = np.zeros((39, 83), bool)
        mask[2:35, 10:53] = True
        save_mask_png(path, mask)
        loaded = load_mask_png(path, (39, 83, 3))
        self.assertEqual(loaded.dtype, np.bool_)
        np.testing.assert_array_equal(loaded, mask)
        with self.assertRaisesRegex(ValueError, "must match the original image"):
            load_mask_png(path, self.image.shape)

    def test_import_is_undoable_and_wrong_size_is_logged_without_changing_mask(self):
        dialog = self.dialog()
        path = Path(self.temp.name) / "mask.png"
        save_mask_png(path, np.ones(self.image.shape[:2], bool))
        with patch(
            "ui_persistence.DialogPersistence.open_file", return_value=(str(path), "")
        ):
            dialog.import_mask()
        self.assertTrue(dialog.canvas.mask.all())
        dialog.canvas.undo()
        self.assertFalse(dialog.canvas.has_mask)
        save_mask_png(path, np.ones((11, 19), bool))
        logger = Mock()
        with (
            patch(
                "ui_persistence.DialogPersistence.open_file",
                return_value=(str(path), ""),
            ),
            patch("subject_mask._logger", return_value=logger),
        ):
            dialog.import_mask()
        self.assertFalse(dialog.canvas.has_mask)
        logger.error.assert_called_once()
        self.assertIn("80 × 40", dialog.status.text())

    def test_export_failure_is_visible_and_logged(self):
        dialog = self.dialog(np.ones(self.image.shape[:2], bool))
        logger = Mock()
        path = Path(self.temp.name) / "missing-directory" / "mask.png"
        with (
            patch(
                "ui_persistence.DialogPersistence.save_file",
                return_value=(str(path), ""),
            ),
            patch("subject_mask._logger", return_value=logger),
        ):
            dialog.export_mask()
        logger.error.assert_called_once()
        self.assertIn("Could not save mask PNG", dialog.status.text())


if __name__ == "__main__":
    unittest.main()
