"""Real Qt mask gestures, dialog directories, and explicit preference persistence."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QWidget,
)

from subject_mask import MiDaSSetupDialog, SubjectMaskDialog, load_mask_png
from ui_persistence import DialogPersistence, SettingsStore


class MaskSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent = QWidget()
        self.settings = SettingsStore(self.root / "ui-settings.json")
        self.parent.ui_settings = self.settings
        self.parent.dialogs = DialogPersistence(self.settings, self.parent)
        self.widgets = []
        self.image = np.zeros((40, 80, 3), dtype=np.uint8)

    def tearDown(self):
        for widget in self.widgets:
            widget.close()
        self.app.removeEventFilter(self.parent.dialogs)
        self.parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.temp.cleanup()

    def dialog(self):
        dialog = SubjectMaskDialog(self.image, Mock(), self.parent)
        self.widgets.append(dialog)
        return dialog

    def setup_dialog(self):
        dialog = MiDaSSetupDialog(Mock(), self.parent)
        self.widgets.append(dialog)
        return dialog

    def test_enter_on_focused_cancel_rejects_without_accepting_mask(self):
        dialog = self.dialog()
        dialog.canvas.set_mask(np.ones(self.image.shape[:2], dtype=bool))
        dialog.show()
        cancel = dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel)
        cancel.setFocus()
        self.app.processEvents()
        QTest.keyClick(cancel, Qt.Key.Key_Return)
        self.assertIsNone(dialog.accepted_mask)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)

    def test_shift_enter_accepts_without_running_focused_sam_action(self):
        dialog = self.dialog()
        dialog.canvas.set_mask(np.ones(self.image.shape[:2], dtype=bool))
        run_inference = Mock()
        dialog.run_button.clicked.disconnect()
        dialog.run_button.clicked.connect(run_inference)
        dialog.show()
        dialog.run_button.setFocus()
        self.app.processEvents()
        QTest.keyClick(
            dialog.run_button, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier
        )
        run_inference.assert_not_called()
        self.assertIsNotNone(dialog.accepted_mask)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)

    def test_cancel_persists_only_whitelisted_editor_choices(self):
        dialog = self.dialog()
        dialog.tool.setCurrentText("Brush remove")
        dialog.brush.setValue(7)
        dialog.display_mode.setCurrentText("Mask")
        dialog.opacity.setValue(63)
        dialog.download.setChecked(True)
        dialog.canvas.set_mask(np.ones(self.image.shape[:2], dtype=bool))
        dialog.canvas.points = [(3, 4)]
        dialog.canvas.labels = [1]
        dialog.model_metadata = {"model_id": "test-model"}
        dialog.reject()
        self.assertEqual(
            self.settings.get("subject_mask.preferences"),
            {
                "tool": "Brush remove",
                "brush_radius": 7,
                "display_mode": "Mask",
                "opacity": 63,
            },
        )
        reopened = self.dialog()
        self.assertEqual(reopened.canvas.mode, "Brush remove")
        self.assertEqual(reopened.canvas.radius, 7)
        self.assertEqual(reopened.canvas.display_mode, "Mask")
        self.assertEqual(reopened.canvas.overlay_opacity, 0.63)
        self.assertFalse(reopened.canvas.has_mask)
        self.assertFalse(reopened.canvas.points)
        self.assertIsNone(reopened.model_metadata)
        self.assertFalse(reopened.download.isChecked())

    def test_invalid_saved_choices_fall_back_without_loading_other_fields(self):
        self.settings.set(
            "subject_mask.preferences",
            {
                "tool": "retired tool",
                "display_mode": ["Overlay"],
                "brush_radius": True,
                "opacity": -1,
                "points": [[3, 4]],
            },
        )
        dialog = self.dialog()
        self.assertEqual(dialog.tool.currentText(), "Foreground point")
        self.assertEqual(dialog.display_mode.currentText(), "Overlay")
        self.assertEqual(dialog.brush.value(), 12)
        self.assertEqual(dialog.opacity.value(), 40)
        self.assertFalse(dialog.canvas.points)

    def test_service_settings_work_without_parent_ui_settings_attribute(self):
        del self.parent.ui_settings
        dialog = self.dialog()
        dialog.brush.setValue(18)
        dialog.reject()
        self.assertEqual(
            self.settings.get("subject_mask.preferences")["brush_radius"], 18
        )

    def test_midas_remembers_model_but_not_typed_artifact_paths(self):
        dialog = self.setup_dialog()
        dialog.model.setCurrentText("dpt")
        dialog.source.setText(str(self.root / "source-checkout"))
        dialog.weights.setText(str(self.root / "private-checkpoint.pt"))
        dialog.reject()
        reopened = self.setup_dialog()
        self.assertEqual(reopened.model.currentText(), "dpt")
        self.assertEqual(reopened.source.text(), "")
        self.assertEqual(reopened.weights.text(), "")
        self.assertEqual(self.settings.get("midas_setup.model"), "dpt")

    def test_four_browsed_directories_are_remembered_even_on_cancel(self):
        mask = self.dialog()
        setup = self.setup_dialog()
        mask.canvas.set_mask(np.ones(self.image.shape[:2], dtype=bool))
        actions = {
            "subject_mask_import": mask.import_mask,
            "subject_mask_export": mask.export_mask,
            "midas_source": setup.choose_source,
            "midas_checkpoint": setup.choose_weights,
        }
        for key, action in actions.items():
            directory = self.root / key
            directory.mkdir()

            def browse_and_cancel(file_dialog, destination=directory):
                file_dialog.setDirectory(str(destination))
                return QDialog.DialogCode.Rejected

            with patch.object(QFileDialog, "exec", new=browse_and_cancel):
                action()
            recorded = self.settings.get("directories")[key]["directory"]
            self.assertEqual(Path(recorded), directory)
        for key, action in actions.items():
            observed = []

            def inspect_and_cancel(file_dialog):
                observed.append(Path(file_dialog.directory().absolutePath()))
                return QDialog.DialogCode.Rejected

            with patch.object(QFileDialog, "exec", new=inspect_and_cancel):
                action()
            self.assertEqual(observed, [self.root / key])
        self.assertEqual(set(self.settings.get("directories")), set(actions))

    def test_mask_export_adds_png_suffix_to_a_bare_filename(self):
        dialog = self.dialog()
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        mask[4:9, 6:12] = True
        dialog.canvas.set_mask(mask)
        observed_suffixes = []

        def choose_basename(file_dialog):
            observed_suffixes.append(file_dialog.defaultSuffix())
            file_dialog.setDirectory(str(self.root))
            file_dialog.selectFile("foreground")
            return QDialog.DialogCode.Accepted

        with patch.object(QFileDialog, "exec", new=choose_basename):
            dialog.export_mask()
        self.assertEqual(observed_suffixes, ["png"])
        np.testing.assert_array_equal(
            load_mask_png(self.root / "foreground.png", self.image.shape), mask
        )
        self.assertFalse((self.root / "foreground").exists())

    def test_keyboard_brush_gesture_has_no_gaps_and_one_undo_step(self):
        dialog = self.dialog()
        canvas = dialog.canvas
        dialog.tool.setCurrentText("Brush add")
        dialog.brush.setValue(1)
        canvas.keyboard_cursor = (4, 8)
        revision = dialog._revision
        QTest.keyPress(canvas, Qt.Key.Key_Space)
        QTest.keyClick(canvas, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        QTest.keyRelease(canvas, Qt.Key.Key_Space)
        self.assertTrue(canvas.mask[8, 4:15].all())
        self.assertEqual(len(canvas._undo), 1)
        self.assertGreater(dialog._revision, revision)
        expected = canvas.mask.copy()
        canvas.undo()
        self.assertFalse(canvas.has_mask)
        self.assertFalse(canvas.mask.any())
        canvas.redo()
        np.testing.assert_array_equal(canvas.mask, expected)

    def test_keyboard_box_is_one_edit_and_ignores_repeated_space(self):
        dialog = self.dialog()
        canvas = dialog.canvas
        dialog.tool.setCurrentText("Box")
        canvas.keyboard_cursor = (3, 4)
        QTest.keyClick(canvas, Qt.Key.Key_Space)
        repeated = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Space,
            Qt.KeyboardModifier.NoModifier,
            " ",
            True,
        )
        self.app.sendEvent(canvas, repeated)
        self.assertEqual(canvas._keyboard_box_anchor, (3, 4))
        QTest.keyClick(canvas, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        QTest.keyClick(canvas, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
        self.assertFalse(canvas.can_undo)
        QTest.keyClick(canvas, Qt.Key.Key_Space)
        self.assertEqual(canvas.box, (3, 4, 13, 14))
        self.assertEqual(len(canvas._undo), 1)
        canvas.undo()
        self.assertIsNone(canvas.box)
        canvas.redo()
        self.assertEqual(canvas.box, (3, 4, 13, 14))

    def test_keyboard_points_clamp_cursor_and_ignore_repeated_space(self):
        dialog = self.dialog()
        canvas = dialog.canvas
        canvas.keyboard_cursor = (0, 0)
        QTest.keyClick(canvas, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
        QTest.keyClick(canvas, Qt.Key.Key_Up)
        self.assertEqual(canvas.keyboard_cursor, (0, 0))
        QTest.keyClick(canvas, Qt.Key.Key_Space)
        repeated = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Space,
            Qt.KeyboardModifier.NoModifier,
            " ",
            True,
        )
        self.app.sendEvent(canvas, repeated)
        self.assertEqual(canvas.points, [(0, 0)])
        self.assertEqual(canvas.labels, [1])
        dialog.tool.setCurrentText("Background point")
        QTest.keyClick(canvas, Qt.Key.Key_Right)
        QTest.keyClick(canvas, Qt.Key.Key_Space)
        self.assertEqual(canvas.labels, [1, 0])
        self.assertEqual(len(canvas._undo), 2)

    def test_pending_first_keyboard_box_can_be_undone_immediately(self):
        dialog = self.dialog()
        canvas = dialog.canvas
        dialog.tool.setCurrentText("Box")
        QTest.keyClick(canvas, Qt.Key.Key_Space)
        QTest.keyClick(canvas, Qt.Key.Key_Right)
        QTest.keyClick(canvas, Qt.Key.Key_Down)
        self.assertIsNotNone(canvas.box)
        self.assertFalse(dialog.undo_button.isEnabled())
        QTest.keyClick(canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        self.assertIsNone(canvas.box)
        self.assertIsNone(canvas._keyboard_box_anchor)
        self.assertTrue(canvas.can_redo)

    def test_tool_switch_finishes_box_before_the_next_edit(self):
        dialog = self.dialog()
        canvas = dialog.canvas
        dialog.tool.setCurrentText("Box")
        QTest.keyClick(canvas, Qt.Key.Key_Space)
        QTest.keyClick(canvas, Qt.Key.Key_Right)
        QTest.keyClick(canvas, Qt.Key.Key_Down)
        dialog.tool.setCurrentText("Brush add")
        self.assertIsNone(canvas._keyboard_box_anchor)
        self.assertEqual(len(canvas._undo), 1)
        QTest.keyClick(canvas, Qt.Key.Key_Space)
        self.assertEqual(len(canvas._undo), 2)

    def test_shift_enter_keeps_validation_and_disabled_accept_gate(self):
        dialog = self.dialog()
        dialog.show()
        dialog.canvas.setFocus()
        self.app.processEvents()
        with patch("subject_mask._logger", return_value=Mock()):
            QTest.keyClick(
                dialog.canvas, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier
            )
        self.assertIsNone(dialog.accepted_mask)
        self.assertIn("Create a mask", dialog.status.text())
        dialog.tool.setCurrentText("Brush add")
        QTest.keyClick(dialog.canvas, Qt.Key.Key_Space)
        accept = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
        accept.setEnabled(False)
        QTest.keyClick(
            dialog.canvas, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier
        )
        self.assertIsNone(dialog.accepted_mask)
        accept.setEnabled(True)
        QTest.keyClick(
            dialog.canvas, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier
        )
        self.assertIsNotNone(dialog.accepted_mask)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)

    def test_controls_have_buddies_and_stable_window_names(self):
        dialog = self.dialog()
        setup = self.setup_dialog()
        self.assertEqual(dialog.objectName(), "subjectMaskDialog")
        self.assertEqual(setup.objectName(), "midasSetupDialog")
        for owner, controls in (
            (
                dialog,
                (
                    dialog.canvas,
                    dialog.tool,
                    dialog.brush,
                    dialog.display_mode,
                    dialog.opacity,
                ),
            ),
            (setup, (setup.model, setup.source, setup.weights)),
        ):
            buddies = [label.buddy() for label in owner.findChildren(QLabel)]
            for control in controls:
                self.assertIn(control, buddies)
                self.assertTrue(control.accessibleName())
        self.assertTrue(
            dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isDefault()
        )


if __name__ == "__main__":
    unittest.main()
