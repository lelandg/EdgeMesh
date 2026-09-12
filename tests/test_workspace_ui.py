"""Exercise workspace state against the actual application with isolated storage."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog

import test_workflow_ui as fixtures
from session_state import SessionDocument
from workspace_ui import ImagePreviewLabel


class WorkspaceUITests(unittest.TestCase):
    setUpClass = classmethod(fixtures.WorkflowUITests.setUpClass.__func__)
    setUp = fixtures.WorkflowUITests.setUp
    tearDown = fixtures.WorkflowUITests.tearDown
    assert_no_workflow_error = fixtures.WorkflowUITests.assert_no_workflow_error

    def test_mask_is_visible_after_accept_and_tracks_undo_redo(self):
        from types import SimpleNamespace

        before = self.window.image.copy()
        mask = np.zeros(before.shape[:2], bool)
        mask[2:8, 3:12] = True
        dialog = SimpleNamespace(
            exec=lambda: QDialog.DialogCode.Accepted,
            accepted_mask=mask,
            model_metadata={},
            deleteLater=Mock(),
        )
        with patch("subject_mask.SubjectMaskDialog", return_value=dialog):
            self.window.edit_subject_mask()
        self.assertFalse(self.window.mask_preview.pixmap().isNull())
        self.assertIn("Foreground kept:", self.window.mask_status.text())
        self.assertTrue(self.window.mask_clear_button.isEnabled())
        self.window._history_move(False)
        self.assertIsNone(self.window._subject_mask)
        self.assertIn("No mask.", self.window.mask_status.text())
        self.window._history_move(True)
        np.testing.assert_array_equal(self.window._subject_mask, mask)
        np.testing.assert_array_equal(self.window.image, before)
        self.assert_no_workflow_error()

    def test_mask_view_modes_do_not_change_exported_source_or_mask(self):
        self.window._subject_mask = np.ones(self.window.image.shape[:2], bool)
        source = self.window.image.copy()
        processed = self.window.processed_image.copy()
        for index in range(self.window.mask_view_mode.count()):
            self.window.mask_view_mode.setCurrentIndex(index)
            self.window.mask_opacity.setValue(75)
            self.window._refresh_mask_preview()
        np.testing.assert_array_equal(source, self.window.image)
        np.testing.assert_array_equal(processed, self.window.processed_image)
        self.assertTrue(self.window._subject_mask.all())
        self.assert_no_workflow_error()

    def test_accepted_mask_autosaves_and_restores_preview_and_editor_after_restart(self):
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtTest import QTest

        import edge_mesh
        from project_store import ProjectStore
        from subject_mask import SubjectMaskDialog, mask_preview_bgr

        mask = np.zeros(self.window.image.shape[:2], bool)
        mask[2:8, 3:12] = True

        def accept_mask(dialog):
            dialog.canvas.set_mask(mask)
            dialog.accept()
            return dialog.result()

        with patch.object(SubjectMaskDialog, "exec", accept_mask):
            self.window.edit_subject_mask()
        QTest.qWait(1000)
        project = self.window.project_store.current_path
        self.assertIsNotNone(project)
        np.testing.assert_array_equal(ProjectStore(project.parent).inspect(project).document.mask, mask)

        self.assertTrue(self.window.open_project_path(project))
        np.testing.assert_array_equal(self.window._subject_mask, mask)
        self.assertIn("Foreground kept:", self.window.mask_status.text())

        self.window.close()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.application.processEvents()
        self.window = edge_mesh.MainWindowImageProcessing(verbose=False)
        self.application.processEvents()
        self.assertEqual(self.window.project_store.current_path, project)
        np.testing.assert_array_equal(self.window._subject_mask, mask)
        self.assertIn("Foreground kept:", self.window.mask_status.text())
        expected = mask_preview_bgr(
            self.window.processed_image, mask, mode="Overlay", opacity=0.4
        )
        preview = self.window.mask_preview._source_pixmap.toImage()
        actual = np.array([
            [[preview.pixelColor(x, y).blue(), preview.pixelColor(x, y).green(),
              preview.pixelColor(x, y).red()] for x in range(preview.width())]
            for y in range(preview.height())
        ], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

        def inspect_editor(dialog):
            np.testing.assert_array_equal(dialog.canvas.mask, mask)
            self.assertTrue(dialog.canvas.has_mask)
            dialog.canvas.set_mask(~mask)
            dialog.reject()
            return dialog.result()

        with patch.object(SubjectMaskDialog, "exec", inspect_editor):
            self.window.edit_subject_mask()
        np.testing.assert_array_equal(self.window._subject_mask, mask)
        self.assertTrue(self.window._save_project_now(force=True))
        np.testing.assert_array_equal(ProjectStore(project.parent).inspect(project).document.mask, mask)
        self.assert_no_workflow_error()

    def test_processed_preview_keeps_mask_overlay_and_original_bgr_source(self):
        from subject_mask import mask_preview_bgr

        source = self.window.image.copy()
        mask = np.zeros(source.shape[:2], bool)
        mask[:, : source.shape[1] // 2] = True
        self.window._subject_mask = mask.copy()
        self.window.mask_view_mode.setCurrentText("Overlay")
        self.window.mask_opacity.setValue(60)
        self.window._refresh_mask_preview()
        processed = np.full_like(source, [190, 70, 20])
        self.window.processed_image = processed.copy()
        expected = mask_preview_bgr(processed, mask, mode="Overlay", opacity=0.6)
        # The second refresh reuses the overlay cache and must not leave a raw pixmap.
        for _ in range(2):
            self.window.display_processed_image()
            image = self.window.preview_label._source_pixmap.toImage()
            self.assertEqual((image.height(), image.width()), source.shape[:2])
            actual = np.array(
                [
                    [
                        [
                            image.pixelColor(x, y).blue(),
                            image.pixelColor(x, y).green(),
                            image.pixelColor(x, y).red(),
                        ]
                        for x in range(image.width())
                    ]
                    for y in range(image.height())
                ],
                dtype=np.uint8,
            )
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(self.window.image, source)
        np.testing.assert_array_equal(self.window.processed_image, processed)
        np.testing.assert_array_equal(self.window._subject_mask, mask)
        self.assert_no_workflow_error()

    def test_resize_refits_from_original_resolution_without_distortion(self):
        label = ImagePreviewLabel()
        try:
            original = QPixmap(800, 400)
            original.fill(Qt.GlobalColor.green)
            label.resize(200, 200)
            label.setPixmap(original)
            label.show()
            self.application.processEvents()
            self.assertEqual(label.pixmap().size().width(), 200)
            self.assertEqual(label.pixmap().size().height(), 100)
            label.resize(600, 300)
            self.application.processEvents()
            self.assertEqual(label.pixmap().size().width(), 600)
            self.assertEqual(label.pixmap().size().height(), 300)
            self.assertEqual(label._source_pixmap.size(), original.size())
        finally:
            label.close()
            label.deleteLater()

    def test_history_jump_restores_mask_settings_and_keeps_existing_mesh(self):
        self.window._record_session()
        target = self.window.session_history.index
        expected = self.window.session_history.snapshot_at(target)
        self.window.depth_amount_input.setText("3.25")
        self.window._subject_mask = np.ones(self.window.image.shape[:2], bool)
        self.window._record_session()
        # Geometry must not be claimed as regenerated by restoring a snapshot.
        self.window.mesh_3d = "previous-generated-mesh.obj"
        self.window._restore_history_index(target)
        self.assertEqual(self.window.session_history.index, target)
        self.assertEqual(self.window._settings_snapshot(), expected.settings)
        self.assertIsNone(self.window._subject_mask)
        self.assertEqual(self.window.mesh_3d, "previous-generated-mesh.obj")
        self.assertIn("Generate a mesh", self.window.statusBar().currentMessage())
        self.assert_no_workflow_error()

    def test_failed_history_restore_does_not_move_cursor_or_change_current_source(self):
        self.window._record_session()
        target = self.window.session_history.index
        self.window.depth_amount_input.setText("2.25")
        self.window._record_session()
        index = self.window.session_history.index
        source = self.window.image.copy()
        with (
            patch.object(
                self.window,
                "_restore_document",
                side_effect=ValueError("Unreadable historical image"),
            ),
            patch.object(self.window, "show_error") as error,
        ):
            self.window._restore_history_index(target)
        self.assertEqual(self.window.session_history.index, index)
        np.testing.assert_array_equal(source, self.window.image)
        error.assert_called_once_with("Unreadable historical image")

    def test_restoring_empty_session_clears_all_image_previews(self):
        self.window._restore_document(
            SessionDocument("", self.window._settings_snapshot())
        )
        self.assertIsNone(self.window.image)
        self.assertTrue(self.window.original_label.pixmap().isNull())
        self.assertTrue(self.window.preview_label.pixmap().isNull())
        self.assertIn("No image", self.window.mask_status.text())
        self.assertFalse(self.window.mask_edit_button.isEnabled())
        self.assert_no_workflow_error()

    def test_layout_is_saved_and_can_be_recovered_after_hiding_panels(self):
        self.window.show()
        self.application.processEvents()
        self.window.workspace_splitter.setSizes([250, 600])
        self.window.image_splitter.setSizes([180, 300])
        self.window.save_ui_settings()
        for name in ["workspace_splitter", "image_splitter"]:
            self.assertTrue(self.window.config.get("Workspace", name))
        self.window.controls_dock.hide()
        self.window._error_dock.setFloating(True)
        self.window.workspace_toolbar.hide()
        self.window.reset_workspace_layout()
        self.assertTrue(self.window.controls_dock.isVisible())
        self.assertTrue(self.window.workspace_toolbar.isVisible())
        self.assertFalse(self.window._error_dock.isFloating())
        self.assertEqual(
            self.window.dockWidgetArea(self.window._error_dock),
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self.assert_no_workflow_error()

    def test_saved_layout_restores_docks_created_after_initial_image_load(self):
        self.window.show()
        self.window._error_dock.show()
        self.window.workspace_toolbar.hide()
        self.application.processEvents()
        self.window.save_ui_settings()
        second = fixtures.edge_mesh.MainWindowImageProcessing(verbose=False)
        try:
            second.show()
            self.application.processEvents()
            self.assertTrue(second._error_dock.isVisible())
            self.assertFalse(second.workspace_toolbar.isVisible())
            self.assertEqual(second._error_log.toPlainText(), "")
        finally:
            second._history_timer.stop()
            second.close()
            second.deleteLater()
            self.application.processEvents()


if __name__ == "__main__":
    unittest.main()
