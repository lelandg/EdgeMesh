"""Actual PySide6 application workflows with isolated user storage and no inference."""

import contextlib
import io
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtTest import QTest

import edge_mesh
import feature_workflows
from session_state import SessionDocument, load_session


class WorkflowUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        self.environment = patch.dict(
            os.environ, {"EDGEMESH_DATA_DIR": str(self.root / "data")}
        )
        self.environment.start()
        self.source = self.root / "source.png"
        image = np.zeros((12, 20, 3), np.uint8)
        image[2:10, 4:16] = [40, 150, 230]
        self.assertTrue(cv2.imwrite(str(self.source), image))
        config_path = self.root / "data" / "config.ini"
        config_path.parent.mkdir()
        config_path.write_text(
            "[Settings]\nlast_used_image = " + str(self.source) + "\n"
        )
        self.repo_config = Path(edge_mesh.__file__).parent / "config.ini"
        self.original_repo_config = (
            self.repo_config.read_bytes() if self.repo_config.exists() else None
        )
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.window = edge_mesh.MainWindowImageProcessing(verbose=False)
        self.application.processEvents()

    def tearDown(self):
        if hasattr(self, "window"):
            self.window._history_timer.stop()
            self.window._viewport_timer.stop()
            self.window.close()
            self.window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.application.processEvents()
        self.redirect.__exit__(None, None, None)
        # Do not leave rotating handlers referencing temporary directories.
        for candidate in logging.Logger.manager.loggerDict.values():
            if isinstance(candidate, logging.Logger):
                for handler in list(candidate.handlers):
                    filename = getattr(handler, "baseFilename", None)
                    if filename and Path(filename).is_relative_to(self.root):
                        handler.close()
                        candidate.removeHandler(handler)
        self.environment.stop()
        self.directory.cleanup()

    def assert_no_workflow_error(self):
        self.assertNotIn("Error:", self.output.getvalue())
        self.assertTrue(hasattr(self.window, "health_action"), self.output.getvalue())
        self.assertEqual(self.window._error_log.toPlainText(), "")

    def test_da3_variants_appear_in_actual_model_license_dialog(self):
        from da3_backend import DA3_MODELS
        from PySide6.QtWidgets import QComboBox
        inspected = []

        def inspect_dialog(dialog):
            selector = dialog.findChild(QComboBox)
            inspected.extend(selector.itemData(index) for index in range(selector.count()))
            return QDialog.DialogCode.Rejected

        with patch('model_compliance_ui.QDialog.exec', new=inspect_dialog):
            self.window.model_details()
        self.assertTrue(set(DA3_MODELS).issubset(inspected))

    def test_da3_variants_are_selectable_and_persist_in_settings(self):
        from da3_backend import DA3_ALIASES
        for label in DA3_ALIASES:
            with self.subTest(model=label):
                self.assertGreaterEqual(self.window.depth_method_dropdown.findText(label), 0)
                self.window.depth_method_dropdown.setCurrentText(label)
                self.application.processEvents()
                self.assertEqual(self.window._settings_snapshot()['model'], label)
                self.assertNotIn('NC', self.window.process_button.text())
                self.assertNotIn('Check terms', self.window.process_button.text())
        self.assert_no_workflow_error()

    def test_actual_startup_is_offline_health_optional_and_user_scoped(self):
        self.assert_no_workflow_error()
        self.assertTrue(self.window.initialized)
        self.assertFalse(self.window.health_action.isChecked())
        self.assertFalse(self.window.download_action.isChecked())
        self.assertIsNone(self.window.three_d_viewport)
        self.assertEqual(self.window.image.shape, (12, 20, 3))
        self.assertEqual(
            Path(self.window.CONFIG_FILE_PATH), self.root / "data" / "config.ini"
        )
        self.window.save_ui_settings()
        self.assertEqual(
            self.repo_config.read_bytes() if self.repo_config.exists() else None,
            self.original_repo_config,
        )
        self.assertFalse(self.window._jobs.busy)

    def test_export_dialog_writes_accepted_mesh_without_loading_selected_model(self):
        import open3d as o3d
        from embedded_viewport import EmbeddedMeshViewport, validated_mesh
        from PySide6.QtWidgets import QFileDialog

        viewport = EmbeddedMeshViewport(self.window)
        viewport.mesh = validated_mesh(o3d.geometry.TriangleMesh.create_box())
        self.window.three_d_viewport = viewport
        self.window._seal_accepted_mesh(
            {"model_id": "depth-anything/Depth-Anything-V2-Large-hf",
             "revision": "a" * 40}, settings={"model": "DepthAnythingV2"}
        )
        for model in ("DepthAnythingV2", "DPT", "MiDaS"):
            self.window.depth_method_dropdown.setCurrentText(model)
            self.window._refresh_workspace()
            for suffix in ("obj", "stl"):
                with self.subTest(model=model, format=suffix):
                    target = self.root / f"{model}.{suffix}"

                    def accept_export(dialog):
                        self.assertEqual(dialog.windowTitle(), "Export Mesh")
                        dialog.selectNameFilter(f"{suffix.upper()} Files (*.{suffix})")
                        dialog.selectFile(str(target))
                        return QDialog.DialogCode.Accepted

                    with patch.object(QFileDialog, "exec", accept_export), patch.object(
                        self.window, "_start_generation"
                    ) as generate, patch.object(
                        self.window.model_store, "get_depth",
                        side_effect=AssertionError("Export must not load a depth model"),
                    ) as load_model:
                        self.window.export_mesh_button.click()
                    generate.assert_not_called()
                    load_model.assert_not_called()
                    self.assertTrue(target.is_file())
                    reloaded = o3d.io.read_triangle_mesh(str(target))
                    self.assertEqual(len(reloaded.triangles), len(viewport.mesh.triangles))
                    np.testing.assert_allclose(reloaded.get_min_bound(), viewport.mesh.get_min_bound())
                    np.testing.assert_allclose(reloaded.get_max_bound(), viewport.mesh.get_max_bound())
                    self.assert_no_workflow_error()

    def test_startup_without_image_or_saved_ui_settings(self):
        self.window._history_timer.stop()
        self.window.close()
        # Closing intentionally persists the last project; this test requests a blank startup.
        self.window.ui_settings.update({"last_project": ""})
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        (self.root / "data" / "config.ini").write_text(
            "[Settings]\nlast_used_image =\n"
        )
        example = str(Path(edge_mesh.__file__).parent / "Images" / "example.png")
        real_isfile = os.path.isfile
        with patch(
            "edge_mesh.os.path.isfile",
            side_effect=lambda value: False
            if os.path.normpath(value) == os.path.normpath(example)
            else real_isfile(value),
        ):
            self.window = edge_mesh.MainWindowImageProcessing(verbose=False)
        self.assert_no_workflow_error()
        self.assertIsNone(self.window.image)
        self.assertIsNotNone(self.window.session_history.current)

    def test_explicit_project_startup_skips_stale_previous_project(self):
        self.window._history_timer.stop()
        self.window.close()
        stale_project = self.root / "missing-last-project" / "project.edgemesh.json"
        self.assertFalse(stale_project.exists())
        self.window.ui_settings.update({"last_project": str(stale_project)})
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        (self.root / "data" / "config.ini").write_text(
            "[Settings]\nlast_used_image =\n"
        )
        example = str(Path(edge_mesh.__file__).parent / "Images" / "example.png")
        real_isfile = os.path.isfile
        with patch(
            "edge_mesh.os.path.isfile",
            side_effect=lambda value: False
            if os.path.normpath(value) == os.path.normpath(example)
            else real_isfile(value),
        ):
            self.window = edge_mesh.MainWindowImageProcessing(
                verbose=False, restore_last_project=False
            )
        self.application.processEvents()
        self.assertTrue(self.window.initialized)
        self.assertTrue(self.window._product_ready)
        self.assertIsNone(self.window.image)
        self.assertFalse(self.window.image_path)
        self.assertIsNone(self.window.project_store.current_path)
        self.assertEqual(
            self.window.ui_settings.get("last_project"), str(stale_project)
        )
        self.assert_no_workflow_error()

    def test_failed_close_after_job_keeps_window_open_and_allows_later_work(self):
        self.window._closing_after_job = True
        event = QCloseEvent()
        with patch.object(self.window, "_save_project_now", return_value=False):
            self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertFalse(self.window._closing_after_job)
        self.assertTrue(self.window.initialized)

    def test_suggestions_require_acceptance_and_can_be_undone(self):
        self.assert_no_workflow_error()
        self.window._apply_settings({"depth_amount": 4.0})
        before = self.window._settings_snapshot()
        with patch.object(
            feature_workflows.QDialog, "exec", return_value=QDialog.DialogCode.Rejected
        ):
            self.window.suggest_local_parameters()
        self.assertEqual(self.window._settings_snapshot(), before)
        with patch.object(
            feature_workflows.QDialog, "exec", return_value=QDialog.DialogCode.Accepted
        ):
            self.window.suggest_local_parameters()
        self.assertNotEqual(self.window._settings_snapshot(), before)
        self.window._history_move(False)
        self.assertEqual(self.window._settings_snapshot(), before)
        self.assert_no_workflow_error()

    def test_live_depth_field_records_history_after_editing(self):
        self.assert_no_workflow_error()
        before = float(self.window.depth_amount_input.text())
        self.window._record_session()
        self.window.depth_amount_input.setText("2.25")
        self.window.depth_amount_input.editingFinished.emit()
        QTest.qWait(300)
        self.assertEqual(
            self.window.session_history.current.settings["depth_amount"], 2.25
        )
        self.assertTrue(self.window.session_history.can_undo)
        self.window._history_move(False)
        self.assertEqual(float(self.window.depth_amount_input.text()), before)
        self.assert_no_workflow_error()

    def test_closed_mask_dialog_releases_source_preview(self):
        from subject_mask import SubjectMaskDialog

        with patch.object(
            SubjectMaskDialog, "exec", return_value=QDialog.DialogCode.Rejected
        ):
            for _ in range(3):
                self.window.edit_subject_mask()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertEqual(self.window.findChildren(SubjectMaskDialog), [])

    def test_corrupt_model_identity_does_not_partly_restore(self):
        self.assert_no_workflow_error()
        previous_path = self.window.image_path
        previous_image = self.window.image.copy()
        previous_settings = self.window._settings_snapshot()
        other = self.root / "other.png"
        self.assertTrue(cv2.imwrite(str(other), np.full_like(previous_image, 255)))
        document = SessionDocument(
            str(other),
            {"depth_amount": 3.5},
            {
                "backend": "huggingface",
                "model_type": "depth_pro",
                "revision": "invalid",
            },
        )
        with self.assertRaises(ValueError):
            self.window._restore_document(document)
        self.assertEqual(self.window.image_path, previous_path)
        np.testing.assert_array_equal(self.window.image, previous_image)
        self.assertEqual(self.window._settings_snapshot(), previous_settings)
        self.assert_no_workflow_error()

    def test_session_save_load_restores_settings_and_accepted_mask(self):
        self.assert_no_workflow_error()
        mask = np.zeros((12, 20), bool)
        mask[2:10, 4:16] = True
        self.window._apply_settings({"depth_amount": 2.5, "resolution": 360})
        dialog = SimpleNamespace(
            exec=lambda: QDialog.DialogCode.Accepted,
            accepted_mask=mask,
            deleteLater=Mock(),
        )
        with patch("subject_mask.SubjectMaskDialog", return_value=dialog):
            self.window.edit_subject_mask()
        np.testing.assert_array_equal(self.window._subject_mask, mask)
        destination = self.root / "session.json"
        with patch.object(
            self.window.dialogs, "save_file", return_value=(str(destination), "")
        ):
            self.window.save_current_session()
        saved = load_session(destination)
        np.testing.assert_array_equal(saved.mask, mask)
        self.window.clear_subject_mask()
        self.window._apply_settings({"depth_amount": 0.4, "resolution": 200})
        with patch.object(
            self.window.dialogs, "open_file", return_value=(str(destination), "")
        ):
            self.window.open_session()
        self.assertEqual(self.window.depth_amount_input.text(), "2.5")
        self.assertEqual(self.window.resolution_input.text(), "360")
        np.testing.assert_array_equal(self.window._subject_mask, mask)
        self.assert_no_workflow_error()

    def test_undo_redo_settings_and_mask_acceptance(self):
        self.assert_no_workflow_error()
        baseline = float(self.window.depth_amount_input.text())
        self.window._record_session()
        self.window.depth_amount_input.setText("2.5")
        self.window._record_session()
        self.window._history_move(False)
        self.assertEqual(float(self.window.depth_amount_input.text()), baseline)
        self.window._history_move(True)
        self.assertEqual(float(self.window.depth_amount_input.text()), 2.5)
        mask = np.ones(self.window.image.shape[:2], bool)
        dialog = SimpleNamespace(
            exec=lambda: QDialog.DialogCode.Accepted,
            accepted_mask=mask,
            deleteLater=Mock(),
        )
        with patch("subject_mask.SubjectMaskDialog", return_value=dialog):
            self.window.edit_subject_mask()
        self.assertIsNotNone(self.window._subject_mask)
        self.window._history_move(False)
        self.assertIsNone(self.window._subject_mask)
        self.window._history_move(True)
        np.testing.assert_array_equal(self.window._subject_mask, mask)
        self.assert_no_workflow_error()

    def test_preset_roundtrip_retains_source_and_mask(self):
        self.assert_no_workflow_error()
        self.assertTrue(self.window._save_project_now(force=True))
        managed_source = Path(self.window.image_path)
        self.assertNotEqual(managed_source, self.source)
        self.assertTrue(managed_source.is_relative_to(self.window.project_store.root))
        np.testing.assert_array_equal(
            cv2.imread(str(managed_source)), self.window.image
        )
        mask = np.ones(self.window.image.shape[:2], bool)
        self.window._subject_mask = mask.copy()
        self.window._apply_settings({"depth_amount": 1.8, "grayscale_enabled": True})
        path = self.root / "data" / "presets" / "relief.json"
        with patch.object(
            self.window.dialogs, "save_file", return_value=(str(path), "")
        ):
            self.window.save_current_preset()
        self.window._apply_settings({"depth_amount": 0.5, "grayscale_enabled": False})
        with patch.object(
            self.window.dialogs, "open_file", return_value=(str(path), "")
        ):
            self.window.load_current_preset()
        self.assertEqual(float(self.window.depth_amount_input.text()), 1.8)
        self.assertTrue(self.window.grayscale_checkbox.isChecked())
        self.assertEqual(self.window.image_path, str(managed_source))
        np.testing.assert_array_equal(self.window._subject_mask, mask)
        self.assert_no_workflow_error()


if __name__ == "__main__":
    unittest.main()
