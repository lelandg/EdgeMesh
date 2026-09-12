"""Exercise persisted UI behavior with temporary storage and offscreen widgets."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QDir, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFileDialog, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from ui_persistence import DialogPersistence, SettingsStore, get_dialog_service


class SettingsStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "ui-settings.json"
        self.error = patch("ui_persistence._error").start()

    def tearDown(self):
        patch.stopall()
        self.temporary.cleanup()

    def test_round_trip_and_returned_values_cannot_mutate_accepted_settings(self):
        store = SettingsStore(self.path)
        source = {"width": 800, "tabs": [0, 2], "visible": True}
        self.assertTrue(store.set("layout", source))
        source["tabs"].append(3)
        returned = store.get("layout")
        returned["tabs"].append(4)
        self.assertEqual(SettingsStore(self.path).get("layout")["tabs"], [0, 2])
        self.assertEqual(store.get("layout")["tabs"], [0, 2])
        self.assertEqual(store.get("missing", "default"), "default")

    def test_failed_atomic_write_preserves_disk_and_accepted_data(self):
        store = SettingsStore(self.path)
        self.assertTrue(store.set("zoom", 1.5))
        before = self.path.read_bytes()
        with patch("ui_persistence.atomic_write", side_effect=OSError("disk full")):
            self.assertFalse(store.set("zoom", 9))
        self.assertEqual(store.get("zoom"), 1.5)
        self.assertEqual(self.path.read_bytes(), before)
        self.error.assert_called()

    def test_malformed_file_uses_defaults_without_overwriting_the_original(self):
        self.path.write_text("{broken", encoding="utf-8")
        store = SettingsStore(self.path)
        self.assertEqual(store.get("layout", {}), {})
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{broken")
        self.error.assert_called_once()

    def test_rejects_secrets_nonfinite_values_and_oversized_structures(self):
        store = SettingsStore(self.path)
        invalid = [
            {"api_key": "do-not-store"}, {"provider": {"refreshToken": "do-not-store"}},
            {"password": "do-not-store"}, {"provider.access_token": "do-not-store"},
            {"zoom": float("nan")}, {"zoom": float("inf")},
            {"blob": "x" * 140_000}, {"count": 2 ** 100}, {"unknown": object()},
        ]
        for value in invalid:
            with self.subTest(keys=list(value)):
                self.assertFalse(store.update(value))
        self.assertFalse(self.path.exists())
        self.assertIsNone(store.get("api_key"))

    def test_rejects_duplicate_keys_and_credentials_read_from_disk(self):
        for text in ['{"zoom":1,"zoom":2}', '{"secret":"do-not-store"}', '[1,2]', '{"zoom":NaN}']:
            self.path.write_text(text, encoding="utf-8")
            store = SettingsStore(self.path)
            self.assertEqual(store.get("zoom", "default"), "default")
            self.assertIsNone(store.get("secret"))
            self.assertEqual(self.path.read_text(encoding="utf-8"), text)

    def test_default_path_is_user_storage(self):
        with patch.dict(os.environ, {"EDGEMESH_DATA_DIR": self.temporary.name}):
            self.assertEqual(SettingsStore().path, self.path)


class DialogPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "ui-settings.json"
        self.store = SettingsStore(self.path)
        self.service = DialogPersistence(self.store)
        self.windows = []
        self.error = patch("ui_persistence._error").start()

    def tearDown(self):
        self.application.removeEventFilter(self.service)
        for window in [*self.windows, *self.service._file_dialogs.values()]:
            try:
                window.close()
                window.deleteLater()
            except RuntimeError:
                pass
        self.service.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.application.processEvents()
        patch.stopall()
        self.temporary.cleanup()

    def window(self, name="test-dialog", kind=QDialog):
        window = kind()
        window.setObjectName(name)
        self.windows.append(window)
        return window

    def test_window_geometry_is_automatically_restored_for_a_new_instance(self):
        first = self.window()
        first.resize(510, 330)
        first.show()
        self.application.processEvents()
        first.close()
        second = self.window()
        second.resize(200, 100)
        second.show()
        self.application.processEvents()
        self.assertEqual(second.size(), first.size())
        self.assertIn("test-dialog", SettingsStore(self.path).get("windows"))

    def test_offscreen_windows_are_recovered_onto_an_available_screen(self):
        first = self.window()
        first.resize(300, 200)
        first.move(-20000, -20000)
        self.assertTrue(self.service.save_window(first))
        second = self.window()
        self.assertTrue(self.service.restore_window(second))
        self.assertTrue(any(screen.availableGeometry().intersects(second.frameGeometry())
                            for screen in self.application.screens()))

    def test_image_titles_do_not_create_unbounded_geometry_keys(self):
        first = self.window("")
        first.setWindowTitle("Preview first-image.png")
        self.service.save_window(first)
        first.setWindowTitle("Preview second-image.png")
        self.service.save_window(first)
        self.assertEqual(len(self.store.get("windows")), 1)

    def test_transient_popups_are_excluded(self):
        popup = self.window("popup", QWidget)
        popup.setWindowFlags(Qt.WindowType.Popup)
        popup.show()
        self.application.processEvents()
        popup.hide()
        self.assertIsNone(self.store.get("windows"))

    def test_service_is_found_through_nested_parent_widgets(self):
        root = self.window()
        root.dialogs = self.service
        child = QWidget(root)
        grandchild = QWidget(child)
        self.assertIs(get_dialog_service(grandchild), self.service)

    def test_shift_enter_uses_a_button_click_and_respects_disabled_acceptance(self):
        dialog = self.window()
        layout = QVBoxLayout(dialog)
        field = QLineEdit(dialog)
        layout.addWidget(field)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(buttons)
        button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        button.setDefault(True)
        clicked = []
        button.clicked.connect(lambda: clicked.append(True))
        dialog.show()
        self.application.processEvents()
        button.setEnabled(False)
        QTest.keyClick(field, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(clicked, [])
        button.setEnabled(True)
        QTest.keyClick(field, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(clicked, [True])

    def test_cancel_remembers_folder_and_filter_without_selecting_a_file(self):
        target = Path(self.temporary.name) / "target"
        target.mkdir()

        def cancel_after_navigation(dialog):
            dialog.setDirectory(str(target))
            dialog.selectNameFilter("All files (*)")
            return QDialog.DialogCode.Rejected

        with patch.object(QFileDialog, "exec", cancel_after_navigation):
            result = self.service.open_file(None, "Open image", "images", "Images (*.png);;All files (*)",
                                            self.temporary.name)
        self.assertEqual(result, ("", "All files (*)"))
        settings = SettingsStore(self.path).get("directories")["images"]
        self.assertEqual(Path(settings["directory"]), target)
        self.assertEqual(settings["filter"], "All files (*)")
        first = self.service._file_dialogs[("open", "images")]
        reused = self.service._dialog(None, "Load another image", "images", "open",
                                     "Images (*.png);;All files (*)", self.temporary.name)
        self.assertIs(reused, first)
        self.assertEqual(Path(reused.directory().absolutePath()), target)
        self.assertEqual(reused.selectedNameFilter(), "All files (*)")
        self.assertFalse(reused.testOption(QFileDialog.Option.DontUseNativeDialog))
        self.assertIn("state", self.store.get("windows")["file-dialog:open:images"])
        owner = self.window("new-file-dialog-owner")
        reparented = self.service._dialog(owner, "Load image", "images", "open",
                                         "Images (*.png);;All files (*)", self.temporary.name)
        self.assertIs(reparented, first)
        self.assertIs(reparented.parentWidget(), owner)

    def test_shift_enter_prefers_acceptance_over_an_unrelated_default_action(self):
        dialog = self.window()
        layout = QVBoxLayout(dialog)
        field = QLineEdit(dialog)
        layout.addWidget(field)
        preview = QPushButton("Run preview", dialog)
        layout.addWidget(preview)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, dialog)
        layout.addWidget(buttons)
        clicked = []
        preview.clicked.connect(lambda: clicked.append("preview"))
        buttons.accepted.connect(lambda: clicked.append("accept"))
        dialog.show()
        self.application.processEvents()
        preview.setDefault(True)
        QTest.keyClick(field, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
        QTest.keyClick(field, Qt.Key.Key_Enter,
                       Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.KeypadModifier)
        self.assertEqual(clicked, ["accept", "accept"])

    def test_shift_enter_cannot_fall_back_to_an_unrelated_default_action(self):
        dialog = self.window()
        layout = QVBoxLayout(dialog)
        field = QLineEdit(dialog)
        layout.addWidget(field)
        preview = QPushButton("Run preview", dialog)
        layout.addWidget(preview)
        clicked = []
        preview.clicked.connect(lambda: clicked.append("preview"))
        dialog.show()
        self.application.processEvents()
        preview.setDefault(True)
        QTest.keyClick(field, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
        QTest.keyClick(field, Qt.Key.Key_Enter,
                       Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.KeypadModifier)
        self.assertEqual(clicked, [])

    def test_shift_enter_supports_explicit_custom_acceptance(self):
        dialog = self.window()
        layout = QVBoxLayout(dialog)
        accept = QPushButton("Use mask", dialog)
        accept.setProperty("edgemeshAcceptButton", True)
        layout.addWidget(accept)
        clicked = []
        accept.clicked.connect(lambda: clicked.append("accept"))
        dialog.show()
        self.application.processEvents()
        QTest.keyClick(dialog, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(clicked, ["accept"])

    def test_binary_dialog_state_cannot_override_the_remembered_target_directory(self):
        first_directory = Path(self.temporary.name) / "first"
        remembered_directory = Path(self.temporary.name) / "remembered"
        first_directory.mkdir()
        remembered_directory.mkdir()
        first = self.service._dialog(None, "Open image", "images", "open", "All files (*)", first_directory)
        self.service.save_window(first)
        # Recreate the dialog, as when a previous temporary parent was closed.
        self.windows.append(first)
        self.service._file_dialogs.clear()
        self.store.set("directories", {"images": {"directory": str(remembered_directory), "filter": "All files (*)"}})
        second = self.service._dialog(None, "Open image", "images", "open", "All files (*)", first_directory)
        second.show()
        self.application.processEvents()
        self.assertEqual(Path(second.directory().absolutePath()), remembered_directory)

    def test_save_and_directory_chooser_return_the_selected_target(self):
        directory = Path(self.temporary.name)
        selected = directory / "new-project.edgeproj"

        def accept(dialog):
            dialog.selectFile(str(selected))
            return QDialog.DialogCode.Accepted

        with patch.object(QFileDialog, "exec", accept):
            path, _ = self.service.save_file(None, "Save project", "projects", "Project (*.edgeproj)",
                                              directory, suggested_name="new-project.edgeproj")
        self.assertEqual(Path(path), selected)

        def accept_directory(dialog):
            dialog.selectFile(str(directory))
            return QDialog.DialogCode.Accepted

        with patch.object(QFileDialog, "exec", accept_directory):
            path = self.service.choose_directory(None, "Project directory", "project-folders", directory)
        self.assertEqual(Path(path), directory)
        self.assertTrue(json.loads(self.path.read_text(encoding="utf-8"))["directories"])

    def test_all_file_chooser_modes_allow_native_dialogs(self):
        def cancel(dialog):
            self.assertFalse(dialog.testOption(QFileDialog.Option.DontUseNativeDialog))
            return QDialog.DialogCode.Rejected

        with patch.object(QFileDialog, "exec", cancel):
            self.assertEqual(self.service.open_file(None, "Load project", "projects")[0], "")
            self.assertEqual(self.service.save_file(None, "Save project", "projects")[0], "")
            self.assertEqual(self.service.choose_directory(None, "Folder", "folders"), "")

    def test_widget_fallback_accepts_an_absolute_path_with_spaces(self):
        # The offscreen platform has no native picker. Exercise PySide6's fallback
        # filename field with the same full path a user can type or paste.
        target_directory = Path(self.temporary.name) / "Project folder"
        target_directory.mkdir()
        target = target_directory / "saved project.edgeproj"
        target.write_bytes(b"file chooser selection only")

        def enter_path(dialog):
            field = dialog.findChild(QLineEdit, "fileNameEdit")
            self.assertIsNotNone(field)
            field.setText(str(target))
            dialog.accept()
            return dialog.result()

        with patch.object(QFileDialog, "exec", enter_path):
            selected, _ = self.service.open_file(None, "Load project", "projects",
                "Project (*.edgeproj)", self.temporary.name)
        self.assertEqual(Path(selected), target)
        self.assertEqual(Path(self.store.get("directories")["projects"]["directory"]), target_directory)

    def test_default_save_suffix_only_fills_a_missing_extension_and_can_be_reset(self):
        directory = Path(self.temporary.name)
        cases = [("png", "image", "image.png"), ("png", "mesh.obj", "mesh.obj"),
                 (None, "plain", "plain")]
        for suffix, typed, expected in cases:
            with self.subTest(suffix=suffix, typed=typed):
                def accept(dialog):
                    self.assertEqual(dialog.defaultSuffix(), suffix or "")
                    dialog.selectFile(str(directory / typed))
                    dialog.accept()
                    return dialog.result()

                with patch.object(QFileDialog, "exec", accept):
                    selected, _ = self.service.save_file(None, "Save", "exports",
                        initial_directory=directory, default_suffix=suffix)
                self.assertEqual(Path(selected), directory / expected)

    def test_accepted_file_and_directory_targets_win_over_the_browsed_folder(self):
        browsed = Path(self.temporary.name) / "browsed"
        chosen = Path(self.temporary.name) / "chosen"
        browsed.mkdir()
        chosen.mkdir()
        target = chosen / "image.png"
        target.write_bytes(b"file chooser selection only")

        def accept_file(dialog):
            dialog.selectFile(str(target))
            return QDialog.DialogCode.Accepted

        with patch.object(QFileDialog, "exec", accept_file), \
                patch.object(QFileDialog, "directory", return_value=QDir(str(browsed))):
            selected, _ = self.service.open_file(None, "Open", "typed-images", initial_directory=browsed)
        self.assertEqual(Path(selected), target)
        self.assertEqual(Path(self.store.get("directories")["typed-images"]["directory"]), chosen)

        def accept_directory(dialog):
            dialog.selectFile(str(chosen))
            return QDialog.DialogCode.Accepted

        with patch.object(QFileDialog, "exec", accept_directory), \
                patch.object(QFileDialog, "directory", return_value=QDir(str(browsed))):
            selected = self.service.choose_directory(None, "Directory", "chosen-folders", browsed)
        self.assertEqual(Path(selected), chosen)
        self.assertEqual(Path(self.store.get("directories")["chosen-folders"]["directory"]), chosen)


if __name__ == "__main__":
    unittest.main()
