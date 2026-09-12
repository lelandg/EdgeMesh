"""Panel persistence and explicit-context regressions; no runtime or network calls."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from assistant_panel import AssistantPanel


class AssistantPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.settings = Path(self.directory.name) / "assistant.json"
        self.context = Mock(return_value={"resolution": 128})
        self.log_patches = [patch("assistant_panel.get_logger", return_value=Mock()),
                            patch("agent_runtime.get_logger", return_value=Mock())]
        for item in self.log_patches:
            item.start()
        self.panel = AssistantPanel(context_provider=self.context, settings_path=self.settings)

    def tearDown(self):
        self.panel.shutdown()
        self.panel.deleteLater()
        self.app.processEvents()
        for item in self.log_patches:
            item.stop()
        self.directory.cleanup()

    def test_settings_do_not_persist_keys_prompts_replies_or_context(self):
        self.panel.api_key.setText("private-test-value")
        self.panel.prompt.setPlainText("private prompt")
        self.panel._append_output("private response")
        self.panel.include_context.setChecked(True)
        self.panel._save_preferences()
        raw = self.settings.read_text()
        for secret in ("private-test-value", "private prompt", "private response", "resolution"):
            self.assertNotIn(secret, raw)
        self.context.assert_not_called()
        self.assertIn("providers", json.loads(raw))

    def test_context_provider_called_only_when_opted_in(self):
        self.panel.runtime.config = self.panel._config()
        self.panel.prompt.setPlainText("Help")
        with patch.object(self.panel.runtime, "send") as send:
            self.panel._send()
            self.context.assert_not_called()
            self.assertIsNone(send.call_args.args[1])
            self.panel.include_context.setChecked(True)
            self.panel._send()
            self.context.assert_called_once()
            self.assertEqual(send.call_args.args[1], {"resolution": 128})

    def test_enter_sends_shift_enter_inserts_newline(self):
        submitted = Mock()
        self.panel.prompt.submit.connect(submitted)
        QTest.keyClick(self.panel.prompt, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
        self.assertIn("\n", self.panel.prompt.toPlainText())
        submitted.assert_not_called()
        QTest.keyClick(self.panel.prompt, Qt.Key.Key_Return)
        submitted.assert_called_once()

    def test_shutdown_forgets_session_key(self):
        self.panel.api_key.setText("private-test-value")
        self.panel.shutdown()
        self.assertEqual(self.panel.api_key.text(), "")
        self.assertFalse(self.panel.is_busy)

    def test_editing_key_invalidates_the_old_connection(self):
        self.panel.auth.setCurrentIndex(1)
        self.panel.runtime.ready = True
        self.panel.runtime._secret = "old-key"
        QTest.keyClick(self.panel.api_key, Qt.Key.Key_A)
        self.assertFalse(self.panel.runtime.ready)
        self.assertEqual(self.panel.runtime._secret, "")


if __name__ == "__main__":
    unittest.main()
