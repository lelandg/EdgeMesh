"""Accessible, opt-in settings advice using the user's installed agent runtimes."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import shutil

from PySide6.QtCore import QByteArray, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

from agent_runtime import AgentRuntime, PROVIDERS, RuntimeConfig, safe_login_url, validate_proposal
from log_utils import get_logger
from user_state import UserPaths, atomic_write


class PromptEdit(QPlainTextEdit):
    submit = Signal()

    def keyPressEvent(self, event):
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and event.modifiers() in {
            Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ControlModifier
        }:
            self.submit.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class AssistantPanel(QWidget):
    """No process starts until Connect is pressed; no returned proposal is applied.

    ``context_provider`` returns JSON-compatible metadata. It is called only for
    an explicit context preview or a send with Include context checked. Inject
    settings_path/state_dir in tests to avoid touching the user's application state.
    """

    def __init__(self, parent=None, context_provider=None, settings_path=None, state_dir=None):
        super().__init__(parent)
        self.setObjectName("assistantPanel")
        self.context_provider = context_provider
        if settings_path is None:
            settings_path = UserPaths.discover().root / "assistant.json"
        self.settings_path = Path(settings_path)
        self.state_dir = Path(state_dir) if state_dir else self.settings_path.parent / "assistant"
        self._preferences = self._load_preferences()
        self._last_provider = ""
        self._response = ""
        self.runtime = AgentRuntime(self.state_dir, self)
        self._build_ui()
        self._load_provider()
        self.runtime.output.connect(self._append_output)
        self.runtime.status_changed.connect(self.status.setText)
        self.runtime.error.connect(self._show_error)
        self.runtime.busy_changed.connect(self._update_controls)
        self.runtime.ready_changed.connect(self._update_controls)
        self.runtime.login_url.connect(self._open_login)
        self._update_controls()

    @property
    def is_busy(self):
        return self.runtime.is_busy

    def _build_ui(self):
        layout = QVBoxLayout(self)
        intro = QLabel("Ask about depth, masks, and mesh settings. Responses are suggestions for you to review. "
                       "Your prompt goes to the selected provider; its account limits or API charges apply.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self.provider = QComboBox()
        self.provider.setAccessibleName("Assistant provider")
        for key, name in PROVIDERS.items():
            self.provider.addItem(name, key)
        self.provider.setCurrentIndex(max(0, self.provider.findData(self._preferences.get("provider", "codex"))))
        form.addRow("&Provider", self.provider)
        row = QHBoxLayout()
        self.runtime_path = QLineEdit()
        self.runtime_path.setAccessibleName("Runtime executable")
        self.runtime_path.setPlaceholderText("Installed executable, or leave blank to search PATH")
        self.browse_button = QPushButton("&Browse…")
        self.browse_button.clicked.connect(self._browse)
        row.addWidget(self.runtime_path, 1)
        row.addWidget(self.browse_button)
        path_label = QLabel("&Runtime")
        path_label.setBuddy(self.runtime_path)
        form.addRow(path_label, row)
        self.auth = QComboBox()
        self.auth.addItem("Account sign-in (OAuth)", "oauth")
        self.auth.addItem("API key (this session only)", "api_key")
        self.auth.setAccessibleName("Assistant authentication mode")
        form.addRow("&Authentication", self.auth)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setAccessibleName("Session-only API key")
        self.api_key.setPlaceholderText("Not saved by EdgeMesh; may also come from the provider's API-key environment variable")
        key_row = QHBoxLayout()
        key_row.addWidget(self.api_key, 1)
        self.forget_key_button = QPushButton("&Forget key")
        self.forget_key_button.clicked.connect(self._forget_key)
        key_row.addWidget(self.forget_key_button)
        key_label = QLabel("API &key")
        key_label.setBuddy(self.api_key)
        form.addRow(key_label, key_row)
        self.model = QLineEdit()
        self.model.setAccessibleName("Assistant model override")
        self.model.setPlaceholderText("Optional model identifier supported by this runtime and account")
        form.addRow("&Model", self.model)
        layout.addLayout(form)
        self.auth_hint = QLabel()
        self.auth_hint.setWordWrap(True)
        layout.addWidget(self.auth_hint)
        buttons = QHBoxLayout()
        self.connect_button = QPushButton("&Connect / check status")
        self.login_button = QPushButton("&Sign in…")
        self.connect_button.clicked.connect(self._connect)
        self.login_button.clicked.connect(self._login)
        buttons.addWidget(self.connect_button)
        buttons.addWidget(self.login_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.status = QLabel("Choose a runtime, then connect. Nothing is sent automatically.")
        self.status.setWordWrap(True)
        self.status.setAccessibleName("Assistant connection and request status")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByKeyboard | Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        context_row = QHBoxLayout()
        self.include_context = QCheckBox("&Include current settings and mesh summary")
        self.include_context.setChecked(False)
        self.include_context.setToolTip("Opt in for this session. No image pixels or files are attached by this panel.")
        self.include_context.setEnabled(self.context_provider is not None)
        self.preview_context_button = QPushButton("Pre&view context…")
        self.preview_context_button.setEnabled(self.context_provider is not None)
        self.preview_context_button.clicked.connect(self._preview_context)
        context_row.addWidget(self.include_context)
        context_row.addWidget(self.preview_context_button)
        context_row.addStretch()
        layout.addLayout(context_row)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.prompt = PromptEdit()
        self.prompt.setAccessibleName("Assistant prompt")
        self.prompt.setAccessibleDescription("Enter or Control Enter sends. Shift Enter inserts a new line.")
        self.prompt.setPlaceholderText("Example: Why is my mesh flat, and which settings should I try?\nEnter sends · Shift+Enter adds a line")
        self.prompt.setMinimumHeight(80)
        self.prompt.submit.connect(self._send)
        self.response = QPlainTextEdit()
        self.response.setAccessibleName("Assistant response")
        self.response.setReadOnly(True)
        self.response.setPlaceholderText("The response appears here. Suggestions never change your project automatically.")
        self.response.setMinimumHeight(100)
        self.splitter.addWidget(self.prompt)
        self.splitter.addWidget(self.response)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        layout.addWidget(self.splitter, 1)
        actions = QHBoxLayout()
        self.send_button = QPushButton("&Send")
        self.cancel_button = QPushButton("C&ancel request")
        self.clear_button = QPushButton("C&lear conversation")
        self.validate_button = QPushButton("Check &JSON proposal")
        self.send_button.clicked.connect(self._send)
        self.cancel_button.clicked.connect(self.runtime.cancel)
        self.clear_button.clicked.connect(self._clear)
        self.validate_button.clicked.connect(self._check_proposal)
        for button in (self.send_button, self.cancel_button, self.clear_button, self.validate_button):
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)
        self.provider.currentIndexChanged.connect(self._provider_changed)
        self.auth.currentIndexChanged.connect(self._auth_changed)
        self.api_key.textEdited.connect(self._key_edited)
        shortcut = QShortcut(QKeySequence("Escape"), self)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self._cancel_if_busy)
        self._shortcuts = [shortcut]
        encoded = self._preferences.get("splitter", "")
        if isinstance(encoded, str) and encoded:
            try:
                self.splitter.restoreState(QByteArray(base64.b64decode(encoded, validate=True)))
            except ValueError:
                get_logger().error("Assistant splitter state was invalid; using the default layout")

    def _load_preferences(self):
        try:
            if self.settings_path.is_file() and self.settings_path.stat().st_size <= 65_536:
                result = json.loads(self.settings_path.read_text(encoding="utf-8"))
                if isinstance(result, dict):
                    providers = result.get("providers", {})
                    result["providers"] = {
                        name: {field: value[field] for field in ("executable", "auth_mode", "model")
                               if isinstance(value.get(field), str)}
                        for name, value in providers.items()
                        if name in PROVIDERS and isinstance(value, dict)
                    } if isinstance(providers, dict) else {}
                    return result
        except (OSError, ValueError):
            get_logger().error("Assistant preferences could not be read; using defaults")
        return {}

    def _remember_provider(self):
        if self._last_provider:
            providers = self._preferences.setdefault("providers", {})
            if not isinstance(providers, dict):
                providers = self._preferences["providers"] = {}
            providers[self._last_provider] = {"executable": self.runtime_path.text(),
                                             "auth_mode": self.auth.currentData(), "model": self.model.text()}

    def _load_provider(self):
        provider = self.provider.currentData()
        providers = self._preferences.get("providers", {})
        saved = providers.get(provider, {}) if isinstance(providers, dict) else {}
        if not isinstance(saved, dict):
            saved = {}
        self.runtime_path.setText(str(saved.get("executable", shutil.which(provider) or "")))
        self.auth.setCurrentIndex(max(0, self.auth.findData(saved.get("auth_mode", "oauth"))))
        self.model.setText(str(saved.get("model", "gpt-5.6-terra" if provider == "codex" else "")))
        self._last_provider = provider
        self._auth_changed()

    def _provider_changed(self):
        self._remember_provider()
        self.runtime.shutdown()
        self.api_key.clear()
        self._load_provider()
        self._save_preferences()

    def _auth_changed(self):
        key_mode = self.auth.currentData() == "api_key"
        sdk_mode = self.provider.currentData() == "agy" and key_mode
        self.api_key.setEnabled(key_mode and not self.is_busy)
        self.forget_key_button.setEnabled(key_mode)
        self.login_button.setEnabled(not key_mode and not self.is_busy)
        self.runtime_path.setEnabled(not sdk_mode and not self.is_busy)
        self.browse_button.setEnabled(not sdk_mode and not self.is_busy)
        if sdk_mode:
            self.auth_hint.setText("Antigravity API mode uses the optional SDK and its bundled runtime. "
                                   "Use a Gemini API model ID, or leave Model empty for the SDK default.")
        elif self.provider.currentData() == "agy":
            self.auth_hint.setText("Antigravity account mode uses the selected CLI. An older CLI can open "
                                   "its native sign-in window; safe in-app requests require streaming support.")
        else:
            self.auth_hint.setText("Account sign-in is managed by the provider's runtime. EdgeMesh retains API keys only for this session.")

    def _key_edited(self):
        self.runtime.shutdown()
        self.status.setText("API key changed. Connect again to use this key.")

    def _save_preferences(self):
        self._remember_provider()
        self._preferences["provider"] = self.provider.currentData()
        self._preferences["splitter"] = base64.b64encode(bytes(self.splitter.saveState())).decode()
        # Persist an allowlist only. Prompts, replies, context, and keys stay in memory.
        allowed = {name: self._preferences[name] for name in (
            "provider", "providers", "splitter", "last_runtime_dir", "context_geometry") if name in self._preferences}
        try:
            atomic_write(self.settings_path, json.dumps(allowed, indent=2).encode())
        except OSError:
            self._show_error("Assistant preferences could not be saved.")
            get_logger().error("Assistant preferences could not be saved")

    def _browse(self):
        directory = self._preferences.get("last_runtime_dir", "") or str(Path(self.runtime_path.text()).parent)
        path, _ = QFileDialog.getOpenFileName(self, "Choose assistant runtime", directory,
                                              "Runtime executables (*.exe *.cmd *.bat);;All files (*)")
        if path:
            self.runtime_path.setText(path)
            self._preferences["last_runtime_dir"] = str(Path(path).parent)
            self._save_preferences()

    def _config(self):
        return RuntimeConfig(self.provider.currentData(), self.runtime_path.text().strip(),
                             self.auth.currentData(), self.model.text().strip())

    def _connect(self):
        self._save_preferences()
        try:
            self.runtime.connect_runtime(self._config(), self.api_key.text())
        except (ValueError, OSError):
            # Values and exception text could include a pasted credential in a path.
            get_logger().error("Assistant connection setup failed")
            self._show_error("Connection setup failed. Check the runtime path and supply a key if using API authentication.")
        self._update_controls()

    def _login(self):
        if self.runtime.config != self._config() or not self.runtime._program:
            self._connect()
            self.status.setText("Runtime check started. Press Sign in again when it finishes.")
            return
        try:
            self.runtime.login()
        except (ValueError, OSError) as exc:
            get_logger().error("Assistant sign-in could not start")
            self._show_error(str(exc))

    def _send(self):
        if self.is_busy:
            return
        if self.runtime.config != self._config():
            self._show_error("The runtime settings changed. Connect again before sending.")
            return
        try:
            context = self.context_provider() if self.include_context.isChecked() and self.context_provider else None
            self.runtime.send(self.prompt.toPlainText(), context)
        except Exception:
            get_logger().error("Assistant request could not start")
            self._show_error("Request could not start. Check connection, sign-in, prompt length, and project context.")
            return
        self._response = ""
        self.response.clear()
        self._update_controls()

    def _append_output(self, text):
        self._response += text
        self.response.moveCursor(QTextCursor.MoveOperation.End)
        self.response.insertPlainText(text)
        self.response.ensureCursorVisible()

    def _show_error(self, message):
        get_logger().error("Assistant panel reported an error; private details omitted")
        self.status.setText(message)
        self.status.setAccessibleDescription("Assistant error: " + message)

    def _update_controls(self, *_):
        busy = self.is_busy
        for widget in (self.provider, self.runtime_path, self.browse_button, self.auth, self.model, self.connect_button):
            widget.setEnabled(not busy)
        self.send_button.setEnabled(self.runtime.ready and not busy)
        self.cancel_button.setEnabled(busy)
        self.clear_button.setEnabled(not busy)
        self.validate_button.setEnabled(not busy)
        self._auth_changed()

    def _open_login(self, url):
        if safe_login_url(url, self.provider.currentData()) and QDesktopServices.openUrl(QUrl(url)):
            return
        get_logger().error("Assistant browser sign-in could not be opened")
        self._show_error("The sign-in page could not be opened. Try the runtime's native sign-in.")

    def _cancel_if_busy(self):
        if self.is_busy:
            self.runtime.cancel()

    def _forget_key(self):
        self.api_key.clear()
        self.runtime.shutdown()
        self.status.setText("The session key was forgotten. Reconnect to continue.")

    def _clear(self):
        self.prompt.clear()
        self.response.clear()
        self._response = ""
        self.prompt.setFocus()

    def _check_proposal(self):
        try:
            proposal = validate_proposal(self._response)
        except ValueError as exc:
            get_logger().error("Assistant proposal validation failed")
            self._show_error(str(exc))
            return
        self.status.setText(f"Valid advisory JSON with {len(proposal['settings'])} settings. No settings have been applied.")

    def _preview_context(self):
        try:
            text = json.dumps(self.context_provider(), indent=2, ensure_ascii=False, allow_nan=False)
            if len(text.encode()) > 64_000:
                raise ValueError("Context is too large")
        except Exception:
            get_logger().error("Assistant context preview failed")
            self._show_error("The current context could not be previewed.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Context sent only when Include context is checked")
        dialog.resize(640, 480)
        layout = QVBoxLayout(dialog)
        contents = QPlainTextEdit(text)
        contents.setReadOnly(True)
        contents.setAccessibleName("Project context preview")
        layout.addWidget(contents)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        geometry = self._preferences.get("context_geometry")
        if isinstance(geometry, str):
            try:
                dialog.restoreGeometry(QByteArray(base64.b64decode(geometry, validate=True)))
            except ValueError:
                get_logger().error("Assistant context preview geometry was invalid")
        dialog.exec()
        self._preferences["context_geometry"] = base64.b64encode(bytes(dialog.saveGeometry())).decode()
        self._save_preferences()

    def shutdown(self):
        self._save_preferences()
        self.runtime.shutdown()
        self.api_key.clear()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
