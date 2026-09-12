"""Atomic per-user UI preferences and persistent, accessible PySide6 dialogs.

Only geometry and explicitly named caller settings are stored. Text controls are
never scraped: provider credentials belong to their own credential store.
"""
from __future__ import annotations

import base64
import copy
import json
import logging
import math
from pathlib import Path
import re
import weakref

from PySide6.QtCore import QByteArray, QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFileDialog, QMainWindow,
    QPushButton, QSplitter, QWidget,
)

from user_state import UserPaths, atomic_write


MAX_SETTINGS_BYTES = 1_048_576
_SECRET_KEYS = {
    "apikey", "accesstoken", "refreshtoken", "idtoken", "token", "password",
    "passwd", "secret", "clientsecret", "authorization", "credential",
    "credentials", "privatekey", "bearer", "oauthcode", "authorizationcode",
}


def _error(message):
    """Log failure without logging settings values or secret-bearing inputs."""
    try:
        from log_utils import get_logger
        get_logger().error(message)
    except (OSError, RuntimeError):
        logging.getLogger(__name__).error(message)


def _valid_key(key):
    if not isinstance(key, str) or not key or len(key) > 256:
        raise ValueError("Settings keys must be nonempty, bounded strings")
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    parts = re.split(r"[./:\\]", key.lower())
    if normalized in _SECRET_KEYS or any(normalized.endswith(suffix) for suffix in (
        "apikey", "token", "secret", "password", "privatekey", "credential", "credentials",
    )) or any(
        re.sub(r"[^a-z0-9]", "", part) in _SECRET_KEYS for part in parts
    ):
        raise ValueError("Credential fields must not be stored in UI settings")


def _validate(value, depth=0, budget=None):
    budget = [20_000] if budget is None else budget
    budget[0] -= 1
    if depth > 16 or budget[0] < 0:
        raise ValueError("UI settings exceed structural limits")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if value.bit_length() > 64:
            raise ValueError("UI settings integer is too large")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("UI settings numbers must be finite")
        return
    if isinstance(value, str):
        if len(value) > 131_072:
            raise ValueError("UI settings string is too large")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _valid_key(key)
            _validate(item, depth + 1, budget)
        return
    if isinstance(value, list):
        for item in value:
            _validate(item, depth + 1, budget)
        return
    raise ValueError("UI settings must contain JSON data only")


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate UI settings key")
        result[key] = value
    return result


class SettingsStore:
    """Bounded JSON preferences; failed writes preserve the last accepted data."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else UserPaths.discover().root / "ui-settings.json"
        self._data = {}
        try:
            with self.path.open("rb") as stream:
                content = stream.read(MAX_SETTINGS_BYTES + 1)
            if len(content) > MAX_SETTINGS_BYTES:
                raise ValueError("UI settings file is too large")
            data = json.loads(content.decode("utf-8-sig"), object_pairs_hook=_object_pairs)
            if not isinstance(data, dict):
                raise ValueError("UI settings root must be an object")
            _validate(data)
            self._data = data
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError, ValueError, RecursionError):
            _error("Could not read UI settings; using defaults and preserving the original file")

    def get(self, key, default=None):
        """Return an independent value so callers cannot bypass atomic writes."""
        return copy.deepcopy(self._data.get(key, default))

    def set(self, key, value):
        return self.update({key: value})

    def update(self, values):
        """Merge named top-level preferences, returning False on any failure."""
        try:
            if not isinstance(values, dict):
                raise ValueError("UI settings update must be an object")
            candidate = {**self._data, **values}
            _validate(candidate)
            data = json.dumps(candidate, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
            if len(data) > MAX_SETTINGS_BYTES:
                raise ValueError("UI settings file is too large")
            if candidate == self._data and self.path.is_file():
                return True
            atomic_write(self.path, data)
            self._data = copy.deepcopy(candidate)
            return True
        except (OSError, UnicodeError, TypeError, ValueError, RecursionError):
            _error("Could not save UI settings; previous accepted settings retained")
            return False


def _encoded(value):
    return base64.b64encode(bytes(value)).decode("ascii")


def _decoded(value):
    if not isinstance(value, str) or len(value) > 131_072:
        raise ValueError("Invalid saved PySide6 state")
    return QByteArray(base64.b64decode(value, validate=True))


class DialogPersistence(QObject):
    """Persist real windows and explicit file dialogs without scraping controls.

    Give windows a stable objectName to distinguish instances of the same class.
    Unnamed windows share their class key; changing image filenames in titles do
    not produce a new setting on every open. Call install() if constructed before
    QApplication. Keep this service alive for the lifetime of the application.
    """

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._installed_app = None
        self._restored = weakref.WeakSet()
        self._busy = weakref.WeakSet()
        self._file_dialogs = {}
        self.install()

    def install(self, app=None):
        app = app or QApplication.instance()
        if app is not None and app is not self._installed_app:
            if self._installed_app is not None:
                self._installed_app.removeEventFilter(self)
            app.installEventFilter(self)
            self._installed_app = app
        return self

    @staticmethod
    def _key(window, key=None):
        return str(key or window.objectName() or f"{type(window).__module__}.{type(window).__qualname__}")

    @staticmethod
    def _persistent_window(window):
        return (isinstance(window, QWidget) and window.isWindow()
                and window.windowType() not in {
                    Qt.WindowType.ToolTip, Qt.WindowType.Popup, Qt.WindowType.SplashScreen,
                } and not window.property("edgemeshSkipPersistence"))

    def eventFilter(self, watched, event):
        try:
            if event.type() == QEvent.Type.KeyPress and isinstance(watched, QWidget):
                if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                        and (event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier)
                        == Qt.KeyboardModifier.ShiftModifier):
                    window = watched.window()
                    if isinstance(window, QDialog):
                        self._click_accept_button(window)
                        # Do not let PySide6 fall back to an unrelated auto-default
                        # action when acceptance is unavailable or disabled.
                        return True
            if self._persistent_window(watched) and watched not in self._busy:
                if event.type() == QEvent.Type.Show and watched not in self._restored:
                    self._restored.add(watched)
                    self.restore_window(watched)
                elif event.type() in (QEvent.Type.Hide, QEvent.Type.Close) and watched in self._restored:
                    self.save_window(watched)
        except (RuntimeError, TypeError, ValueError):
            _error("Could not persist a window event")
        return False

    @staticmethod
    def _click_accept_button(dialog):
        # A real enabled button click preserves validation and license gates.
        buttons = [button for button in dialog.findChildren(QPushButton)
                   if button.isVisible() and button.isEnabled() and button.window() is dialog]
        roles = {button: box.buttonRole(button)
                 for box in dialog.findChildren(QDialogButtonBox) for button in box.buttons()}
        accept_roles = {QDialogButtonBox.ButtonRole.AcceptRole, QDialogButtonBox.ButtonRole.YesRole}
        accepting = [button for button in buttons if roles.get(button) in accept_roles]
        if accepting:
            next((button for button in accepting if button.isDefault()), accepting[0]).click()
            return True
        # A standalone button has no accepting role; its owner must opt in.
        custom = [button for button in buttons if button not in roles
                  and button.property("edgemeshAcceptButton") is True]
        if custom:
            next((button for button in custom if button.isDefault()), custom[0]).click()
            return True
        return False

    def restore_window(self, window, key=None):
        self._busy.add(window)
        try:
            windows = self.settings.get("windows", {})
            saved = windows.get(self._key(window, key), {}) if isinstance(windows, dict) else {}
            if not isinstance(saved, dict):
                raise ValueError("Invalid saved window record")
            if saved.get("geometry") and not window.restoreGeometry(_decoded(saved["geometry"])):
                raise ValueError("PySide6 rejected saved window geometry")
            if isinstance(window, QMainWindow) and saved.get("state"):
                if not window.restoreState(_decoded(saved["state"]), 1):
                    raise ValueError("PySide6 rejected saved main window state")
            elif isinstance(window, QFileDialog) and saved.get("state"):
                if not window.restoreState(_decoded(saved["state"])):
                    raise ValueError("PySide6 rejected saved file dialog state")
            splitters = saved.get("splitters", {})
            if not isinstance(splitters, dict):
                raise ValueError("Invalid saved splitter settings")
            for splitter in window.findChildren(QSplitter):
                name = splitter.objectName()
                if name and splitter.window() is window and name in splitters:
                    if not splitter.restoreState(_decoded(splitters[name])):
                        raise ValueError("PySide6 rejected saved splitter state")
            self._recover_screen(window)
            self._restored.add(window)
            return True
        except (RuntimeError, TypeError, ValueError):
            _error("Could not restore a saved window layout; keeping the available layout")
            self._recover_screen(window)
            return False
        finally:
            self._busy.discard(window)

    @staticmethod
    def _recover_screen(window):
        screens = QApplication.screens()
        if not screens:
            return
        frame = window.frameGeometry()
        available = [screen.availableGeometry() for screen in screens]
        best = max(available, key=lambda rect: rect.intersected(frame).width() * rect.intersected(frame).height())
        if not best.intersects(frame):
            primary = QApplication.primaryScreen()
            best = primary.availableGeometry() if primary else available[0]
        if window.isMaximized() or window.isFullScreen():
            return
        window.resize(min(window.width(), best.width()), min(window.height(), best.height()))
        frame = window.frameGeometry()
        x = min(max(frame.x(), best.left()), max(best.left(), best.right() - frame.width() + 1))
        y = min(max(frame.y(), best.top()), max(best.top(), best.bottom() - frame.height() + 1))
        window.move(x, y)

    def save_window(self, window, key=None):
        self._busy.add(window)
        try:
            saved = {"geometry": _encoded(window.saveGeometry())}
            if isinstance(window, QMainWindow):
                saved["state"] = _encoded(window.saveState(1))
            elif isinstance(window, QFileDialog):
                saved["state"] = _encoded(window.saveState())
            splitters = {splitter.objectName(): _encoded(splitter.saveState())
                         for splitter in window.findChildren(QSplitter)
                         if splitter.objectName() and splitter.window() is window}
            if splitters:
                saved["splitters"] = splitters
            windows = self.settings.get("windows", {})
            if not isinstance(windows, dict):
                windows = {}
            windows[self._key(window, key)] = saved
            return self.settings.set("windows", windows)
        except (RuntimeError, TypeError, ValueError):
            _error("Could not save a window layout")
            return False
        finally:
            self._busy.discard(window)

    def _dialog(self, parent, caption, directory_key, mode, name_filter, initial_directory):
        cache_key = (mode, directory_key)
        dialog = self._file_dialogs.get(cache_key)
        try:
            if dialog is not None:
                dialog.setWindowTitle(caption)
        except RuntimeError:
            dialog = None  # Its temporary parent was already destroyed by PySide6.
        if dialog is None:
            dialog = QFileDialog(parent, caption)
            # Prefer the operating system picker (including Windows Explorer's
            # editable address bar). PySide6 supplies its widget picker when a native
            # dialog is unavailable. Set this before any other dialog properties.
            dialog.setOption(QFileDialog.Option.DontUseNativeDialog, False)
            dialog.setObjectName(f"file-dialog:{mode}:{directory_key}")
            self._file_dialogs[cache_key] = dialog
        elif dialog.parentWidget() is not parent:
            self._busy.add(dialog)
            try:
                dialog.setParent(parent, dialog.windowFlags())
            finally:
                self._busy.discard(dialog)
            self._restored.discard(dialog)
        # QFileDialog's binary state also contains a directory. Restore it
        # before applying the separately remembered, validated target below.
        # A first-ever dialog keeps PySide6's normal initial centering behavior.
        windows = self.settings.get("windows", {})
        if isinstance(windows, dict) and self._key(dialog) in windows:
            self.restore_window(dialog)
            self._restored.add(dialog)
        directories = self.settings.get("directories", {})
        previous = directories.get(directory_key, {}) if isinstance(directories, dict) else {}
        previous = previous if isinstance(previous, dict) else {}
        choices = [previous.get("directory"), initial_directory, str(Path.home())]
        for candidate in choices:
            if candidate:
                try:
                    path = Path(candidate).expanduser()
                    if path.is_file():
                        path = path.parent
                    if path.is_dir():
                        dialog.setDirectory(str(path))
                        break
                except (OSError, TypeError, ValueError):
                    _error("Could not restore a file dialog directory")
        dialog.setNameFilter(name_filter)
        if previous.get("filter") in dialog.nameFilters():
            dialog.selectNameFilter(previous["filter"])
        return dialog

    def _remember_dialog(self, dialog, directory_key, selected_path="", directory_selection=False):
        directories = self.settings.get("directories", {})
        if not isinstance(directories, dict):
            directories = {}
        directory = dialog.directory().absolutePath()
        if selected_path:
            target = Path(selected_path)
            if not target.is_absolute():
                target = Path(directory) / target
            directory = str(target if directory_selection else target.parent)
        directories[directory_key] = {
            "directory": directory, "filter": dialog.selectedNameFilter(),
        }
        self.settings.set("directories", directories)
        self.save_window(dialog)

    def open_file(self, parent, caption, directory_key, filter="All files (*)", initial_directory=None):
        dialog = self._dialog(parent, caption, directory_key, "open", filter, initial_directory)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.selectFile("")
        selected = ""
        try:
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            files = dialog.selectedFiles()
            selected = files[0] if accepted and files else ""
            return selected, dialog.selectedNameFilter()
        finally:
            self._remember_dialog(dialog, directory_key, selected_path=selected)

    def save_file(self, parent, caption, directory_key, filter="All files (*)", initial_directory=None,
                  suggested_name="", default_suffix: str | None = None):
        """Choose a target; PySide6 adds the optional suffix only when one is absent."""
        dialog = self._dialog(parent, caption, directory_key, "save", filter, initial_directory)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setDefaultSuffix(default_suffix or "")
        dialog.selectFile(suggested_name)
        selected = ""
        try:
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            files = dialog.selectedFiles()
            selected = files[0] if accepted and files else ""
            return selected, dialog.selectedNameFilter()
        finally:
            self._remember_dialog(dialog, directory_key, selected_path=selected)

    def choose_directory(self, parent, caption, directory_key, initial_directory=None):
        dialog = self._dialog(parent, caption, directory_key, "directory", "Directories (*)", initial_directory)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        selected = ""
        try:
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            files = dialog.selectedFiles()
            selected = files[0] if accepted and files else ""
            return selected
        finally:
            self._remember_dialog(dialog, directory_key, selected_path=selected, directory_selection=True)


def get_dialog_service(parent=None):
    """Find the owning window's service or retain one per-user app fallback."""
    current = parent
    while current is not None:
        service = getattr(current, "dialogs", None)
        if isinstance(service, DialogPersistence):
            return service
        current = current.parent() if isinstance(current, QObject) else None
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QApplication is required for file dialogs")
    service = getattr(app, "_edgemesh_dialog_persistence", None)
    if service is None:
        service = DialogPersistence(SettingsStore(), app)
        app._edgemesh_dialog_persistence = service
    return service
