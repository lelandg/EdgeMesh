"""A readable, current-session history browser for image-to-mesh settings."""

from pathlib import PureWindowsPath

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QLabel, QPlainTextEdit, QPushButton,
    QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)


SETTING_LABELS = {
    "model": "Depth model", "resolution": "Resolution", "depth_amount": "Depth amount",
    "depth_drop_percentage": "Background depth drop", "sensitivity": "Edge sensitivity",
    "line_thickness": "Edge line thickness", "blend_amount": "Image blend",
    "background_tolerance": "Background tolerance", "smoothing_method": "Smoothing",
    "grayscale_enabled": "Grayscale", "edge_detection_enabled": "Edge detection",
    "invert_colors_enabled": "Invert colors", "project_on_original": "Project on original",
    "flat_back_enabled": "Flat back", "drop_background_enabled": "Drop background",
    "use_processed_image_enabled": "Use processed image", "use_selected_color": "Use selected color",
    "current_selected_color": "Selected RGB color",
}


def _source_name(path):
    # PureWindowsPath accepts both separators, including sessions made on Linux.
    return PureWindowsPath(path).name or "Untitled image"


def _value(value):
    if value is None:
        return "Not set"
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(part) for part in value)
    return str(value)


def _operation(summary):
    if summary["index"] == 0:
        return "First retained state"
    changes = []
    if summary["source_changed"]:
        changes.append("Changed image")
    if summary["mask_changed"]:
        changes.append("Updated mask" if summary["has_mask"] else "Removed mask")
    settings = summary["changed_settings"]
    if len(settings) == 1:
        key = next(iter(settings))
        changes.append(f"Changed {SETTING_LABELS.get(key, key.replace('_', ' ')).lower()}")
    elif settings:
        changes.append(f"Changed {len(settings)} settings")
    return " · ".join(changes) or "Saved state"


class HistoryPanel(QWidget):
    """Preview saved settings and request restoration without moving the cursor."""

    restore_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("historyPanel")
        self._summaries = []
        self._current_index = -1
        self._history = None
        self._history_revision = None
        layout = QVBoxLayout(self)
        self.status_label = QLabel("No saved states in this session yet.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.list = QTreeWidget()
        self.list.setObjectName("historyStates")
        self.list.setHeaderLabels(["State / change", "Source", "Depth model", "Resolution", "Depth", "Mask"])
        self.list.setRootIsDecorated(False)
        self.list.setAlternatingRowColors(True)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setUniformRowHeights(True)
        self.list.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.list.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.list.header().setStretchLastSection(False)
        self.list.setMinimumHeight(100)
        self.splitter.addWidget(self.list)

        self.details = QPlainTextEdit()
        self.details.setObjectName("historyDetails")
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("Select a saved state to compare its settings.")
        self.details.setMinimumHeight(80)
        self.splitter.addWidget(self.details)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        layout.addWidget(self.splitter, 1)

        notice = QLabel(
            "Restoring a state restores its image, settings and mask. "
            "Regenerate the mesh to apply them to geometry. "
            "New changes after a restore replace later states."
        )
        notice.setObjectName("historyRestoreNotice")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.restore_button = QPushButton("Restore selected state")
        self.restore_button.setObjectName("restoreHistoryState")
        self.restore_button.setEnabled(False)
        layout.addWidget(self.restore_button)
        self.list.itemSelectionChanged.connect(self._show_selection)
        self.list.itemDoubleClicked.connect(self._restore_selection)
        self.restore_button.clicked.connect(self._restore_selection)

    def refresh(self, history):
        """Refresh metadata, preserving inspection only while history is unchanged."""
        selected = self.list.currentItem()
        selected_index = selected.data(0, Qt.ItemDataRole.UserRole) if selected is not None else None
        if history is not self._history or history.revision != self._history_revision:
            selected_index = history.index
        self._history = history
        self._history_revision = history.revision
        self._current_index = history.index
        self._summaries = history.summaries()
        self.list.blockSignals(True)
        try:
            self.list.clear()
            for summary in self._summaries:
                index = summary["index"]
                settings = summary["settings"]
                marker = "Current · " if index == self._current_index else ""
                row = QTreeWidgetItem([
                    f"{marker}{index + 1}. {_operation(summary)}",
                    _source_name(summary["source_path"]),
                    _value(settings.get("model")),
                    _value(settings.get("resolution")),
                    _value(settings.get("depth_amount")),
                    "Saved" if summary["has_mask"] else "None",
                ])
                row.setData(0, Qt.ItemDataRole.UserRole, index)
                if index == self._current_index:
                    font = row.font(0)
                    font.setBold(True)
                    row.setFont(0, font)
                self.list.addTopLevelItem(row)
            if self._summaries:
                if selected_index is None or not 0 <= selected_index < len(self._summaries):
                    selected_index = self._current_index
                self.list.setCurrentItem(self.list.topLevelItem(selected_index))
        finally:
            self.list.blockSignals(False)
        count = len(self._summaries)
        self.status_label.setText(
            f"Current session: {count} saved {'state' if count == 1 else 'states'} "
            f"(up to {history.limit}). Save a session to keep its current settings and mask."
            if count else "No saved states in this session yet. Load an image to begin."
        )
        self._show_selection()

    def _show_selection(self):
        item = self.list.currentItem()
        if item is None:
            self.details.clear()
            self.restore_button.setEnabled(False)
            return
        index = item.data(0, Qt.ItemDataRole.UserRole)
        summary = self._summaries[index]
        self.restore_button.setEnabled(index != self._current_index)
        lines = [f"State {index + 1}: {_operation(summary)}",
                 f"Image: {_source_name(summary['source_path'])}"]
        shape = summary["mask_shape"]
        lines.append(f"Mask: {shape[1]} × {shape[0]} pixels" if shape else "Mask: None")
        if index == self._current_index:
            lines.append("This is the current state.")
        lines.extend(["", "Changes from previous retained state"])
        if index == 0:
            lines.append("This is the earliest state still in history.")
        else:
            if summary["source_changed"]:
                previous = self._summaries[index - 1]
                lines.append(f"Image: {_source_name(previous['source_path'])} → {_source_name(summary['source_path'])}")
            if summary["mask_changed"]:
                lines.append("Mask updated." if summary["has_mask"] else "Mask removed.")
            for key, change in summary["changed_settings"].items():
                label = SETTING_LABELS.get(key, key.replace("_", " ").capitalize())
                lines.append(f"{label}: {_value(change['before'])} → {_value(change['after'])}")
            if not (summary["source_changed"] or summary["mask_changed"] or summary["changed_settings"]):
                lines.append("No image, mask or settings changes.")
        lines.extend(["", "Saved settings"])
        for key, value in sorted(summary["settings"].items()):
            label = SETTING_LABELS.get(key, key.replace("_", " ").capitalize())
            lines.append(f"{label}: {_value(value)}")
        self.details.setPlainText("\n".join(lines))

    def _restore_selection(self, *_args):
        item = self.list.currentItem()
        if item is not None:
            index = item.data(0, Qt.ItemDataRole.UserRole)
            if index != self._current_index:
                self.restore_requested.emit(index)
