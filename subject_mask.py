"""Optional SAM2 prompts and manual foreground masks on an unchanged BGR image."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QRectF, QPointF
from PySide6.QtGui import (
    QColor,
    QImage,
    QImageReader,
    QImageWriter,
    QKeySequence,
    QPainter,
    QPen,
    QShortcut,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from model_store import ModelPreparationCancelled
from ui_persistence import get_dialog_service


def _preference_store(widget):
    """Use the application's explicit nonsecret settings, including nested dialogs."""
    owner = widget.parent()
    while owner is not None:
        settings = getattr(owner, "ui_settings", None)
        if settings is not None:
            return settings
        owner = owner.parent()
    return get_dialog_service(widget).settings


def _buddy_label(text, widget):
    label = QLabel(text)
    label.setBuddy(widget)
    if not widget.accessibleName():
        widget.setAccessibleName(text.replace("&", ""))
    return label


_ACTIVE_THREADS = set()  # Keep cancelled workers alive until their current stage ends.


def has_active_jobs():
    """GUI shutdown guard: defer app exit until cancelled native workers finish."""
    return bool(_ACTIVE_THREADS)


def cancel_active_jobs():
    """Request cooperative cancellation; the owner must poll has_active_jobs."""
    for thread in tuple(_ACTIVE_THREADS):
        thread.cancelled.set()


def _logger():
    from log_utils import get_logger

    try:
        return get_logger(__name__)
    except OSError:
        return logging.getLogger(__name__)


def image_rect(image_shape, view_width, view_height):
    """Letterboxed rectangle (x, y, width, height), without crop/distortion."""
    height, width = image_shape[:2]
    if min(height, width, view_width, view_height) <= 0:
        raise ValueError("Image and preview dimensions must be positive.")
    scale = min(view_width / width, view_height / height)
    return (
        (view_width - width * scale) / 2,
        (view_height - height * scale) / 2,
        width * scale,
        height * scale,
    )


def source_point(position, image_shape, view_size):
    left, top, width, height = image_rect(image_shape, *view_size)
    x, y = position
    if not (left <= x < left + width and top <= y < top + height):
        return None
    return (
        min(image_shape[1] - 1, int((x - left) * image_shape[1] / width)),
        min(image_shape[0] - 1, int((y - top) * image_shape[0] / height)),
    )


def validated_mask(mask, shape):
    array = np.asarray(mask)
    if array.shape != tuple(shape[:2]) or array.ndim != 2:
        raise ValueError("Subject mask must match the original image height and width.")
    if not np.issubdtype(array.dtype, np.bool_) and (
        not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all()
    ):
        raise ValueError("Subject mask must contain finite boolean/numeric values.")
    return (array != 0).copy()


def paint_mask(mask, point, radius, foreground):
    """Mutate only a clipped circular brush patch, using original pixel units."""
    x, y = point
    radius = max(1, int(radius))
    left, right = max(0, x - radius), min(mask.shape[1], x + radius + 1)
    top, bottom = max(0, y - radius), min(mask.shape[0], y + radius + 1)
    yy, xx = np.ogrid[top:bottom, left:right]
    patch = mask[top:bottom, left:right]
    patch[(xx - x) ** 2 + (yy - y) ** 2 <= radius**2] = bool(foreground)


def mask_preview_bgr(image_bgr, mask=None, *, mode="Overlay", opacity=0.4):
    """Render a source-size preview without changing image or mask pixels."""
    image = np.asarray(image_bgr)
    if mode not in ("Overlay", "Mask", "Source"):
        raise ValueError("Mask display mode must be Overlay, Mask, or Source.")
    if not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("Mask overlay opacity must be between zero and one.")
    foreground = (
        validated_mask(mask, image.shape)
        if mask is not None
        else np.zeros(image.shape[:2], dtype=bool)
    )
    if mode == "Mask":
        return np.repeat((foreground.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
    preview = image.copy()
    if mode == "Overlay" and mask is not None:
        preview[foreground] = (
            preview[foreground].astype(float) * (1 - opacity)
            + np.array([125, 205, 30]) * opacity
        ).astype(np.uint8)
    return preview


def load_mask_png(path, image_shape):
    """Read a PNG mask, requiring source dimensions instead of resizing."""
    reader = QImageReader(str(path))
    if bytes(reader.format()).lower() != b"png":
        raise ValueError(
            "Choose a PNG mask: white keeps pixels and black removes them."
        )
    image = reader.read()
    if image.isNull():
        raise ValueError(f"Could not read mask PNG: {reader.errorString()}")
    height, width = image_shape[:2]
    if (image.width(), image.height()) != (width, height):
        raise ValueError(
            f"Mask is {image.width()} × {image.height()}; it must match the original "
            f"image ({width} × {height})."
        )
    gray = image.convertToFormat(QImage.Format.Format_Grayscale8)
    pixels = np.frombuffer(gray.constBits(), dtype=np.uint8).reshape(
        gray.height(), gray.bytesPerLine()
    )[:, : gray.width()]
    return (pixels >= 128).copy()


def save_mask_png(path, mask):
    """Save foreground as white at original resolution, with Unicode path support."""
    array = np.asarray(mask)
    pixels = np.ascontiguousarray(validated_mask(array, array.shape), dtype=np.uint8)
    pixels *= 255
    image = QImage(
        pixels.data,
        pixels.shape[1],
        pixels.shape[0],
        pixels.strides[0],
        QImage.Format.Format_Grayscale8,
    )
    writer = QImageWriter(str(path), b"png")
    if not writer.write(image):
        raise OSError(f"Could not save mask PNG: {writer.errorString()}")


@dataclass(frozen=True)
class _MaskState:
    pixels: bytes
    has_mask: bool
    points: tuple
    labels: tuple
    box: tuple | None


def infer_subject_mask(
    image_bgr,
    points,
    labels,
    box,
    model_store,
    *,
    allow_download=False,
    cancelled=None,
    device=None,
):
    """Run on a worker, returning an original-size bool mask and model identity."""
    import torch

    if not points and box is None:
        raise ValueError("Add a foreground point or a box before running SAM2.")
    if len(points) != len(labels) or any(label not in (0, 1) for label in labels):
        raise ValueError("Each prompt point needs a foreground/background label.")

    def check():
        if cancelled is not None and cancelled():
            raise ModelPreparationCancelled("Subject segmentation cancelled.")

    check()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, metadata = model_store.get_sam2(
        device, allow_download=allow_download, cancelled=cancelled
    )
    check()
    prompts = {}
    if points:
        prompts.update(input_points=[[list(points)]], input_labels=[[list(labels)]])
    if box is not None:
        prompts["input_boxes"] = [[list(box)]]
    inputs = processor(
        images=np.ascontiguousarray(image_bgr[:, :, ::-1]),
        return_tensors="pt",
        **prompts,
    )
    original_sizes = inputs["original_sizes"]
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    check()
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)
    check()
    # Installed transformers 4.56 SAM2 takes original_sizes, not SAM1's resized size.
    masks = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(), original_sizes
    )[0]
    mask = masks.detach().cpu().numpy().reshape((-1, *image_bgr.shape[:2]))[0]
    check()
    return validated_mask(mask, image_bgr.shape), metadata


class _TaskThread(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.cancelled = threading.Event()

    def run(self):
        try:
            result = self.operation(self.cancelled.is_set)
            if not self.cancelled.is_set():
                self.completed.emit(result)
        except ModelPreparationCancelled:
            pass
        except Exception as error:
            _logger().exception("Subject/model preparation failed")
            if not self.cancelled.is_set():
                self.failed.emit(str(error))


def _start_retained(thread):
    _ACTIVE_THREADS.add(thread)

    def release():
        _ACTIVE_THREADS.discard(thread)
        thread.deleteLater()

    thread.finished.connect(release)
    thread.start()


class MaskCanvas(QWidget):
    changed = Signal()
    history_changed = Signal()
    HISTORY_LIMIT = 32
    HISTORY_BYTES = 64 * 1024 * 1024

    def __init__(self, image_bgr, initial_mask=None, parent=None):
        super().__init__(parent)
        image = np.asarray(image_bgr)
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or min(image.shape[:2]) < 1
        ):
            raise ValueError("Subject editor requires a nonempty uint8 BGR image.")
        self.image = image.copy()
        self.mask = (
            validated_mask(initial_mask, image.shape)
            if initial_mask is not None
            else np.zeros(image.shape[:2], dtype=bool)
        )
        self.has_mask = initial_mask is not None
        self.points, self.labels, self.box = [], [], None
        self.mode = "Foreground point"
        self.radius = 12
        self.display_mode = "Overlay"
        self.overlay_opacity = 0.4
        self.drag_start = None
        self._undo = []
        self._redo = []
        self._pending_state = None
        self.keyboard_cursor = (image.shape[1] // 2, image.shape[0] // 2)
        self._keyboard_box_anchor = None
        self._keyboard_painting = False
        self.setObjectName("subjectMaskCanvas")
        self.setMinimumSize(320, 200)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Subject mask canvas")
        self.setAccessibleDescription(
            "Arrow keys move one image pixel; Shift plus arrows moves ten. "
            "Space applies the selected tool. For a box, press Space at each corner. "
            "Hold Space while moving arrows to paint a brush stroke."
        )

    def set_tool(self, mode):
        self._finish_edit()
        self.mode = mode
        self.update()

    def _keyboard_changed(self):
        self.changed.emit()
        self.update()

    def _keyboard_box(self):
        start, point = self._keyboard_box_anchor, self.keyboard_cursor
        self.box = (
            min(start[0], point[0]),
            min(start[1], point[1]),
            max(start[0], point[0]),
            max(start[1], point[1]),
        )
        self._keyboard_changed()

    def keyPressEvent(self, event):
        # A first unfinished gesture has no enabled dialog Undo shortcut yet.
        if event.matches(QKeySequence.StandardKey.Undo):
            self.undo()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Redo):
            self.redo()
            event.accept()
            return
        modifiers = event.modifiers()
        if modifiers & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
        ):
            super().keyPressEvent(event)
            return
        directions = {
            Qt.Key.Key_Left: (-1, 0),
            Qt.Key.Key_Right: (1, 0),
            Qt.Key.Key_Up: (0, -1),
            Qt.Key.Key_Down: (0, 1),
        }
        if event.key() in directions:
            step = 10 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1
            dx, dy = directions[event.key()]
            old = self.keyboard_cursor
            self.keyboard_cursor = (
                min(self.image.shape[1] - 1, max(0, old[0] + dx * step)),
                min(self.image.shape[0] - 1, max(0, old[1] + dy * step)),
            )
            if self._keyboard_box_anchor is not None:
                self._keyboard_box()
            elif self._keyboard_painting:
                # Fill between cursor positions, even with a small brush and Shift.
                point = self.keyboard_cursor
                steps = max(abs(point[0] - old[0]), abs(point[1] - old[1]), 1)
                for fraction in np.linspace(0, 1, steps + 1):
                    location = (
                        round(old[0] + fraction * (point[0] - old[0])),
                        round(old[1] + fraction * (point[1] - old[1])),
                    )
                    paint_mask(
                        self.mask, location, self.radius, self.mode == "Brush add"
                    )
                self._keyboard_changed()
            self.update()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space:
            event.accept()
            if event.isAutoRepeat():
                return
            if self._keyboard_box_anchor is not None:
                self._finish_edit()
                return
            self._finish_edit()
            self._pending_state = self._snapshot()
            point = self.keyboard_cursor
            if self.mode.endswith("point"):
                self.points.append(point)
                self.labels.append(1 if self.mode.startswith("Foreground") else 0)
                self._keyboard_changed()
                self._finish_edit()
            elif self.mode == "Box":
                self._keyboard_box_anchor = point
                self._keyboard_box()
            elif self.mode.startswith("Brush"):
                self._keyboard_painting = True
                paint_mask(self.mask, point, self.radius, self.mode == "Brush add")
                self.has_mask = True
                self._keyboard_changed()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat() and self._keyboard_painting:
                self._finish_edit()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event):
        self._finish_edit()
        super().focusOutEvent(event)
        self.update()

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    def _snapshot(self):
        return _MaskState(
            np.packbits(self.mask).tobytes(),
            self.has_mask,
            tuple(self.points),
            tuple(self.labels),
            self.box,
        )

    def _restore(self, state):
        self.mask = (
            np.unpackbits(
                np.frombuffer(state.pixels, dtype=np.uint8), count=self.mask.size
            )
            .reshape(self.mask.shape)
            .astype(bool)
        )
        self.has_mask = state.has_mask
        self.points, self.labels, self.box = (
            list(state.points),
            list(state.labels),
            state.box,
        )
        self.changed.emit()
        self.history_changed.emit()
        self.update()

    def _remember(self, before):
        if before == self._snapshot():
            return False
        self._undo.append(before)
        self._redo.clear()
        while self._undo and (
            len(self._undo) > self.HISTORY_LIMIT
            or sum(len(state.pixels) for state in self._undo) > self.HISTORY_BYTES
        ):
            self._undo.pop(0)
        self.history_changed.emit()
        return True

    def _finish_edit(self):
        if self._pending_state is not None:
            before, self._pending_state = self._pending_state, None
            self._remember(before)
        self.drag_start = None
        self._keyboard_box_anchor = None
        self._keyboard_painting = False

    def undo(self):
        self._finish_edit()
        if self._undo:
            self._redo.append(self._snapshot())
            self._restore(self._undo.pop())

    def redo(self):
        self._finish_edit()
        if self._redo:
            self._undo.append(self._snapshot())
            self._restore(self._redo.pop())

    def set_mask(self, mask):
        """Replace the mask as one undoable edit; keep the current prompts."""
        replacement = validated_mask(mask, self.image.shape)
        self._finish_edit()
        before = self._snapshot()
        self.mask, self.has_mask = replacement, True
        if self._remember(before):
            self.changed.emit()
            self.update()

    def clear(self):
        self._finish_edit()
        before = self._snapshot()
        self.mask[:] = False
        self.has_mask = False
        self.points.clear()
        self.labels.clear()
        self.box = None
        if self._remember(before):
            self.changed.emit()
            self.update()

    def set_display_mode(self, mode):
        if mode not in ("Overlay", "Mask", "Source"):
            raise ValueError("Mask display mode must be Overlay, Mask, or Source.")
        self.display_mode = mode
        self.update()

    def set_overlay_opacity(self, percent):
        self.overlay_opacity = min(100, max(0, percent)) / 100
        self.update()

    def _point(self, event):
        return source_point(
            (event.position().x(), event.position().y()),
            self.image.shape,
            (self.width(), self.height()),
        )

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._point(event)
        if point is None:
            return
        self._finish_edit()
        self._pending_state = self._snapshot()
        self.keyboard_cursor = point
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if self.mode.endswith("point"):
            self.points.append(point)
            self.labels.append(1 if self.mode.startswith("Foreground") else 0)
        elif self.mode == "Box":
            self.drag_start = point
            self.box = (*point, *point)
        else:
            self.drag_start = point
            paint_mask(self.mask, point, self.radius, self.mode == "Brush add")
            self.has_mask = True
        self.changed.emit()
        self.update()
        if self.mode.endswith("point"):
            self._finish_edit()

    def mouseMoveEvent(self, event):
        if self.drag_start is None or not event.buttons() & Qt.MouseButton.LeftButton:
            return
        point = self._point(event)
        if point is None:
            return
        self.keyboard_cursor = point
        if self.mode == "Box":
            self.box = (
                min(self.drag_start[0], point[0]),
                min(self.drag_start[1], point[1]),
                max(self.drag_start[0], point[0]),
                max(self.drag_start[1], point[1]),
            )
        elif self.mode.startswith("Brush"):
            start = self.drag_start
            steps = max(abs(point[0] - start[0]), abs(point[1] - start[1]), 1)
            for fraction in np.linspace(0, 1, steps + 1):
                location = (
                    round(start[0] + fraction * (point[0] - start[0])),
                    round(start[1] + fraction * (point[1] - start[1])),
                )
                paint_mask(self.mask, location, self.radius, self.mode == "Brush add")
            self.drag_start = point
        self.changed.emit()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._finish_edit()

    def paintEvent(self, event):
        rgb = mask_preview_bgr(
            self.image,
            self.mask if self.has_mask else None,
            mode=self.display_mode,
            opacity=self.overlay_opacity,
        )[:, :, ::-1].copy()
        image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        )
        target = QRectF(*image_rect(self.image.shape, self.width(), self.height()))
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202829"))
        painter.drawImage(target, image)
        scale = target.width() / self.image.shape[1]
        if self.hasFocus():
            cursor = QPointF(
                target.x() + (self.keyboard_cursor[0] + 0.5) * scale,
                target.y() + (self.keyboard_cursor[1] + 0.5) * scale,
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for color, width in (("black", 3), ("white", 1)):
                painter.setPen(QPen(QColor(color), width))
                painter.drawLine(cursor + QPointF(-8, 0), cursor + QPointF(8, 0))
                painter.drawLine(cursor + QPointF(0, -8), cursor + QPointF(0, 8))
            painter.setPen(QPen(QColor("#8cc8ff"), 2))
            painter.drawRect(self.rect().adjusted(1, 1, -2, -2))
        for (x, y), label in zip(self.points, self.labels):
            painter.setPen(QPen(QColor("white"), 1))
            painter.setBrush(QColor("#16c878" if label else "#ed6666"))
            painter.drawEllipse(
                QPointF(target.x() + x * scale, target.y() + y * scale), 5, 5
            )
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#ffd867"), 2))
            painter.drawRect(
                QRectF(
                    target.x() + x1 * scale,
                    target.y() + y1 * scale,
                    (x2 - x1) * scale,
                    (y2 - y1) * scale,
                )
            )


class SubjectMaskDialog(QDialog):
    def __init__(
        self,
        image_bgr,
        model_store,
        parent=None,
        initial_mask=None,
        allow_download=False,
    ):
        super().__init__(parent)
        self.setObjectName("subjectMaskDialog")
        self._preferences = _preference_store(self)
        self.setWindowTitle("Subject mask — preview before applying")
        self.resize(900, 760)
        self.model_store = model_store
        self.accepted_mask = None
        self.model_metadata = None
        self._thread = None
        self._closing = False
        self._revision = 0
        layout = QVBoxLayout(self)
        instructions = QLabel(
            "Add foreground/background points or drag a box, then run SAM2. Brush tools work without a model. Green marks foreground kept in the depth mesh. Accept applies the mask; Cancel preserves the previous state."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.canvas = MaskCanvas(image_bgr, initial_mask)
        layout.addWidget(_buddy_label("&Mask canvas", self.canvas))
        layout.addWidget(self.canvas, 1)
        keyboard_help = QLabel(
            "Keyboard: arrows move the cursor; Shift + arrows moves 10 pixels. "
            "Space applies a point or brush. Hold Space + arrows to paint. "
            "For a box, press Space at each corner. Ctrl+Z undoes an edit."
        )
        keyboard_help.setWordWrap(True)
        layout.addWidget(keyboard_help)
        tools = QHBoxLayout()
        self.tool = QComboBox()
        self.tool.addItems(
            ["Foreground point", "Background point", "Box", "Brush add", "Brush remove"]
        )
        self.tool.setObjectName("subjectMaskTool")
        self.tool.currentTextChanged.connect(self.canvas.set_tool)
        tools.addWidget(_buddy_label("&Tool", self.tool))
        tools.addWidget(self.tool)
        self.brush = QSpinBox()
        self.brush.setRange(1, 200)
        self.brush.setValue(12)
        self.brush.setSuffix(" px")
        self.brush.valueChanged.connect(
            lambda value: setattr(self.canvas, "radius", value)
        )
        self.brush.setObjectName("subjectMaskBrushRadius")
        tools.addWidget(_buddy_label("&Brush radius", self.brush))
        tools.addWidget(self.brush)
        clear = QPushButton("C&lear mask + prompts")
        clear.setAccessibleName("Clear mask and prompts")
        clear.clicked.connect(self.clear_mask)
        tools.addWidget(clear)
        layout.addLayout(tools)
        history = QHBoxLayout()
        self.undo_button = QPushButton("&Undo")
        self.undo_button.setAccessibleName("Undo mask edit")
        self.undo_button.setShortcut(QKeySequence("Ctrl+Z"))
        self.undo_button.setToolTip(
            "Undo the last completed edit (up to 32 changes). Ctrl+Z"
        )
        self.undo_button.clicked.connect(self.canvas.undo)
        history.addWidget(self.undo_button)
        self.redo_button = QPushButton("&Redo")
        self.redo_button.setAccessibleName("Redo mask edit")
        self.redo_button.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        self.redo_button.setToolTip("Redo an undone edit. Ctrl+Shift+Z or Ctrl+Y")
        self.redo_button.clicked.connect(self.canvas.redo)
        self.redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.redo_shortcut.activated.connect(self.canvas.redo)
        history.addWidget(self.redo_button)
        history.addStretch()
        self.import_button = QPushButton("&Import mask PNG…")
        self.import_button.setAccessibleName("Import mask PNG")
        self.import_button.setToolTip(
            "White/light pixels are kept; black/dark pixels are removed. "
            "The PNG must match the original image dimensions."
        )
        self.import_button.clicked.connect(self.import_mask)
        history.addWidget(self.import_button)
        self.export_button = QPushButton("&Export mask PNG…")
        self.export_button.setAccessibleName("Export mask PNG")
        self.export_button.clicked.connect(self.export_mask)
        history.addWidget(self.export_button)
        layout.addLayout(history)
        view = QHBoxLayout()
        self.display_mode = QComboBox()
        self.display_mode.setObjectName("subjectMaskDisplayMode")
        view.addWidget(_buddy_label("&View", self.display_mode))
        self.display_mode.addItems(["Overlay", "Mask", "Source"])
        self.display_mode.currentTextChanged.connect(self.canvas.set_display_mode)
        view.addWidget(self.display_mode)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setObjectName("subjectMaskOpacity")
        view.addWidget(_buddy_label("Overlay &opacity", self.opacity))
        self.opacity.setRange(0, 100)
        self.opacity.setValue(40)
        self.opacity.setAccessibleName("Mask overlay opacity")
        self.opacity.valueChanged.connect(self.canvas.set_overlay_opacity)
        view.addWidget(self.opacity, 1)
        self.opacity_label = QLabel("40%")
        self.opacity.valueChanged.connect(
            lambda value: self.opacity_label.setText(f"{value}%")
        )
        view.addWidget(self.opacity_label)
        self.display_mode.currentTextChanged.connect(
            lambda mode: self.opacity.setEnabled(mode == "Overlay")
        )
        layout.addLayout(view)
        self.mask_summary = QLabel()
        layout.addWidget(self.mask_summary)
        self.download = QCheckBox("Allow SAM2 model &download (optional; Apache 2.0)")
        self.download.setAccessibleName("Allow optional SAM2 model download")
        self.download.setChecked(allow_download)
        layout.addWidget(self.download)
        license_label = QLabel(
            '<a href="https://huggingface.co/facebook/sam2.1-hiera-tiny">Review SAM2 model and license</a> · Images stay local. Downloading weights contacts Hugging Face.'
        )
        license_label.setOpenExternalLinks(True)
        license_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        license_label.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(license_label)
        row = QHBoxLayout()
        self.run_button = QPushButton("Run &SAM2 preview")
        self.run_button.setAccessibleName("Run SAM2 preview")
        self.run_button.clicked.connect(self.run_segmentation)
        row.addWidget(self.run_button)
        self.cancel_job = QPushButton("Cancel in&ference")
        self.cancel_job.setAccessibleName("Cancel SAM2 inference")
        self.cancel_job.setEnabled(False)
        self.cancel_job.clicked.connect(self.cancel_inference)
        row.addWidget(self.cancel_job)
        layout.addLayout(row)
        self.status = QLabel("No model is loaded until Run SAM2 preview is requested.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("&Accept mask")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setAccessibleName(
            "Apply subject mask"
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setAccessibleName(
            "Cancel subject mask editing"
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.canvas.changed.connect(self._edited)
        self.canvas.history_changed.connect(self._sync_history)
        self._sync_history()
        self._update_summary()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setDefault(True)
        self._restore_preferences()
        self.finished.connect(self._save_preferences)

    def _restore_preferences(self):
        values = self._preferences.get("subject_mask.preferences", {})
        if not isinstance(values, dict):
            return
        for key, control in (("tool", self.tool), ("display_mode", self.display_mode)):
            value = values.get(key)
            if isinstance(value, str) and control.findText(value) >= 0:
                control.setCurrentText(value)
        for key, control in (("brush_radius", self.brush), ("opacity", self.opacity)):
            value = values.get(key)
            if type(value) is int and control.minimum() <= value <= control.maximum():
                control.setValue(value)

    def _save_preferences(self, _result=0):
        # Deliberate allowlist: never save image pixels, prompts, model artifacts,
        # or download permission as incidental editor preferences.
        self._preferences.set(
            "subject_mask.preferences",
            {
                "tool": self.tool.currentText(),
                "brush_radius": self.brush.value(),
                "display_mode": self.display_mode.currentText(),
                "opacity": self.opacity.value(),
            },
        )

    def _edited(self):
        self._revision += 1
        self.model_metadata = None
        self._update_summary()

    def _sync_history(self):
        self.undo_button.setEnabled(self.canvas.can_undo)
        self.redo_button.setEnabled(self.canvas.can_redo)
        self.redo_shortcut.setEnabled(self.canvas.can_redo)

    def _update_summary(self):
        height, width = self.canvas.mask.shape
        count = int(np.count_nonzero(self.canvas.mask))
        detail = (
            f"Keeps {count:,} pixels ({count / self.canvas.mask.size:.1%})"
            if self.canvas.has_mask
            else "No mask yet"
        )
        self.mask_summary.setText(f"{detail} · Original size {width} × {height}")
        self.export_button.setEnabled(self.canvas.has_mask)

    def clear_mask(self):
        self.canvas.clear()
        self.status.setText(
            "Mask and prompts cleared. Undo restores the previous edit."
        )

    def import_mask(self):
        path, _ = get_dialog_service(self).open_file(
            self,
            "Import original-size mask",
            "subject_mask_import",
            "PNG masks (*.png)",
        )
        if not path:
            return
        try:
            self.canvas.set_mask(load_mask_png(path, self.canvas.image.shape))
        except (ValueError, OSError) as error:
            self._show_error(str(error))
            return
        self.status.setText(
            "Mask imported. White keeps pixels; refine with brushes or accept."
        )

    def export_mask(self):
        if not self.canvas.has_mask:
            self._show_error("Create or import a mask before exporting.")
            return
        path, _ = get_dialog_service(self).save_file(
            self,
            "Export original-size mask",
            "subject_mask_export",
            "PNG masks (*.png)",
            suggested_name="subject-mask.png",
            default_suffix="png",
        )
        if not path:
            return
        try:
            save_mask_png(path, self.canvas.mask)
        except (ValueError, OSError) as error:
            self._show_error(str(error))
            return
        self.status.setText("Mask PNG exported at the original image dimensions.")

    def _show_error(self, message):
        _logger().error(message)
        self.status.setText(message)

    def run_segmentation(self):
        if self._thread is not None:
            return
        self.canvas._finish_edit()
        if not self.canvas.points and self.canvas.box is None:
            self._show_error(
                "Add a foreground point or box first, or use the brush without SAM2."
            )
            return
        if self.canvas.box is not None and (
            self.canvas.box[0] >= self.canvas.box[2]
            or self.canvas.box[1] >= self.canvas.box[3]
        ):
            self._show_error("Choose two box corners with nonzero width and height.")
            return
        revision = self._revision
        image, points, labels, box = (
            self.canvas.image.copy(),
            list(self.canvas.points),
            list(self.canvas.labels),
            self.canvas.box,
        )
        allow_download = self.download.isChecked()
        store = self.model_store
        self._thread = _TaskThread(
            lambda cancelled: (
                infer_subject_mask(
                    image,
                    points,
                    labels,
                    box,
                    store,
                    allow_download=allow_download,
                    cancelled=cancelled,
                ),
                revision,
            )
        )
        self._thread.completed.connect(self._delivered)
        self._thread.failed.connect(self._show_error)
        self._thread.finished.connect(self._finished)
        self.run_button.setEnabled(False)
        self.cancel_job.setEnabled(True)
        self.status.setText(
            "Preparing model / running inference. Cancellation takes effect after the current stage."
        )
        _start_retained(self._thread)

    def _delivered(self, payload):
        result, revision = payload
        self._received(result, revision)

    def _received(self, result, revision):
        if self._closing:
            return
        if revision != self._revision:
            self.status.setText(
                "The preview changed while SAM2 was running; its stale result was discarded. Run again for the current prompts."
            )
            return
        mask, metadata = result
        self.canvas.set_mask(mask)
        self.model_metadata = metadata
        self.status.setText("Preview ready. Refine with brushes or accept this mask.")

    def _finished(self):
        self._thread = None
        if not self._closing:
            self.run_button.setEnabled(True)
            self.cancel_job.setEnabled(False)

    def cancel_inference(self):
        self._revision += 1
        if self._thread is not None:
            self._thread.cancelled.set()
        self.status.setText(
            "Inference cancelled; any pending result will be discarded. The current mask is unchanged."
        )

    def accept(self):
        self.canvas._finish_edit()
        if not self.canvas.has_mask:
            self._show_error("Create a mask using SAM2 or a brush before accepting.")
            return
        if not self.canvas.mask.any():
            self._show_error(
                "The mask keeps no foreground pixels. Add foreground before accepting."
            )
            return
        self.accepted_mask = validated_mask(self.canvas.mask, self.canvas.image.shape)
        self._closing = True
        self.cancel_inference()
        super().accept()

    def reject(self):
        self._closing = True
        self.cancel_inference()
        super().reject()

    def closeEvent(self, event):
        self.reject()
        event.accept()


class MiDaSSetupDialog(QDialog):
    """User-facing registration of existing reviewed local MiDaS artifacts."""

    def __init__(self, model_store, parent=None):
        super().__init__(parent)
        self.setObjectName("midasSetupDialog")
        self._preferences = _preference_store(self)
        self.setWindowTitle("Register local MiDaS / DPT model")
        self.resize(660, 330)
        self._thread = None
        self._closing = False
        self.model_store = model_store
        layout = QVBoxLayout(self)
        instructions = QLabel(
            'Choose a clean local checkout of <a href="https://github.com/isl-org/MiDaS">official MiDaS source</a> and its matching pretrained checkpoint. MiDaS uses the MiDaS architecture; DPT uses DPT_Large. The source must contain hubconf.py and unchanged git metadata. Keep checkpoints outside the source checkout. Registration records the source commit and checkpoint SHA256; it does not download or install anything.'
        )
        instructions.setWordWrap(True)
        instructions.setOpenExternalLinks(True)
        instructions.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        instructions.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(instructions)
        self.model = QComboBox()
        self.model.addItems(["midas", "dpt"])
        self.model.setObjectName("midasSetupModel")
        remembered_model = self._preferences.get("midas_setup.model", "midas")
        if (
            isinstance(remembered_model, str)
            and self.model.findText(remembered_model) >= 0
        ):
            self.model.setCurrentText(remembered_model)
        layout.addWidget(_buddy_label("&Model", self.model))
        layout.addWidget(self.model)
        self.source = QLineEdit()
        self.source.setAccessibleName("MiDaS source checkout directory")
        self.source.setPlaceholderText("Clean MiDaS source directory")
        self.weights = QLineEdit()
        self.weights.setAccessibleName("MiDaS checkpoint file")
        self.weights.setPlaceholderText("Matching checkpoint file (.pt / .pth)")
        for field, label, button_text, callback in [
            (self.source, "&Source directory", "&Browse source…", self.choose_source),
            (self.weights, "Check&point file", "Choose &weights…", self.choose_weights),
        ]:
            row = QHBoxLayout()
            row.addWidget(_buddy_label(label, field))
            row.addWidget(field)
            button = QPushButton(button_text)
            button.setAccessibleName(button_text.replace("&", ""))
            button.clicked.connect(callback)
            row.addWidget(button)
            layout.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "&Verify and register"
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setAccessibleName(
            "Verify and register model"
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setAccessibleName(
            "Cancel model registration"
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setDefault(True)
        self.buttons.accepted.connect(self.register)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.finished.connect(self._save_preferences)

    def _save_preferences(self, _result=0):
        self._preferences.set("midas_setup.model", self.model.currentText())

    def choose_source(self):
        path = get_dialog_service(self).choose_directory(
            self, "Select clean MiDaS source checkout", "midas_source"
        )
        if path:
            self.source.setText(path)

    def choose_weights(self):
        path, _ = get_dialog_service(self).open_file(
            self,
            "Select matching MiDaS checkpoint",
            "midas_checkpoint",
            "Checkpoints (*.pt *.pth);;All files (*)",
        )
        if path:
            self.weights.setText(path)

    def register(self):
        if self._thread is not None:
            return
        if not self.source.text().strip() or not self.weights.text().strip():
            self._error("Choose both a source directory and a checkpoint file.")
            return
        model, source, weights = (
            self.model.currentText(),
            self.source.text().strip(),
            self.weights.text().strip(),
        )
        store = self.model_store
        self._thread = _TaskThread(
            lambda cancelled: store.register_midas(
                model, source, weights, cancelled=cancelled
            )
        )
        self._thread.completed.connect(self._registered)
        self._thread.failed.connect(self._error)
        self._thread.finished.connect(self._finished)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.status.setText("Verifying source and hashing checkpoint…")
        _start_retained(self._thread)

    def _registered(self, result):
        if not self._closing:
            super().accept()

    def _error(self, message):
        _logger().error(message)
        self.status.setText(message)

    def _finished(self):
        self._thread = None
        if not self._closing:
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    def reject(self):
        # Hashing is safe to finish; closing does not terminate the worker thread.
        self._closing = True
        if self._thread is not None:
            self._thread.cancelled.set()
        super().reject()

    def closeEvent(self, event):
        self.reject()
        event.accept()
