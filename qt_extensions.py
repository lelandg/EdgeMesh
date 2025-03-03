from PyQt6.QtCore import Qt, QRect, QSize, QPoint
from PyQt6.QtWidgets import QApplication, QMainWindow, QLineEdit, QVBoxLayout, QWidget, QLayout, QSizePolicy


class ExpandableLineEdit(QLineEdit):
    def __init__(self, expanded_width=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store the initial size
        self.initial_width = self.sizeHint().width()
        if expanded_width is not None:
            self.set_expanded_width(expanded_width)
        else:
            self.set_expanded_width(self.initial_width + 100)  # Amount to expand
        print(f"Initial width: {self.initial_width}, Expanded width: {self.expanded_width}")

    def set_expanded_width(self, width):
        self.initial_width = self.sizeHint().width()
        self.expanded_width = width

    def focusInEvent(self, event):
        # Expand the line edit when it gains focus
        self.setFixedWidth(self.expanded_width)
        super().focusInEvent(event)  # Call the base class implementation

    def focusOutEvent(self, event):
        # Shrink the line edit when it loses focus
        self.setFixedWidth(self.initial_width)
        super().focusOutEvent(event)  # Call the base class implementation


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=10):
        super().__init__(parent)

        self.itemList = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.doLayout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        size += QSize(2 * self.contentsMargins().top(), 2 * self.contentsMargins().top())
        return size

    def doLayout(self, rect, testOnly=False):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            widget = item.widget()
            spaceX = self.spacing() + widget.style().layoutSpacing(
                QSizePolicy.ControlType.DefaultType, QSizePolicy.ControlType.DefaultType, Qt.Orientation.Horizontal
            )
            spaceY = self.spacing() + widget.style().layoutSpacing(
                QSizePolicy.ControlType.DefaultType, QSizePolicy.ControlType.DefaultType, Qt.Orientation.Vertical
            )
            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()

def state_to_bool(checkbox, state):
    print(f"State: {state}")
    if state == 2:
        checkbox.setChecked(True)
    elif state == 0:
        checkbox.setChecked(False)
    elif state == 3:
        print ("Not implemented yet")
    else:
        print(f"Invalid state: {state}")
        return
