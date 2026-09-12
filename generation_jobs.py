"""Cooperative Qt jobs; workers never read or mutate widgets."""
import threading
import traceback
from PySide6.QtCore import QObject, QThread, Signal, Slot


class JobCancelled(Exception):
    pass


class Cancellation:
    def __init__(self):
        self.event = threading.Event()

    def cancel(self):
        self.event.set()

    def check(self):
        if self.event.is_set():
            raise JobCancelled('Operation cancelled.')


class _Worker(QObject):
    progress = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    discarded = Signal(object)
    finished = Signal()

    def __init__(self, function, cancellation):
        super().__init__()
        self.function = function
        self.cancellation = cancellation

    @Slot()
    def run(self):
        try:
            self.cancellation.check()
            result = self.function(self.cancellation.check, self.progress.emit)
            if self.cancellation.event.is_set():
                self.discarded.emit(result)
                raise JobCancelled('Operation cancelled.')
            self.succeeded.emit(result)
        except JobCancelled:
            self.cancelled.emit()
        except Exception:
            if self.cancellation.event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(traceback.format_exc())
        finally:
            self.finished.emit()


class JobController(QObject):
    progress = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    discarded = Signal(object)
    idle = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.worker = None
        self.token = None

    @property
    def busy(self):
        return self.thread is not None

    def start(self, function):
        if self.busy:
            raise RuntimeError('Wait for the current job to finish cancelling.')
        self.token = Cancellation()
        self.thread = QThread(self)
        self.worker = _Worker(function, self.token)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress)
        self.worker.succeeded.connect(self.succeeded)
        self.worker.failed.connect(self.failed)
        self.worker.cancelled.connect(self.cancelled)
        self.worker.discarded.connect(self.discarded)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._finished)
        self.thread.start()

    def cancel(self):
        if self.token:
            self.token.cancel()

    @Slot()
    def _finished(self):
        self.thread.deleteLater()
        self.thread = None
        self.worker = None
        self.token = None
        self.idle.emit()
