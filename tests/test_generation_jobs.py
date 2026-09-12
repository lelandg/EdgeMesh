import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import threading
import unittest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication
from generation_jobs import JobController


class JobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_job(self, controller):
        loop = QEventLoop()
        controller.idle.connect(loop.quit)
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(5000)
        loop.exec()
        self.assertFalse(controller.busy)

    def test_job_runs_off_main_thread_and_returns(self):
        main = threading.get_ident()
        controller = JobController()
        results = []
        controller.succeeded.connect(results.append)
        controller.start(lambda check, progress: threading.get_ident())
        self.wait_job(controller)
        self.assertEqual(len(results), 1)
        self.assertNotEqual(results[0], main)

    def test_cancel_discards_result(self):
        entered = threading.Event()
        release = threading.Event()
        def work(check, progress):
            entered.set()
            release.wait(2)
            check()
            return 'stale'
        controller = JobController()
        results, cancelled = [], []
        controller.succeeded.connect(results.append)
        controller.cancelled.connect(lambda: cancelled.append(True))
        controller.start(work)
        self.assertTrue(entered.wait(2))
        controller.cancel()
        release.set()
        self.wait_job(controller)
        self.assertFalse(results)
        self.assertTrue(cancelled)

    def test_cancel_after_function_return_exposes_discard_for_cleanup(self):
        controller = JobController()
        discarded, results = [], []
        def work(check, progress):
            controller.cancel()
            return {'folder': 'staged'}
        controller.discarded.connect(discarded.append)
        controller.succeeded.connect(results.append)
        controller.start(work)
        self.wait_job(controller)
        self.assertEqual(discarded, [{'folder': 'staged'}])
        self.assertEqual(results, [])
