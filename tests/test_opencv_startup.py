"""Exercise native OpenCV thread failure recovery through actual PySide6 startup."""

import unittest
from unittest.mock import Mock, patch

import cv2

import test_workflow_ui as fixtures


class OpenCVStartupTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.WorkflowUITests.setUpClass.__func__)
    tearDown = fixtures.WorkflowUITests.tearDown
    assert_no_workflow_error = fixtures.WorkflowUITests.assert_no_workflow_error

    def setUp(self):
        # The host's native backend can itself be broken before fault injection.
        # Keep the real pixel operations deterministic and restore healthy hosts.
        try:
            original_threads = cv2.getNumThreads()
        except cv2.error:
            original_threads = 1
        cv2.setNumThreads(1)
        self.addCleanup(cv2.setNumThreads, original_threads)

    def test_startup_recovers_when_native_parallel_runtime_is_unavailable(self):
        serial = False
        convert_color = cv2.cvtColor
        set_threads = cv2.setNumThreads
        logger = Mock()
        native_error = cv2.error("Unknown C++ exception from OpenCV code")

        def thread_count():
            if not serial:
                raise native_error
            return 1

        def configure_threads(count):
            nonlocal serial
            set_threads(count)
            serial = count == 1

        def guarded_conversion(*args, **kwargs):
            if not serial:
                raise native_error
            return convert_color(*args, **kwargs)

        with (
            patch.object(cv2, "getNumThreads", side_effect=thread_count),
            patch.object(cv2, "setNumThreads", side_effect=configure_threads) as configure,
            patch.object(cv2, "cvtColor", side_effect=guarded_conversion),
            patch("log_utils.get_logger", return_value=logger),
        ):
            fixtures.WorkflowUITests.setUp(self)
            self.assert_no_workflow_error()
            self.assertTrue(self.window.initialized)
            self.assertFalse(self.window.mask_preview.pixmap().isNull())
            configure.assert_called_once_with(1)
            logger.warning.assert_called_once()
            self.assertTrue(logger.warning.call_args.kwargs["exc_info"])

    def test_startup_preserves_a_working_parallel_runtime(self):
        with (
            patch.object(cv2, "getNumThreads", return_value=8),
            patch.object(cv2, "setNumThreads", wraps=cv2.setNumThreads) as configure,
        ):
            fixtures.WorkflowUITests.setUp(self)
            self.assert_no_workflow_error()
            configure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
