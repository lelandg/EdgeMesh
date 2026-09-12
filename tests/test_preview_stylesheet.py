"""Capture Qt's stylesheet diagnostics during real preview initialization."""

import unittest

from PySide6.QtCore import qInstallMessageHandler

import test_workflow_ui as fixtures


class PreviewStylesheetTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.WorkflowUITests.setUpClass.__func__)
    tearDown = fixtures.WorkflowUITests.tearDown

    def setUp(self):
        self.qt_messages = []
        previous_handler = qInstallMessageHandler(
            lambda message_type, context, message: self.qt_messages.append(message)
        )
        self.addCleanup(qInstallMessageHandler, previous_handler)
        fixtures.WorkflowUITests.setUp(self)

    def test_preview_initialization_has_no_stylesheet_parse_warnings(self):
        warnings = [
            message for message in self.qt_messages
            if "Could not parse stylesheet" in message
        ]
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
