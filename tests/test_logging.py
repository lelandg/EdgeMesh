import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import log_utils


class LoggingTests(unittest.TestCase):
    def tearDown(self):
        for name in ('test.edgemesh', 'test.other', 'test.edgemesh.child'):
            logger = logging.getLogger(name)
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
        log_utils.the_logger = None

    def test_creates_directory_and_keeps_loggers_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / 'nested' / 'one.log'
            second = Path(folder) / 'two.log'
            logger = log_utils.setup_logger('test.edgemesh', str(first))
            other = log_utils.setup_logger('test.other', str(second))
            logger.error('first error')
            other.error('second error')
            self.assertIn('first error', first.read_text())
            self.assertNotIn('second error', first.read_text())
            self.assertIn('second error', second.read_text())
            self.assertIs(log_utils.setup_logger('test.edgemesh', str(first)), logger)
            self.assertEqual(len(logger.handlers), 1)
            self.tearDown()

    def test_child_logger_does_not_duplicate_errors_in_parent_file(self):
        with tempfile.TemporaryDirectory() as folder:
            parent_path = Path(folder) / 'parent.log'
            child_path = Path(folder) / 'child.log'
            log_utils.setup_logger('test.edgemesh', parent_path)
            logger = log_utils.setup_logger('test.edgemesh.child', child_path)
            logger.error('child-only')
            self.assertEqual(parent_path.read_text(), '')
            self.assertIn('child-only', child_path.read_text())
            self.tearDown()

    def test_default_log_is_in_user_data_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict('os.environ', {'LOCALAPPDATA': folder, 'XDG_STATE_HOME': folder}):
                logger = log_utils.get_logger('test.edgemesh')
            self.assertTrue(Path(logger.handlers[0].baseFilename).is_relative_to(folder))
            self.tearDown()


if __name__ == '__main__':
    unittest.main()
