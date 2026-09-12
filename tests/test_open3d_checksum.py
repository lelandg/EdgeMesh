import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from install_open3d import install_open3d, verify_pinned_wheel


class Open3DChecksumTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.wheel = Path(self.directory.name) / 'open3d-test.whl'
        self.wheel.write_bytes(b'local wheel fixture, never installed')
        self.digest = hashlib.sha256(self.wheel.read_bytes()).hexdigest()
        self.environment = patch.dict('os.environ', {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_sidecar_checksum(self):
        Path(str(self.wheel) + '.sha256').write_text(self.digest + '  ' + self.wheel.name)
        self.assertEqual(verify_pinned_wheel(self.wheel), self.digest)

    def test_environment_checksum(self):
        with patch.dict('os.environ', {'EDGEMESH_OPEN3D_SHA256': self.digest.upper()}):
            self.assertEqual(verify_pinned_wheel(self.wheel), self.digest)

    def test_missing_mismatched_and_malformed_digest_fail(self):
        for expected in ('', '0' * 64, 'invalid'):
            with self.subTest(expected=expected):
                with patch.dict('os.environ', {'EDGEMESH_OPEN3D_SHA256': expected}):
                    with self.assertLogs('install_open3d', level='ERROR'), self.assertRaises(ValueError):
                        verify_pinned_wheel(self.wheel)

    def test_install_verifies_explicit_wheel_even_on_python312(self):
        for expected, valid in ((self.digest, True), ('f' * 64, False)):
            with self.subTest(valid=valid):
                with patch.dict('os.environ', {'EDGEMESH_OPEN3D_WHEEL': str(self.wheel),
                                               'EDGEMESH_OPEN3D_SHA256': expected}):
                    with patch('install_open3d.sys.version', '3.12.10'), patch('builtins.print'):
                        with patch('install_open3d.subprocess.check_call') as install:
                            if valid:
                                install_open3d()
                                self.assertEqual(install.call_count, 1)
                                self.assertEqual(install.call_args.args[0][-1], str(self.wheel))
                            else:
                                with self.assertLogs('install_open3d', level='ERROR'), self.assertRaises(ValueError):
                                    install_open3d()
                                install.assert_not_called()

    def test_cached_wheel_cannot_install_without_checksum(self):
        with patch('install_open3d.sys.version', '3.14.0'):
            with patch('install_open3d.find_pinned_wheel', return_value=str(self.wheel)):
                with patch('install_open3d.subprocess.check_call') as install, patch('builtins.print'):
                    with self.assertLogs('install_open3d', level='ERROR'), self.assertRaises(ValueError):
                        install_open3d()
                    install.assert_not_called()


if __name__ == '__main__':
    unittest.main()
