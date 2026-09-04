"""Installer planning tests; subprocess is always mocked, so nothing installs."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import install_requirements


class InstallerPathTests(unittest.TestCase):
    def verify_paths(self, use_fallback):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements = root / ('MeshTools/requirements.txt' if use_fallback else 'requirements.txt')
            script = root / 'install_requirements.py'
            # Only the intended absolute requirements path exists. This models
            # invocation from a different directory, including one with decoys.
            with patch.object(install_requirements, '__file__', str(script)):
                with patch('install_requirements.os.path.exists',
                           side_effect=lambda path: os.fspath(path) == str(requirements)):
                    with patch('install_requirements.subprocess.check_call') as run:
                        with patch('builtins.print'):
                            install_requirements.install_requirements()
            calls = [call.args[0] for call in run.call_args_list]
            self.assertIn([install_requirements.sys.executable, str(root / 'install_open3d.py')], calls)
            requirement_calls = [args for args in calls if '-r' in args]
            self.assertEqual(len(requirement_calls), 1)
            self.assertEqual(requirement_calls[0][-1], str(requirements))

    def test_root_requirements_resolved_relative_to_script(self):
        self.verify_paths(use_fallback=False)

    def test_meshtools_fallback_resolved_relative_to_script(self):
        self.verify_paths(use_fallback=True)


if __name__ == '__main__':
    unittest.main()
