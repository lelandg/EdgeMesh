"""Cold-start checks for Python 3.12's unsafe Windows WMI query path."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "win32" and sys.version_info[:2] == (3, 12),
                     "Python 3.12 Windows startup workaround")
class PlatformStartupTests(unittest.TestCase):
    def test_public_launch_paths_avoid_wmi_before_scientific_imports(self):
        for entry in (
            "import edge_mesh",
            "from edgemesh_bootstrap.cli import main; main(['diagnose-depth', '--help'])",
            "from depth_diagnostics import main; main(['--help'])",
            "from agent_sdk_worker import load_sdk\n"
            "try: load_sdk()\n"
            "except ImportError: pass",
        ):
            with self.subTest(entry=entry):
                code = textwrap.dedent("""
                    import atexit
                    import platform
                    import sys
                    from pathlib import Path

                    marker = Path(sys.argv[2])
                    def unsafe_query(*args):
                        marker.write_text('WMI was called', encoding='utf-8')
                        raise OSError('Injected WMI timeout')

                    platform._wmi_query = unsafe_query
                    def check_fallback():
                        platform._uname_cache = None
                        platform.machine()
                    atexit.register(check_fallback)
                    # A pre-imported platform module must also be protected.
                    sys.path.insert(0, sys.argv[1])
                    exec(sys.argv[3])
                """)
                import tempfile
                with tempfile.TemporaryDirectory(prefix="edgemesh-platform-test-") as directory:
                    marker = Path(directory) / "wmi-called.txt"
                    result = subprocess.run(
                        [sys.executable, "-c", code, str(ROOT), str(marker), entry],
                        capture_output=True, text=True, timeout=45,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertFalse(marker.exists(), "Startup called the unsafe WMI query")


if __name__ == "__main__":
    unittest.main()
