"""Opt-in real Windows window/event-loop test, isolated from saved user data."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("EDGEMESH_TEST_NATIVE_GUI") == "1",
    "Set EDGEMESH_TEST_NATIVE_GUI=1 on an interactive Windows desktop",
)
class NativeStartupTests(unittest.TestCase):
    def test_console_launcher_displays_window_and_survives_delayed_processing(self):
        code = textwrap.dedent("""
            import faulthandler
            import logging
            import sys
            import traceback
            faulthandler.enable()
            from PySide6.QtCore import QTimer
            from PySide6.QtWidgets import QApplication

            original_exec = QApplication.exec
            failures = []
            checked = []
            def checked_exec(application):
                def verify_window():
                    try:
                        windows = [w for w in application.topLevelWidgets()
                                   if hasattr(w, '_workspace_ready')]
                        assert len(windows) == 1
                        window = windows[0]
                        assert window.isVisible() and window.initialized
                        assert window._workspace_ready and window._product_ready
                        assert not window._error_log.toPlainText(), window._error_log.toPlainText()
                        assert not window.mask_preview.pixmap().isNull()
                        import cv2
                        from concurrent.futures import ThreadPoolExecutor
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            edges = pool.submit(cv2.Canny, window.image, 50, 150).result(timeout=10)
                        assert edges.shape == window.image.shape[:2]
                        checked.append(True)
                        print('NATIVE_STARTUP_PASS', flush=True)
                    except BaseException:
                        failures.append(traceback.format_exc())
                    finally:
                        application.closeAllWindows()
                        application.quit()
                # WMI worker failures can arrive well after initial construction.
                QTimer.singleShot(8000, verify_window)
                return original_exec()

            QApplication.exec = checked_exec
            from edgemesh_bootstrap.cli import main
            result = main([])
            logging.shutdown()
            assert checked and not failures, '\\n'.join(failures)
            assert result == 0, result
        """)
        with tempfile.TemporaryDirectory(prefix="edgemesh-native-test-") as directory:
            root = Path(directory)
            source = root / "source.ppm"
            source.write_bytes(b"P6\n20 12\n255\n" + bytes((40, 150, 230)) * 240)
            (root / "config.ini").write_text(
                "[Settings]\nlast_used_image = " + str(source) + "\n",
                encoding="utf-8",
            )
            environment = dict(os.environ, QT_QPA_PLATFORM="windows", EDGEMESH_DATA_DIR=directory)
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=ROOT, env=environment,
                capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("NATIVE_STARTUP_PASS", result.stdout)
        self.assertNotIn("Windows fatal exception", result.stderr)


if __name__ == "__main__":
    unittest.main()
