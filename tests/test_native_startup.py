"""Opt-in real Windows window/event-loop test, isolated from saved user data."""

import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]

# Layout captured from the failing preview. These blobs contain Qt window/dock
# geometry only; the image and application data paths are created by each test.
SAVED_GEOMETRY = (
    "01d9d0cb000300000000004d0000006e000006c60000044c0000004d0000008d"
    "000006c60000044c00000000020000000d700000004d0000008d000006c60000044c"
)
SAVED_STATE = (
    "000000ff00000000fd00000002000000010000014400000419fc0200000001fb0000001c"
    "0070006100720061006d006500740065007200730044006f0063006b010000004500000419"
    "0000006c00ffffff0000000300000d70000000d8fc0100000001fb0000001e006400690061"
    "0067006e006f007300740069006300730044006f0063006b010000000000000d7000000000"
    "0000000000000c280000041900000004000000040000000800000008fc0000000100000002"
    "000000010000001e0077006f0072006b0066006c006f00770054006f006f006c006200610072"
    "0100000000ffffffff0000000000000000"
)
SAVED_WINDOW_RECORD = {
    'geometry': 'AdnQywADAAAAAAAyAAAAEwAABZUAAANlAAAAMgAAADIAAAWVAAADZQAAAAAAAAAADXAAAAAyAAAAMgAABZUAAANl',
    'state': (
        'AAAA/wAAAAH9AAAAAgAAAAEAAAFEAAAB9PwCAAAAAfsAAAAcAHAAYQByAGEAbQBlAHQAZQByAHMARABvAGMAawEAAABFAAAB9AAAAGwA////AAAAAwAABWQAAADY/AEAAAAB+wAAAB4AZABpAGEAZwBuAG8AcwB0AGkAYwBzAEQAbwBjAGsBAAAAAAAABWQAAABSAP///wAABBwAAAH0AAAABAAAAAQAAAAIAAAACPwAAAABAAAAAgAAAAIAAAAeAHcAbwByAGsAZgBsAG8AdwBUAG8AbwBsAGIAYQByAQAAAAD/////AAAAAAAAAAAAAAAcAHAAcgBvAGoAZQBjAHQAVABvAG8AbABiAGEAcgEAAAKn/////wAAAAAAAAAA'
    ),
    'splitters': {
        'workspaceSplitter': 'AAAA/wAAAAEAAAACAAABpAAAAooA/////wEAAAABAA==',
        'imageSplitter': 'AAAA/wAAAAEAAAACAAABGAAAASwA/////wEAAAACAA==',
    },
}


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("EDGEMESH_TEST_NATIVE_GUI") == "1",
    "Set EDGEMESH_TEST_NATIVE_GUI=1 on an interactive Windows desktop",
)
class NativeStartupTests(unittest.TestCase):
    def test_console_launcher_displays_window_and_survives_delayed_processing(self):
        self._assert_native_startup(source_launch=False)

    def test_source_launcher_restores_saved_geometry_and_docks_without_aborting(self):
        self._assert_native_startup(source_launch=True)

    def _assert_native_startup(self, *, source_launch):
        code = textwrap.dedent("""
            import faulthandler
            import logging
            import sys
            import traceback
            faulthandler.enable()
            if sys.argv[1] == 'source':
                from edge_mesh import main as source_main
            source_launch = sys.argv[1] == "source"
            from PySide6.QtCore import QByteArray, QTimer, Qt
            from PySide6.QtWidgets import QApplication, QMainWindow

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
                        if source_launch:
                            import configparser
                            import os
                            from pathlib import Path
                            config = configparser.ConfigParser()
                            config.read(Path(os.environ['EDGEMESH_DATA_DIR']) / 'expected-layout.ini')
                            geometry = QByteArray.fromHex(config['UI_Settings']['windowgeometry'].encode())
                            expected = QMainWindow()
                            expected.setMinimumSize(window.minimumSize())
                            assert expected.restoreGeometry(geometry)
                            assert window.normalGeometry().size() == expected.normalGeometry().size(), (
                                window.normalGeometry(), expected.normalGeometry())
                            assert window.isMaximized() == expected.isMaximized()
                            assert window.dockWidgetArea(window.controls_dock) == Qt.RightDockWidgetArea
                            assert window.controls_dock.isVisible()
                            expected.close()
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
            if sys.argv[1] == 'source':
                sys.argv = ['edge_mesh.py']
                try:
                    source_main(len(sys.argv), sys.argv)
                except SystemExit as exit_result:
                    result = exit_result.code
                else:
                    raise AssertionError('Source launcher must exit through its real main function')
            else:
                from edgemesh_bootstrap.cli import main
                result = main([])
            logging.shutdown()
            assert checked and not failures, '\\n'.join(failures)
            assert result == 0, result
        """)
        with tempfile.TemporaryDirectory(prefix="edgemesh-native-test-") as directory:
            root = Path(directory)
            if source_launch:
                source = root / "source.png"
                shutil.copyfile(ROOT / 'Images' / 'example.png', source)
            else:
                source = root / "source.ppm"
                source.write_bytes(b"P6\n20 12\n255\n" + bytes((40, 150, 230)) * 240)
            saved_layout = (
                "\n[UI_Settings]\nwindowgeometry = " + SAVED_GEOMETRY
                + "\nwindowstate = " + SAVED_STATE + "\n"
                + textwrap.dedent("""
                    invert_colors = False
                    grayscale = False
                    drop_background = False
                    background_tolerance = 10.0
                    resolution = 700
                    edge_detection = True
                    sensitivity = 150
                    line_thickness = 1
                    project_on_original = True
                    use_processed_image = False
                    depth_amount = 1.0
                    flat_back = False
                    depth_drop_percentage = 0.0
                    blend_amount = 100
                    model = Depth Pro
                    smoothing_method = anisotropic
                    use_selected_color = False
                    selected_color = [30, 30, 30]

                    [Workflow]
                    mesh_health = False
                    allow_downloads = False

                    [Workspace]
                    workspace_splitter = 000000ff0000000100000002000001a40000028a00ffffffff010000000100
                    image_splitter = 000000ff0000000100000002000001180000012c00ffffffff010000000200
                    mask_view_mode = Overlay
                    mask_opacity = 40
                    parameter_tab = 0
                    active_tab = 0
                """)
                if source_launch else ""
            )
            (root / "config.ini").write_text(
                "[Settings]\nlast_used_image = " + str(source) + "\n" + saved_layout,
                encoding="utf-8",
            )
            if source_launch:
                shutil.copyfile(root / "config.ini", root / "expected-layout.ini")
                # Both persistence stores existed in the reported failure. The
                # JSON record used to restore the main window again on Show.
                (root / 'ui-settings.json').write_text(
                    json.dumps({'windows': {'edgeMeshMainWindow': SAVED_WINDOW_RECORD}}),
                    encoding='utf-8',
                )
            environment = dict(os.environ, QT_QPA_PLATFORM="windows", EDGEMESH_DATA_DIR=directory)
            result = subprocess.run(
                [sys.executable, "-c", code, 'source' if source_launch else 'console'],
                cwd=ROOT, env=environment,
                capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("NATIVE_STARTUP_PASS", result.stdout)
        self.assertNotIn("Windows fatal exception", result.stderr)


if __name__ == "__main__":
    unittest.main()
