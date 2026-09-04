"""Exercise UI actions without opening model or renderer windows."""
import configparser
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# These actions do not use model inference or rendering. Keep those expensive
# dependencies out of import while testing the actual application methods.
spec = importlib.util.spec_from_file_location(
    "edgemesh_ui_under_test", Path(__file__).resolve().parents[1] / "edge_mesh.py"
)
ui = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {
    "depth_to_3d": SimpleNamespace(DepthTo3D=Mock(), model_names={}),
    "mesh_generator": SimpleNamespace(MeshGenerator=Mock()),
    "MeshTools.viewport_3d": SimpleNamespace(ThreeDViewport=Mock()),
}):
    spec.loader.exec_module(ui)


class UIRegressionTests(unittest.TestCase):
    def test_first_launch_creates_valid_config(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.ini"
            window = SimpleNamespace(CONFIG_FILE_PATH=str(target), SETTINGS=ui.MainWindowImageProcessing.SETTINGS)
            config = ui.MainWindowImageProcessing.initialize_config(window)
            self.assertTrue(config.has_section("Settings"))
            loaded = configparser.ConfigParser()
            loaded.read(target)
            self.assertTrue(loaded.has_section("Settings"))

    def test_save_without_image_reports_error(self):
        window = SimpleNamespace(processed_image=None, show_error=Mock())
        ui.MainWindowImageProcessing.save_image(window)
        window.show_error.assert_called_once()

    def test_black_image_can_be_saved_and_failed_write_is_reported(self):
        for success in (True, False):
            with self.subTest(success=success):
                window = SimpleNamespace(processed_image=np.zeros((2, 2, 3), np.uint8), show_error=Mock())
                with patch.object(ui.QFileDialog, "getSaveFileName", return_value=("test.png", "")), patch.object(ui.cv2, "imwrite", return_value=success) as write:
                    ui.MainWindowImageProcessing.save_image(window)
                write.assert_called_once()
                self.assertEqual(window.show_error.call_count, 0 if success else 1)

    def test_export_without_viewport_reports_error(self):
        window = SimpleNamespace(three_d_viewport=None, show_error=Mock())
        ui.MainWindowImageProcessing.export_mesh(window)
        window.show_error.assert_called_once()

    def test_failed_load_preserves_current_image_and_settings(self):
        image = np.ones((2, 2, 3), np.uint8)
        window = SimpleNamespace(image_path="previous.png", image=image,
            _update_config=Mock(), show_error=Mock(), display_original_image=Mock(), update_preview=Mock())
        with patch.object(ui.cv2, "imread", return_value=None):
            ui.MainWindowImageProcessing.load_image(window, "broken.png")
        self.assertEqual(window.image_path, "previous.png")
        self.assertIs(window.image, image)
        window._update_config.assert_not_called()
        window.show_error.assert_called_once()

    def test_export_uses_loaded_mesh_and_explicit_extension(self):
        viewport = SimpleNamespace(mesh=object(), export_mesh_as_obj=Mock(), export_mesh_as_stl=Mock())
        window = SimpleNamespace(three_d_viewport=viewport, show_error=Mock())
        with patch.object(ui.QFileDialog, "getSaveFileName", return_value=("mesh.STL", "OBJ Files (*.obj)")):
            ui.MainWindowImageProcessing.export_mesh(window)
        viewport.export_mesh_as_stl.assert_called_once_with("mesh.STL")
        viewport.export_mesh_as_obj.assert_not_called()
        window.show_error.assert_not_called()

    def test_grayscale_preview_does_not_write_source_directory(self):
        source = np.zeros((8, 8, 3), np.uint8)
        source[:, 4:] = [10, 100, 200]
        window = SimpleNamespace(image_path="readonly/source.png", image=source,
            grayscale_enabled=True, edge_detection_enabled=False,
            invert_colors_enabled=False, blend_amount=100,
            display_processed_image=Mock(), show_error=Mock())
        grayscale = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        with patch.object(ui.cv2, "imread", return_value=grayscale), patch.object(ui.cv2, "imwrite", side_effect=PermissionError("read only")) as write:
            ui.MainWindowImageProcessing.update_preview(window)
        write.assert_not_called()
        window.show_error.assert_not_called()
        expected = cv2.cvtColor(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        np.testing.assert_array_equal(window.processed_image, expected)


if __name__ == "__main__":
    unittest.main()
