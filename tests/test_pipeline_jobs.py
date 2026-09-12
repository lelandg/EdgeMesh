"""Real worker + tensor-to-mesh integration without network or native OpenGL."""
import threading
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import trimesh
from PySide6.QtCore import QEventLoop, QTimer
import test_workflow_ui as ui


class PipelineJobTests(unittest.TestCase):
    setUpClass = classmethod(ui.WorkflowUITests.setUpClass.__func__)
    setUp = ui.WorkflowUITests.setUp
    tearDown = ui.WorkflowUITests.tearDown

    def wait_job(self):
        loop = QEventLoop()
        self.window._jobs.idle.connect(loop.quit)
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(5000)
        loop.exec()
        self.assertFalse(self.window._jobs.busy)

    def model_pair(self):
        model = Mock(return_value=SimpleNamespace(predicted_depth=torch.arange(240.).reshape(1, 12, 20)))
        processor = Mock(return_value={'pixel_values': torch.zeros(1, 3, 12, 20)})
        metadata = {'backend': 'huggingface', 'model_type': 'depth_pro', 'revision': 'a' * 40}
        self.window.model_store = SimpleNamespace(get_depth=Mock(return_value=(model, processor, metadata)))
        self.window._apply_settings({'model': 'Depth Pro', 'resolution': 0, 'smoothing_method': '(none)',
            'edge_detection_enabled': False, 'depth_amount': 1.0})
        return model

    def test_real_pipeline_stages_masked_mesh_and_retains_source(self):
        self.model_pair()
        self.window._subject_mask = np.zeros((12, 20), bool)
        self.window._subject_mask[3:9, 5:15] = True
        before = self.source.read_bytes()
        with patch.object(self.window, 'update_3d_viewport') as viewport:
            self.window._start_generation()
            self.wait_job()
        self.assertEqual(self.window._error_log.toPlainText(), '')
        result = Path(self.window.mesh_3d)
        self.assertTrue(result.is_relative_to(self.window.paths.work_dir))
        mesh = trimesh.load(result, force='mesh')
        self.assertGreater(len(mesh.faces), 0)
        self.assertLessEqual(np.ptp(mesh.vertices[:, 0]), 9.01)
        self.assertLessEqual(np.ptp(mesh.vertices[:, 1]), 5.01)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(list(self.root.glob('*.ply')), [])
        self.assertEqual(len(self.window._document().history), 1)
        viewport.assert_called_once()

    def test_setting_change_cancels_inference_and_preserves_previous_mesh(self):
        model = self.model_pair()
        entered, release = threading.Event(), threading.Event()
        def infer(**kwargs):
            entered.set()
            release.wait(3)
            return SimpleNamespace(predicted_depth=torch.arange(240.).reshape(1, 12, 20))
        model.side_effect = infer
        self.window.mesh_3d = 'previous.ply'
        with patch.object(self.window, 'update_3d_viewport') as viewport:
            self.window._start_generation()
            self.assertTrue(entered.wait(3))
            self.window.depth_amount_input.setText('2.0')
            release.set()
            self.wait_job()
        self.assertEqual(self.window.mesh_3d, 'previous.ply')
        self.assertEqual(list(self.window.paths.work_dir.iterdir()), [])
        self.assertEqual(self.window._error_log.toPlainText(), '')
        viewport.assert_not_called()

    def test_flat_depth_zero_exports_a_plane(self):
        self.model_pair()
        self.window._apply_settings({'depth_amount': 0})
        with patch.object(self.window, 'update_3d_viewport'):
            self.window._start_generation()
            self.wait_job()
        self.assertEqual(self.window._error_log.toPlainText(), '')
        mesh = trimesh.load(self.window.mesh_3d, force='mesh')
        self.assertGreater(len(mesh.faces), 0)
        self.assertAlmostEqual(float(np.ptp(mesh.vertices[:, 2])), 0)

    def test_failure_keeps_mesh_and_cleans_staging(self):
        self.model_pair().side_effect = RuntimeError('test model failure')
        self.window.mesh_3d = 'previous.ply'
        with patch.object(self.window, 'update_3d_viewport') as viewport:
            self.window._start_generation()
            self.wait_job()
        self.assertEqual(self.window.mesh_3d, 'previous.ply')
        self.assertEqual(list(self.window.paths.work_dir.iterdir()), [])
        self.assertIn('test model failure', self.window._error_log.toPlainText())
        viewport.assert_not_called()

    def test_successful_replacement_removes_old_staging(self):
        self.model_pair()
        with patch.object(self.window, 'update_3d_viewport'):
            self.window._start_generation()
            self.wait_job()
            previous = Path(self.window.mesh_3d).parent
            self.window._start_generation()
            self.wait_job()
        self.assertFalse(previous.exists())
        self.assertEqual(len(list(self.window.paths.work_dir.iterdir())), 1)

    def test_retained_mask_worker_blocks_model_mutation_without_waiting(self):
        with patch('subject_mask.has_active_jobs', return_value=True), patch.object(self.window.model_store, 'clear') as clear:
            self.window.unload_models()
        clear.assert_not_called()
        self.assertIn('Wait for the active job', self.window._error_log.toPlainText())
