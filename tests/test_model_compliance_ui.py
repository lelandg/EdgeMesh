"""License display follows accepted evidence, independently of model selection."""
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QPushButton

from model_compliance_ui import ModelComplianceMixin, ModelLicenseDialog
from model_licensing import policy_for


class ComplianceHarness(ModelComplianceMixin):
    def __init__(self, root):
        self.paths = SimpleNamespace(root=root)
        self._mesh = SimpleNamespace(vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 1.]]),
                                     triangles=np.array([[0, 1, 2]]))
        self.three_d_viewport = SimpleNamespace(set_provenance_status=Mock(), save_view_state=lambda: {})
        self.depth_method_dropdown = QComboBox()
        self.depth_method_dropdown.addItems(['depth_anything_v2', 'depth_pro'])
        self.process_button = QPushButton()
        self.generate_mesh_button = QPushButton()
        self.model_license_status = QLabel()
        self._subject_mask = None
        self._subject_mask_info = {}
        self._init_compliance()

    def _current_mesh(self):
        return self._mesh


class ComplianceUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        logger_patch = patch('log_utils.get_logger', return_value=logging.getLogger('model_licensing'))
        logger_patch.start()
        self.addCleanup(logger_patch.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.window = ComplianceHarness(self.root)
        self.identity = {'model_id': 'depth-anything/Depth-Anything-V2-Large-hf', 'revision': 'a' * 40}

    def tearDown(self):
        self.directory.cleanup()

    def test_acknowledgment_is_required_before_continue(self):
        dialog = ModelLicenseDialog(policy_for(self.identity['model_id']).as_dict())
        source_url = policy_for(self.identity['model_id']).source_url
        self.assertTrue(any(source_url in label.text() and 'Official model source' in label.text()
                            for label in dialog.findChildren(QLabel)))
        self.assertFalse(dialog.continue_button.isEnabled())
        dialog.acknowledgment.setChecked(True)
        self.assertTrue(dialog.continue_button.isEnabled())
        dialog.close()

    def test_offline_preparation_skips_download_acknowledgment(self):
        self.window.download_action = Mock()
        self.window.download_action.isChecked.return_value = False
        self.window.model_store = Mock()
        self.assertTrue(self.window._ensure_model_license('depth_anything_v2'))
        self.window.model_store.consent_request.assert_not_called()

    def test_online_preparation_still_checks_license_consent(self):
        self.window.download_action = Mock()
        self.window.download_action.isChecked.return_value = True
        self.window.model_store = Mock()
        self.window.model_store.consent_request.return_value = {'consent_needed': False}
        self.assertTrue(self.window._ensure_model_license('depth_anything_v2'))
        self.window.model_store.consent_request.assert_called_once_with('depth_anything_v2')

    def test_changing_selection_does_not_relabel_accepted_mesh(self):
        self.window._seal_accepted_mesh(self.identity)
        before = self.window.three_d_viewport.set_provenance_status.call_args
        self.assertEqual(before.kwargs['usage_class'], 'noncommercial')
        self.assertTrue(before.kwargs['verified'])
        self.window.depth_method_dropdown.setCurrentIndex(1)
        self.window._refresh_license_indicators()
        self.assertEqual(self.window.three_d_viewport.set_provenance_status.call_args, before)

    def test_edited_model_identity_never_gets_verified_permissive_badge(self):
        self.window._seal_accepted_mesh(self.identity)
        edited = json.loads(json.dumps(self.window._accepted_provenance))
        edited['payload']['models'][0]['model_id'] = 'depth-anything/Depth-Anything-V2-Small-hf'
        self.window._accepted_provenance = edited
        self.window._refresh_mesh_provenance_badge()
        shown = self.window.three_d_viewport.set_provenance_status.call_args
        self.assertFalse(shown.kwargs['verified'])
        self.assertNotEqual(shown.kwargs['usage_class'], 'permissive')

    def test_imported_mask_source_remains_unverified_after_depth_inference(self):
        self.window._seal_accepted_mesh(self.identity,
            {'model_id': 'facebook/sam2.1-hiera-tiny', 'revision': 'b' * 40})
        shown = self.window.three_d_viewport.set_provenance_status.call_args
        self.assertFalse(shown.kwargs['verified'])
        self.assertEqual(shown.kwargs['usage_class'], 'noncommercial')
        self.assertIn('Input mask model unverified', shown.args[0])

    def test_malformed_imported_payload_displays_unverified(self):
        self.window._accepted_provenance = {'payload': None}
        self.window._refresh_mesh_provenance_badge()
        shown = self.window.three_d_viewport.set_provenance_status.call_args
        self.assertFalse(shown.kwargs['verified'])

    def test_export_record_hashes_exact_export_bytes(self):
        self.window._seal_accepted_mesh(self.identity)
        accepted = json.loads(json.dumps(self.window._accepted_provenance))
        target = self.root / 'mesh.obj'
        target.write_bytes(b'v 0 0 0\n')
        self.window._write_export_provenance(target)
        record = json.loads(target.with_name('mesh.obj.edgemesh.json').read_text())
        self.assertEqual(record['mesh_file_sha256'], hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertEqual(record['export_binding'], 'signed')
        self.assertEqual(record['provenance']['payload']['parameters']['export_file_sha256'], record['mesh_file_sha256'])
        self.assertEqual(self.window._accepted_provenance, accepted)
        self.assertEqual(self.window.provenance_store.verify(record['provenance'])['status'], 'verified')
        record['provenance']['payload']['parameters']['export_file_sha256'] = '0' * 64
        self.assertEqual(self.window.provenance_store.verify(record['provenance'])['status'], 'tampered')

    def test_export_cannot_upgrade_unverified_model_claims(self):
        imported = {'payload': {'models': [self.identity]}, 'signature': 'invalid'}
        self.window._accepted_provenance = imported
        target = self.root / 'mesh.obj'
        target.write_bytes(b'v 0 0 0\n')
        self.window._write_export_provenance(target)
        record = json.loads(target.with_name('mesh.obj.edgemesh.json').read_text())
        self.assertEqual(record['export_binding'], 'unverified')
        self.assertEqual(record['provenance'], imported)

    def test_export_key_failure_does_not_claim_a_signed_file_binding(self):
        self.window._seal_accepted_mesh(self.identity)
        accepted = json.loads(json.dumps(self.window._accepted_provenance))
        target = self.root / 'mesh.obj'
        target.write_bytes(b'v 0 0 0\n')
        original_key = self.window.provenance_store._key

        def unavailable_for_sealing(create=False):
            if create:
                raise OSError('Simulated key access failure during export signing')
            return original_key()

        with patch.object(self.window.provenance_store, '_key', side_effect=unavailable_for_sealing):
            with self.assertLogs('model_licensing', 'ERROR'):
                self.window._write_export_provenance(target)
        record = json.loads(target.with_name('mesh.obj.edgemesh.json').read_text())
        self.assertEqual(record['export_binding'], 'unverified')
        self.assertEqual(record['provenance']['signature'], '')
        self.assertEqual(record['provenance']['integrity'], 'unverified')
        self.assertEqual(self.window._accepted_provenance, accepted)

    def test_local_repair_preserves_unverified_parent(self):
        imported = {'payload': {'models': [self.identity]}, 'signature': 'invalid'}
        self.window._seal_mesh_repair(self.window._mesh, imported)
        self.assertEqual(self.window._accepted_provenance, imported)
        self.assertFalse(self.window.three_d_viewport.set_provenance_status.call_args.kwargs['verified'])


if __name__ == '__main__':
    unittest.main()
