"""Project workflow transactions with real manifests/assets and an inert viewport."""
import configparser
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from project_store import ProjectStore
from project_workflows import ProjectWorkflowMixin, _startup_project_store
from session_state import SessionDocument, SessionHistory, validate_settings
from ui_persistence import SettingsStore


class _Choice:
    def __init__(self, index=0, text='Overlay', value=40):
        self.index, self.text, self.number = index, text, value
        self.widget = None

    def currentIndex(self):
        return self.index

    def setCurrentIndex(self, index):
        self.index = index

    def currentText(self):
        return self.text

    def setCurrentText(self, text):
        self.text = text

    def value(self):
        return self.number

    def setValue(self, value):
        self.number = value

    def setCurrentWidget(self, widget):
        self.widget = widget


class _Mesh:
    def __init__(self, label):
        self.label = label
        self.vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        self.triangles = np.array([[0, 1, 2]])


class _Viewport:
    def __init__(self, mesh):
        self.mesh = mesh
        self.view_state = {'projection': 'parallel', 'distance': 4.0}
        self.hidden = False
        self.stopped = False
        self.deleted = False

    def save_view_state(self):
        return deepcopy(self.view_state)

    def restore_view_state(self, state):
        if state.get('reject'):
            return False
        self.view_state = deepcopy(state)
        return True

    def isHidden(self):
        return self.hidden

    def hide(self):
        self.hidden = True

    def setVisible(self, visible):
        self.hidden = not visible

    def shutdown(self):
        self.stopped = True

    def deleteLater(self):
        self.deleted = True


class _Workspace(ProjectWorkflowMixin):
    """Keep real project/history/settings I/O while avoiding PySide6 or GPU startup."""

    def __init__(self, root, source=None):
        self.paths = SimpleNamespace(root=root / 'profile', work_dir=root / 'work')
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.work_dir.mkdir(parents=True, exist_ok=True)
        self.ui_settings = SettingsStore(self.paths.root / 'ui-settings.json')
        self.project_store = ProjectStore(root / 'projects')
        self.model_store = Mock()
        self.config = configparser.ConfigParser(interpolation=None)
        self.config.add_section('Settings')
        self.config.add_section('Workspace')
        self._product_ready = True
        self._restoring_project = False
        self._restoring_session = False
        self._saved_project_token = None
        self._saved_project_mesh_token = None
        self._autosave_error = None
        self._autosave_timer = Mock()
        self._history_timer = Mock()
        self._history_timer.isActive.return_value = False
        self.workspace_tabs = _Choice()
        self.parameter_tabs = _Choice()
        self.mask_view_mode = _Choice()
        self.mask_opacity = _Choice()
        self.setup_page = object()
        self.project_status = Mock()
        self.mesh_placeholder = Mock()
        self.mesh_preview_layout = Mock()
        self.original_label = Mock()
        self.preview_label = Mock()
        self.parameters = {'model': 'DepthAnythingV2', 'depth_amount': 1.0, 'resolution': 64}
        self.image_path = str(source) if source is not None else None
        self.image = cv2.imread(str(source)) if source is not None else None
        self.processed_image = self.image.copy() if self.image is not None else None
        self._subject_mask = None
        self._subject_mask_info = {}
        self._subject_mask_provenance_trusted = False
        self._last_model_info = {}
        self._processing_history = []
        self._accepted_provenance = None
        self._accepted_provenance_check = {'status': 'unverified'}
        self._provenance_mesh_token = None
        self._mesh_before_repair = None
        self._mesh_before_repair_provenance = None
        self.mesh_from_2d = None
        self.mesh_3d = None
        self.depth_labels = None
        self.three_d_viewport = None
        self.session_history = SessionHistory(limit=6)
        self.created_viewports = []
        self.errors = []
        self.fail_refresh_for = None
        self._record_session()

    def _settings_snapshot(self):
        return validate_settings(self.parameters)

    def _document(self):
        metadata = deepcopy(self._last_model_info)
        if self._subject_mask_info:
            metadata['subject_mask_model'] = deepcopy(self._subject_mask_info)
        if self._current_mesh() is not None:
            metadata['accepted_mesh'] = {
                'provenance': deepcopy(self._accepted_provenance),
                'view_state': self.three_d_viewport.save_view_state(),
            }
        return SessionDocument(self.image_path or '', self._settings_snapshot(), metadata,
                               self._subject_mask, deepcopy(self._processing_history))

    def _record_session(self):
        if not self._restoring_session:
            self.session_history.record(self._document())

    def _apply_settings(self, settings):
        self.parameters.update(validate_settings(settings))
        self.processed_image = self.image.copy() if self.image is not None else None

    def _restore_document(self, document):
        image = cv2.imread(document.source_path) if document.source_path else None
        if document.source_path and image is None:
            raise ValueError('The project source could not be decoded')
        if document.mask is not None and image is not None and document.mask.shape != image.shape[:2]:
            raise ValueError('The saved mask does not match the image')
        self.image_path = document.source_path or None
        self.image = image
        self._subject_mask = None if document.mask is None else document.mask.copy()
        self._last_model_info = deepcopy(document.model_info)
        self._subject_mask_info = self._last_model_info.pop('subject_mask_model', {})
        self._subject_mask_provenance_trusted = False
        self._processing_history = deepcopy(document.history)
        self._apply_settings(document.settings)
        self._refresh_workspace()
        return True

    def _current_mesh(self):
        return self.three_d_viewport.mesh if self.three_d_viewport is not None else None

    def _model_work_active(self):
        return False

    def _mesh_file_for_project(self):
        mesh = self._current_mesh()
        if mesh is None:
            return None, None
        saved = self.project_store.current_mesh_path
        if self._saved_project_mesh_token == id(mesh) and saved is not None and saved.is_file():
            return saved, None
        path = self.paths.work_dir / 'accepted-mesh.ply'
        path.write_text(mesh.label, encoding='utf-8')
        return path, path

    def update_3d_viewport(self):
        mesh = self.mesh_from_2d
        if mesh is None:
            mesh = _Mesh(Path(self.mesh_3d).read_text(encoding='utf-8'))
        if mesh.label == 'reject-render':
            return False
        if self.three_d_viewport is None:
            self.three_d_viewport = _Viewport(mesh)
            self.created_viewports.append(self.three_d_viewport)
        else:
            self.three_d_viewport.mesh = mesh
        self.workspace_tabs.setCurrentIndex(0)
        self._refresh_workspace()
        return True

    def _refresh_workspace(self):
        if self.fail_refresh_for is not None and self.project_store.current_path == self.fail_refresh_for:
            self.fail_refresh_for = None
            raise RuntimeError('A late UI refresh failed')
        self._refresh_product_state()

    def _refresh_license_indicators(self):
        pass

    def _refresh_mesh_provenance_badge(self):
        pass

    def save_ui_settings(self):
        pass

    def display_original_image(self):
        pass

    def display_processed_image(self):
        pass

    def show_error(self, message):
        self.errors.append(message)


class HistorySourceRelocationTests(unittest.TestCase):
    def test_relocation_preserves_cursor_redo_masks_and_provenance(self):
        history = SessionHistory(limit=6)
        first = np.array([[True, False], [False, True]])
        seal = {'original_signature': 'retain-this-value'}
        history.record(SessionDocument('source.png', {'depth_amount': 1.0}, seal, first))
        history.record(SessionDocument('source.png', {'depth_amount': 2.0}, seal, ~first))
        history.record(SessionDocument('another.png', {'depth_amount': 3.0}))
        history.select(0)
        revision = history.revision
        self.assertEqual(history.relocate_source('source.png', 'managed/source.png'), 2)
        self.assertEqual(history.index, 0)
        self.assertTrue(history.can_redo)
        self.assertEqual(history.revision, revision + 1)
        np.testing.assert_array_equal(history.snapshot_at(0).mask, first)
        self.assertEqual(history.snapshot_at(0).model_info, seal)
        self.assertEqual(history.snapshot_at(1).source_path, 'managed/source.png')
        self.assertEqual(history.snapshot_at(2).source_path, 'another.png')
        self.assertEqual(history.relocate_source('source.png', 'managed/source.png'), 0)
        self.assertEqual(history.revision, revision + 1)
        with self.assertRaises(ValueError):
            history.relocate_source('managed/source.png', '')
        self.assertEqual(history.snapshot_at(0).source_path, 'managed/source.png')


class ProjectWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'Wooden Bird.png'
        image = np.zeros((12, 20, 3), np.uint8)
        image[2:10, 4:16] = [40, 150, 230]
        self.assertTrue(cv2.imwrite(str(self.source), image))
        self.window = _Workspace(self.root, self.source)
        warning_patch = patch('project_workflows._project_warning')
        self.warning = warning_patch.start()
        self.addCleanup(warning_patch.stop)

    def _candidate(self, mesh_label=None, view_state=None, model_info=None):
        source = self.root / 'Other Source.png'
        self.assertTrue(cv2.imwrite(str(source), np.full((12, 20, 3), 170, np.uint8)))
        metadata = deepcopy(model_info or {})
        mesh = None
        if mesh_label is not None:
            mesh = self.root / 'candidate.ply'
            mesh.write_text(mesh_label, encoding='utf-8')
            metadata['accepted_mesh'] = {
                'provenance': {'original_signature': 'imported-unchanged'},
                'view_state': view_state or {'projection': 'perspective', 'distance': 2.0},
            }
        document = SessionDocument(str(source), {'model': 'DepthAnythingV2', 'depth_amount': 2.5,
                                                'resolution': 96}, metadata,
                                   np.ones((12, 20), bool))
        return ProjectStore(self.root / 'other-projects').create(document, 'Other project', mesh_path=mesh)

    def _accepted_mesh(self):
        mesh = _Mesh('previous-mesh')
        self.window.mesh_from_2d = mesh
        self.window.update_3d_viewport()
        self.window.three_d_viewport.view_state['distance'] = 7.0
        self.window._accepted_provenance = {'original_signature': 'previous-unchanged'}
        self.window._mesh_before_repair = _Mesh('before-repair')
        self.window._mesh_before_repair_provenance = {'original_signature': 'before-repair'}
        self.window._subject_mask_provenance_trusted = True
        self.assertTrue(self.window._save_project_now(force=True))
        return self.window.three_d_viewport

    def test_autosave_relocates_every_history_source_and_reopens_without_original(self):
        mask = np.zeros((12, 20), bool)
        mask[2:10, 4:16] = True
        self.window.parameters['depth_amount'] = 2.0
        self.window._subject_mask = mask.copy()
        self.window._record_session()
        self.window.parameters['depth_amount'] = 3.0
        self.window._record_session()
        self.assertTrue(self.window._save_project_now(force=True))
        managed = Path(self.window.image_path)
        self.assertNotEqual(managed, self.source)
        self.assertTrue(managed.is_file())
        self.assertEqual({row['source_path'] for row in self.window.session_history.summaries()}, {str(managed)})
        self.source.unlink()
        self.assertTrue(self.window._restore_document(self.window.session_history.undo()))
        self.assertEqual(self.window.parameters['depth_amount'], 2.0)
        np.testing.assert_array_equal(self.window._subject_mask, mask)
        reopened = _Workspace(self.root)
        self.assertTrue(reopened._restore_startup_project())
        self.assertEqual(reopened.image_path, str(managed))
        self.assertEqual(reopened.parameters['depth_amount'], 3.0)
        np.testing.assert_array_equal(reopened._subject_mask, mask)
        self.assertEqual(reopened._project_title(), ' — Wooden Bird')
        self.assertEqual(reopened.errors, [])

    def test_save_flushes_pending_settings_without_turning_camera_changes_into_undo_steps(self):
        self.window.parameters['depth_amount'] = 4.0
        self.window._history_timer.isActive.return_value = True
        self.assertTrue(self.window._save_project_now(force=True))
        self.assertEqual(self.window.session_history.current.settings['depth_amount'], 4.0)
        self.assertEqual(self.window.session_history.current.source_path, self.window.image_path)
        self.window._history_timer.isActive.return_value = False
        viewport = self._accepted_mesh()
        self.window._record_session()
        before = len(self.window.session_history.summaries())
        viewport.view_state['distance'] = 9.0
        self.assertTrue(self.window._save_project_now(force=True))
        self.assertEqual(len(self.window.session_history.summaries()), before)

    def test_new_source_resets_only_after_previous_project_is_saved(self):
        self.window._processing_history = [{'operation': 'depth'}]
        self.window._last_model_info = {'model_type': 'depth_anything_v2'}
        history = self.window.session_history
        previous_store = self.window.project_store
        self.assertTrue(self.window._before_new_source())
        self.assertTrue(previous_store.current_path.is_file())
        self.assertIsNot(self.window.project_store, previous_store)
        self.assertIsNone(self.window.project_store.current_path)
        self.assertIsNot(self.window.session_history, history)
        self.assertEqual(self.window.session_history.limit, history.limit)
        self.assertIsNone(self.window.session_history.current)
        self.assertEqual(self.window._processing_history, [])
        self.assertEqual(self.window._last_model_info, {})

    def test_background_store_validation_failure_is_visible(self):
        with patch.object(self.window.project_store, 'save_current', side_effect=ValueError('Asset checksum changed')):
            self.assertFalse(self.window._save_project_now())
        self.assertEqual(self.window._autosave_error, 'Asset checksum changed')
        self.assertIn('Asset checksum changed', self.window.errors[-1])

    def test_transient_form_input_does_not_report_a_background_storage_error(self):
        with patch.object(self.window, '_document', side_effect=ValueError('Empty numeric field')):
            self.assertFalse(self.window._save_project_now())
        self.assertIsNone(self.window._autosave_error)
        self.assertEqual(self.window.errors, [])

    def test_failed_save_preserves_project_history_mesh_and_metadata(self):
        viewport = self._accepted_mesh()
        previous_store = self.window.project_store
        previous_history = self.window.session_history
        previous_provenance = deepcopy(self.window._accepted_provenance)
        with patch.object(previous_store, 'save_current', side_effect=OSError('Write denied')):
            self.assertFalse(self.window._before_new_source())
        self.assertIs(self.window.project_store, previous_store)
        self.assertIs(self.window.session_history, previous_history)
        self.assertIs(self.window.three_d_viewport, viewport)
        self.assertFalse(viewport.stopped)
        self.assertEqual(self.window._accepted_provenance, previous_provenance)
        self.assertTrue(self.window.errors)

    def test_empty_new_project_waits_for_a_source_without_errors(self):
        blank = _Workspace(self.root / 'blank')
        self.assertTrue(blank._save_project_now(force=True))
        blank.new_project()
        self.assertEqual(blank.errors, [])
        self.assertIsNone(blank.image)
        self.assertIsNone(blank.project_store.current_path)
        self.assertFalse(blank.project_store.root.exists())
        self.assertIsNone(blank.ui_settings.get('last_project'))
        self.assertEqual(blank._project_title(), '')

    def test_failed_candidate_render_keeps_the_original_viewport_and_camera(self):
        viewport = self._accepted_mesh()
        old_store, old_history = self.window.project_store, self.window.session_history
        old_image = self.window.image
        old_view = viewport.save_view_state()
        candidate = self._candidate('reject-render')
        self.assertFalse(self.window.open_project_path(candidate))
        self.assertIs(self.window.project_store, old_store)
        self.assertIs(self.window.session_history, old_history)
        self.assertIs(self.window.three_d_viewport, viewport)
        self.assertIs(self.window.image, old_image)
        self.assertEqual(viewport.save_view_state(), old_view)
        self.assertFalse(viewport.stopped)
        self.assertTrue(self.window._subject_mask_provenance_trusted)
        self.assertFalse(self.window._restoring_project)

    def test_late_failure_discards_candidate_preview_when_previous_project_has_no_mesh(self):
        self.assertTrue(self.window._save_project_now(force=True))
        old_store, old_history = self.window.project_store, self.window.session_history
        old_path = self.window.image_path
        candidate = self._candidate('candidate-mesh')
        self.window.fail_refresh_for = candidate
        self.assertFalse(self.window.open_project_path(candidate))
        self.assertIs(self.window.project_store, old_store)
        self.assertIs(self.window.session_history, old_history)
        self.assertIsNone(self.window.three_d_viewport)
        self.assertEqual(self.window.image_path, old_path)
        self.assertTrue(self.window.created_viewports[-1].stopped)
        self.assertTrue(self.window.created_viewports[-1].deleted)
        self.assertEqual(self.window.ui_settings.get('last_project'), str(old_store.current_path))

    def test_late_failure_opening_meshless_project_restores_mesh_repair_and_page(self):
        viewport = self._accepted_mesh()
        repair = self.window._mesh_before_repair
        provenance = deepcopy(self.window._accepted_provenance)
        self.window.workspace_tabs.setCurrentIndex(1)
        candidate = self._candidate()
        self.window.fail_refresh_for = candidate
        self.assertFalse(self.window.open_project_path(candidate))
        self.assertIs(self.window.three_d_viewport, viewport)
        self.assertIs(self.window._mesh_before_repair, repair)
        self.assertEqual(self.window._accepted_provenance, provenance)
        self.assertEqual(self.window.workspace_tabs.currentIndex(), 1)
        self.assertFalse(viewport.stopped)
        self.assertFalse(viewport.hidden)

    def test_partial_document_restore_does_not_mutate_prior_viewer_or_history(self):
        viewport = self._accepted_mesh()
        old_store = self.window.project_store
        old_history = self.window.session_history
        old_image = self.window.image
        candidate = self._candidate('candidate-mesh')

        def partial_restore(document):
            self.assertIsNone(self.window.three_d_viewport)
            self.assertIsNot(self.window.session_history, old_history)
            self.window.image = np.ones_like(old_image)
            self.window._processing_history.append({'operation': 'candidate'})
            self.window.session_history.record(document)
            raise RuntimeError('Restore failed after a UI mutation')

        with patch.object(self.window, '_restore_document', side_effect=partial_restore):
            self.assertFalse(self.window.open_project_path(candidate))
        self.assertIs(self.window.project_store, old_store)
        self.assertIs(self.window.session_history, old_history)
        self.assertIs(self.window.three_d_viewport, viewport)
        self.assertIs(self.window.image, old_image)
        self.assertEqual(self.window._processing_history, [])
        self.assertFalse(viewport.stopped)
        self.assertFalse(viewport.hidden)

    def test_old_preview_cleanup_failure_keeps_new_project_accepted(self):
        viewport = self._accepted_mesh()
        candidate = self._candidate('candidate-mesh')
        with patch.object(viewport, 'shutdown', side_effect=RuntimeError('Shutdown failed')):
            self.assertTrue(self.window.open_project_path(candidate))
        self.assertEqual(self.window.project_store.current_path, candidate)
        self.assertIsNot(self.window.three_d_viewport, viewport)
        self.assertEqual(self.window.three_d_viewport.mesh.label, 'candidate-mesh')
        self.assertFalse(self.window._restoring_project)
        self.warning.assert_any_call('Could not completely release an inactive project preview.')

    def test_unusable_camera_is_logged_without_losing_the_project_or_original_seal(self):
        old_viewport = self._accepted_mesh()
        candidate = self._candidate('candidate-mesh', {'reject': True})
        self.assertTrue(self.window.open_project_path(candidate))
        self.warning.assert_any_call('Could not restore the project camera; keeping the available view.')
        self.assertEqual(self.window._accepted_provenance, {'original_signature': 'imported-unchanged'})
        self.assertIsNone(self.window._mesh_before_repair)
        self.assertTrue(old_viewport.stopped)
        self.assertTrue(self.window._save_project_now(force=True))
        saved = ProjectStore(self.root / 'inspection').inspect(candidate).document.model_info['accepted_mesh']['provenance']
        self.assertEqual(saved, {'original_signature': 'imported-unchanged'})

    def test_failed_candidate_never_changes_a_model_revision_pin(self):
        old_store = self.window.project_store
        candidate = self._candidate('candidate-mesh', model_info={
            'backend': 'huggingface', 'model_type': 'depth_anything_v2', 'revision': 'a' * 40,
        })
        self.window.fail_refresh_for = candidate
        self.assertFalse(self.window.open_project_path(candidate))
        self.assertIs(self.window.project_store, old_store)
        self.window.model_store.pin_revision.assert_not_called()

    def test_successful_candidate_pins_revision_after_ui_acceptance(self):
        candidate = self._candidate('candidate-mesh', model_info={
            'backend': 'huggingface', 'model_type': 'depth_anything_v2', 'revision': 'a' * 40,
        })
        self.assertTrue(self.window.open_project_path(candidate))
        self.window.model_store.pin_revision.assert_called_once_with('depth_anything_v2', 'a' * 40)
        self.assertEqual(self.window.project_store.current_path, candidate)

    def test_preference_write_failure_does_not_undo_a_successful_project_open(self):
        viewport = self._accepted_mesh()
        candidate = self._candidate('candidate-mesh', model_info={
            'backend': 'huggingface', 'model_type': 'depth_anything_v2', 'revision': 'a' * 40,
        })
        with patch.object(self.window.ui_settings, 'set', side_effect=OSError('Preference write denied')):
            self.assertTrue(self.window._open_project_path(candidate, save_previous=False))
        self.assertEqual(self.window.project_store.current_path, candidate)
        self.assertIsNot(self.window.three_d_viewport, viewport)
        self.assertTrue(viewport.stopped)
        self.window.model_store.pin_revision.assert_called_once_with('depth_anything_v2', 'a' * 40)
        self.warning.assert_any_call('The project opened, but its location could not be remembered for startup.')

    def test_failed_candidate_pin_preserves_the_previous_workspace(self):
        viewport = self._accepted_mesh()
        old_store = self.window.project_store
        candidate = self._candidate('candidate-mesh', model_info={
            'backend': 'huggingface', 'model_type': 'depth_anything_v2', 'revision': 'a' * 40,
        })
        self.window.model_store.pin_revision.side_effect = ValueError('Revision not accepted')
        self.assertFalse(self.window.open_project_path(candidate))
        self.assertIs(self.window.project_store, old_store)
        self.assertIs(self.window.three_d_viewport, viewport)
        self.assertFalse(viewport.stopped)

    def test_startup_recovers_missing_and_non_directory_project_roots(self):
        fallback = self.root / 'profile' / 'projects'
        for configured in (str(self.root / 'missing-volume'), str(self.source), ['invalid']):
            with self.subTest(configured=configured):
                store, message = _startup_project_store(configured, fallback)
                self.assertEqual(store.root, fallback.resolve())
                self.assertIn('configured projects folder is unavailable', message)
        self.assertTrue(self.source.is_file())

    def test_startup_recovers_unwritable_project_root_without_touching_last_project(self):
        configured = self.root / 'old-projects'
        configured.mkdir()
        fallback = self.root / 'profile' / 'projects'
        previous = str(configured / 'important-project' / 'project.json')
        self.window.ui_settings.set('last_project', previous)
        with patch('project_workflows.tempfile.TemporaryFile', side_effect=PermissionError('Write denied')):
            store, message = _startup_project_store(str(configured), fallback)
        self.assertEqual(store.root, fallback.resolve())
        self.assertIn('Write denied', message)
        self.assertEqual(self.window.ui_settings.get('last_project'), previous)
        self.assertEqual(list(configured.iterdir()), [])

    def test_startup_accepts_writable_project_root_and_removes_probe(self):
        configured = self.root / 'configured'
        configured.mkdir()
        store, message = _startup_project_store(str(configured), self.root / 'fallback')
        self.assertEqual(store.root, configured.resolve())
        self.assertIsNone(message)
        self.assertEqual(list(configured.iterdir()), [])

    def test_explicit_startup_image_takes_precedence_over_last_project(self):
        self.window.ui_settings.set('last_project', str(self.root / 'previous.json'))
        self.window._explicit_startup_image = True
        with patch.object(self.window, '_open_project_path') as open_previous:
            self.assertFalse(self.window._restore_startup_project())
        open_previous.assert_not_called()
        self.assertEqual(self.window.image_path, str(self.source))

    def test_project_title_uses_metadata_name_and_handles_absent_metadata(self):
        self.assertEqual(self.window._project_title(), ' — Wooden Bird')
        self.window.project_store.create(self.window._document(), 'Friendly project name')
        self.window._refresh_product_state()
        self.assertEqual(self.window._project_title(), ' — Friendly project name')
        self.window.project_status.setText.assert_called_with('Saved project: Friendly project name')

    def test_assistant_context_excludes_paths_seals_and_arbitrary_model_metadata(self):
        private_path = str(self.root / 'private-source.png')
        envelope = {
            'signature': 'DO_NOT_SHARE_SIGNATURE', 'key_id': 'DO_NOT_SHARE_KEY_ID',
            'payload': {
                'source': {'path': private_path},
                'parameters': {'prompt': 'DO_NOT_SHARE_PROMPT'},
                'models': [
                    {'model_id': 'depth-anything/Depth-Anything-V2-Small-hf', 'revision': 'a' * 40,
                     'checkpoint_path': private_path, 'api_key': 'DO_NOT_SHARE_CREDENTIAL'},
                    {'model_id': private_path, 'revision': private_path},
                    {'model_id': None, 'model_type': 'depth_anything_v2', 'revision': 'b' * 40},
                ],
            },
        }
        self.window._accepted_provenance = deepcopy(envelope)
        self.window._accepted_provenance_check = {'status': 'verified', 'reason': private_path}
        context = self.window._assistant_context()
        serialized = json.dumps(context)
        self.assertNotIn(private_path, serialized)
        self.assertNotIn('private-source.png', serialized)
        self.assertNotIn('DO_NOT_SHARE', serialized)
        self.assertEqual(context['model_provenance'], {
            'trust': 'verified', 'models': [
                {'name': 'Depth Anything V2 Small', 'revision': 'a' * 40},
                {'name': 'Unverified model', 'revision': None},
                {'name': 'Unverified model', 'revision': 'b' * 40},
            ],
        })
        self.assertEqual(self.window._accepted_provenance, envelope)
        self.assertEqual(context['image'], {'width': 20, 'height': 12})

    def test_assistant_summary_does_not_upgrade_unverified_mask_ancestry(self):
        self.window._accepted_provenance = {'payload': {
            'models': [], 'parameters': {'input_mask_provenance': 'unverified'},
        }}
        self.window._accepted_provenance_check = {'status': 'verified'}
        self.assertEqual(self.window._assistant_context()['model_provenance']['trust'], 'unverified')
        self.window._accepted_provenance_check = {'status': 'tampered'}
        self.assertEqual(self.window._assistant_context()['model_provenance']['trust'], 'tampered')


if __name__ == '__main__':
    unittest.main()
