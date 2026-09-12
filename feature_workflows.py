"""GUI orchestration for jobs, sessions and explicit preview/accept workflows."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QTextEdit, QVBoxLayout,
)

from data_contracts import as_bgr, proportional_shape
from generation_jobs import JobController
from session_state import SessionDocument, SessionHistory, load_session, save_session, load_preset, save_preset, validate_settings
from user_state import UserPaths, export_diagnostics
from model_store import ModelStore


SETTING_ATTRIBUTES = (
    'resolution', 'depth_amount', 'depth_drop_percentage', 'smoothing_method',
    'sensitivity', 'line_thickness', 'blend_amount', 'grayscale_enabled',
    'edge_detection_enabled', 'invert_colors_enabled', 'project_on_original',
    'flat_back_enabled', 'drop_background_enabled', 'background_tolerance',
    'use_processed_image_enabled', 'use_selected_color', 'current_selected_color', 'model',
)


def preview_image(image, settings):
    from edge_detection import detect_edges
    result = as_bgr(image)
    if settings.get('grayscale_enabled'):
        result = cv2.cvtColor(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if settings.get('edge_detection_enabled'):
        threshold = 200 - settings.get('sensitivity', 150)
        result = detect_edges(result, threshold, threshold * 3,
                              settings.get('line_thickness', 2), settings.get('project_on_original', True))
    if settings.get('invert_colors_enabled'):
        result = cv2.bitwise_not(result)
    amount = settings.get('blend_amount', 100) / 100
    return cv2.addWeighted(as_bgr(image), 1 - amount, result, amount, 0)


def pixmap(image, width=480, height=360):
    rgb = cv2.cvtColor(as_bgr(image), cv2.COLOR_BGR2RGB)
    qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage).scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


class WorkflowMixin:
    def _open_workflow_file(self, title, key, file_filter, directory=None):
        if hasattr(self, 'dialogs'):
            return self.dialogs.open_file(self, title, key, file_filter, initial_directory=str(directory) if directory else None)
        return QFileDialog.getOpenFileName(self, title, str(directory or ''), file_filter)

    def _save_workflow_file(self, title, key, file_filter, suggested_path, suffix):
        if hasattr(self, 'dialogs'):
            return self.dialogs.save_file(self, title, key, file_filter,
                suggested_name=Path(suggested_path).name, initial_directory=str(Path(suggested_path).parent), default_suffix=suffix)
        return QFileDialog.getSaveFileName(self, title, str(suggested_path), file_filter)

    def _init_workflow_state(self):
        self.paths = UserPaths.discover()
        self.model_store = ModelStore(root=self.paths.root)
        self.session_history = SessionHistory()
        self._subject_mask = None
        self._subject_mask_info = {}
        self._last_model_info = {}
        self._processing_history = []
        self._restoring_session = False
        self._source_generation = 0
        self._job_source_generation = 0
        self._closing_after_job = False
        self._mesh_before_repair = None
        self._accepted_job_folder = None
        self._jobs = JobController(self)
        self._jobs.progress.connect(lambda message: self.statusBar().showMessage(message))
        self._jobs.succeeded.connect(self._generation_succeeded)
        self._jobs.failed.connect(self.show_error)
        self._jobs.cancelled.connect(lambda: self.statusBar().showMessage('Cancelled. Previous mesh retained.'))
        self._jobs.discarded.connect(self._discard_job_result)
        self._jobs.idle.connect(self._generation_idle)
        self._history_timer = QTimer(self)
        self._history_timer.setSingleShot(True)
        self._history_timer.timeout.connect(self._record_session)
        self._viewport_timer = QTimer(self)
        self._viewport_timer.setInterval(20)
        self._viewport_timer.timeout.connect(self._poll_viewport)

    def _init_workflow_actions(self):
        file_menu = self.menuBar().addMenu('&Session')
        for label, method in (
            ('Open session…', self.open_session), ('Save session…', self.save_current_session),
            ('Load preset…', self.load_current_preset), ('Save preset…', self.save_current_preset),
            ('Export sanitized diagnostics…', self.save_diagnostics),
        ):
            file_menu.addAction(label, method)
        edit_menu = self.menuBar().addMenu('&Edit')
        self.undo_action = edit_menu.addAction('Undo settings / mask', lambda: self._history_move(False))
        self.undo_action.setShortcut('Ctrl+Z')
        self.redo_action = edit_menu.addAction('Redo settings / mask', lambda: self._history_move(True))
        self.redo_action.setShortcut('Ctrl+Shift+Z')
        ai_menu = self.menuBar().addMenu('&AI assistance')
        ai_menu.addAction('Suggest local parameters…', self.suggest_local_parameters)
        ai_menu.addAction('Edit subject mask (SAM2 / brushes)…', self.edit_subject_mask)
        ai_menu.addAction('Clear accepted subject mask', self.clear_subject_mask)
        models_menu = self.menuBar().addMenu('&Models')
        self.download_action = QAction('Allow model downloads when requested', self, checkable=True)
        self.download_action.setChecked(self.config.getboolean('Workflow', 'allow_downloads', fallback=False))
        models_menu.addAction(self.download_action)
        models_menu.addAction('Unload cached models', self.unload_models)
        models_menu.addAction('Register local MiDaS / DPT…', self.register_midas)
        models_menu.addAction('Review model licenses and download details…', self.model_details)
        mesh_menu = self.menuBar().addMenu('&Mesh')
        self.health_action = QAction('Check mesh health before export (optional)', self, checkable=True)
        self.health_action.setChecked(self.config.getboolean('Workflow', 'mesh_health', fallback=False))
        mesh_menu.addAction(self.health_action)
        mesh_menu.addAction('Inspect mesh health now…', self.inspect_current_mesh)
        mesh_menu.addAction('Preview conservative repair…', self.preview_mesh_repair)
        mesh_menu.addAction('Undo mesh repair', self.undo_mesh_repair)
        self.cancel_button = QPushButton('Cancel job')
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_generation)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self.statusBar().showMessage('Ready. Model downloads are controlled in the Models menu.')
        self._error_log = QTextEdit()
        self._error_log.setReadOnly(True)
        from PySide6.QtWidgets import QDockWidget
        self._error_dock = QDockWidget('Errors and diagnostics', self)
        self._error_dock.setObjectName('diagnosticsDock')
        self._error_dock.setWidget(self._error_log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._error_dock)
        self._error_dock.hide()
        for name in ('depth_amount_input', 'percentage_input', 'resolution_input'):
            getattr(self, name).textChanged.connect(self._settings_edited)
        for name in ('depth_method_dropdown', 'smoothing_dropdown'):
            getattr(self, name).currentTextChanged.connect(self._settings_edited)
        for name in ('flat_back_checkbox', 'drop_background_checkbox', 'use_processed_image_checkbox'):
            getattr(self, name).toggled.connect(self._settings_edited)
        self.background_tolerance_input.valueChanged.connect(self._settings_edited)
        self._record_session()

    def _settings_edited(self, *args):
        if self._restoring_session:
            return
        self._source_generation += 1
        self.cancel_generation()
        self._schedule_history()

    def model_details(self):
        from model_store import HF_MODELS
        box = QMessageBox(self)
        box.setWindowTitle('Model downloads and licenses')
        box.setText('Downloads are optional and may be large. Review the license of each model for your intended use. Preparation records an immutable model revision. MiDaS/DPT use registered local source and checkpoint files.')
        box.setDetailedText('\n\n'.join(f'{key}: {model_id}\nLicense: {license_name}\nhttps://huggingface.co/{model_id}' for key, (model_id, license_name) in HF_MODELS.items()))
        box.exec()

    def _settings_snapshot(self):
        settings = {key: deepcopy(getattr(self, key)) for key in SETTING_ATTRIBUTES}
        settings.update(
            resolution=int(self.resolution_input.text()),
            depth_amount=float(self.depth_amount_input.text()),
            depth_drop_percentage=float(self.percentage_input.text()),
            model=self.depth_method_dropdown.currentText(),
            smoothing_method=self.smoothing_dropdown.currentText(),
            sensitivity=self.sensitivity_slider.value(),
            line_thickness=self.line_thickness_slider.value(),
        )
        return validate_settings(settings)

    def _document(self):
        metadata = deepcopy(self._last_model_info)
        metadata.pop('accepted_mesh', None)
        if self._subject_mask_info:
            metadata['subject_mask_model'] = deepcopy(self._subject_mask_info)
        if hasattr(self, '_mesh_metadata_for_project'):
            accepted_mesh = self._mesh_metadata_for_project()
            if accepted_mesh is not None:
                metadata['accepted_mesh'] = accepted_mesh
        return SessionDocument(self.image_path or '', self._settings_snapshot(),
                               metadata, self._subject_mask, deepcopy(self._processing_history))

    def _record_session(self):
        if self._restoring_session or not self.initialized:
            return
        try:
            self.session_history.record(self._document())
        except (ValueError, TypeError):
            # QLineEdit validators allow transient empty text while editing.
            return
        if hasattr(self, 'undo_action'):
            self.undo_action.setEnabled(self.session_history.can_undo)
            self.redo_action.setEnabled(self.session_history.can_redo)
        if hasattr(self, '_refresh_workspace'):
            self._refresh_workspace()
        if hasattr(self, '_request_project_save'):
            self._request_project_save()

    def _schedule_history(self):
        if hasattr(self, '_history_timer') and not self._restoring_session:
            self._history_timer.start(250)

    def _apply_settings(self, settings):
        settings = validate_settings(settings)
        self._restoring_session = True
        widgets = []
        try:
            for key, value in settings.items():
                setattr(self, key, deepcopy(value))
            checkboxes = {
                'grayscale_enabled': 'grayscale_checkbox', 'edge_detection_enabled': 'edge_detection_checkbox',
                'invert_colors_enabled': 'invert_checkbox', 'project_on_original': 'project_on_original_checkbox',
                'flat_back_enabled': 'flat_back_checkbox', 'drop_background_enabled': 'drop_background_checkbox',
                'use_processed_image_enabled': 'use_processed_image_checkbox',
            }
            values = {
                'resolution': ('resolution_input', 'setText'), 'depth_amount': ('depth_amount_input', 'setText'),
                'depth_drop_percentage': ('percentage_input', 'setText'), 'sensitivity': ('sensitivity_slider', 'setValue'),
                'line_thickness': ('line_thickness_slider', 'setValue'), 'blend_amount': ('blend_slider', 'setValue'),
                'background_tolerance': ('background_tolerance_input', 'setValue'),
                'model': ('depth_method_dropdown', 'setCurrentText'), 'smoothing_method': ('smoothing_dropdown', 'setCurrentText'),
            }
            for key, name in checkboxes.items():
                if key in settings:
                    widget = getattr(self, name)
                    widgets.append((widget, widget.blockSignals(True)))
                    widget.setChecked(settings[key])
            for key, (name, method) in values.items():
                if key in settings:
                    widget = getattr(self, name)
                    widgets.append((widget, widget.blockSignals(True)))
                    value = str(settings[key]) if method in ('setText', 'setCurrentText') else int(settings[key])
                    getattr(widget, method)(value)
            self.edge_thickness = self.line_thickness
            self.update_color_swatch(self.current_selected_color)
            self.update_preview()
        finally:
            for widget, blocked in widgets:
                widget.blockSignals(blocked)
            self._restoring_session = False

    def _model_work_active(self):
        from subject_mask import has_active_jobs
        return self._jobs.busy or has_active_jobs()

    def _restore_document(self, document):
        if self._model_work_active():
            raise ValueError('Wait for the active job to finish cancelling before restoring a session.')
        settings = validate_settings(document.settings)
        source = document.source_path
        image = None
        if source:
            image = cv2.imread(source, cv2.IMREAD_UNCHANGED)
            if image is None:
                source, _ = self._open_workflow_file('Locate the session source image', 'source_relocation', 'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
                if not source:
                    return False
                image = cv2.imread(source, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError('The selected source image could not be decoded.')
            image = as_bgr(image)
            if document.mask is not None and document.mask.shape != image.shape[:2]:
                raise ValueError('The saved subject mask does not match this image size.')
        if document.model_info.get('backend') == 'huggingface' and not getattr(self, '_restoring_project', False):
            # Validate and persist the revision before mutating visible session state.
            self.model_store.pin_revision(document.model_info.get('model_type'), document.model_info.get('revision'))
        self.cancel_generation()
        self._source_generation += 1
        self.image_path, self.image = source or None, image
        self._subject_mask = None if document.mask is None else document.mask.copy()
        self._last_model_info = deepcopy(document.model_info)
        self._subject_mask_info = self._last_model_info.pop("subject_mask_model", {})
        self._last_model_info.pop('accepted_mesh', None)
        self._subject_mask_provenance_trusted = False
        self._processing_history = deepcopy(document.history)
        self._apply_settings(settings)
        if image is not None:
            self.display_original_image()
        else:
            self.processed_image = None
            self.original_label.clear()
            self.preview_label.clear()
        if hasattr(self, '_refresh_workspace'):
            self._refresh_workspace()
        self.statusBar().showMessage('Session restored. Generate a mesh to apply its settings to geometry.')
        if hasattr(self, '_request_project_save'):
            self._request_project_save()
        return True

    def _history_move(self, redo):
        self._history_timer.stop()
        if not redo:
            self._record_session()
        document = self.session_history.redo() if redo else self.session_history.undo()
        if document is not None:
            try:
                if not self._restore_document(document):
                    self.session_history.undo() if redo else self.session_history.redo()
            except Exception as error:
                self.session_history.undo() if redo else self.session_history.redo()
                self.show_error(str(error))
        self.undo_action.setEnabled(self.session_history.can_undo)
        self.redo_action.setEnabled(self.session_history.can_redo)
        if hasattr(self, '_refresh_workspace'):
            self._refresh_workspace()

    def save_current_session(self):
        path, _ = self._save_workflow_file('Save session', 'sessions', 'EdgeMesh session (*.json)', self.paths.root / 'session.edgemesh.json', 'json')
        if path:
            try:
                save_session(path, self._document())
                self.statusBar().showMessage('Session saved.')
            except Exception as error:
                self.show_error(str(error))

    def open_session(self):
        path, _ = self._open_workflow_file('Open session', 'sessions', 'EdgeMesh session (*.json)', self.paths.root)
        if path:
            try:
                document = load_session(path)
                self._record_session()
                if self._restore_document(document):
                    self._record_session()
            except Exception as error:
                self.show_error(str(error))

    def save_current_preset(self):
        path, _ = self._save_workflow_file('Save preset', 'presets', 'Preset (*.json)', self.paths.presets_dir / 'preset.json', 'json')
        if path:
            try:
                save_preset(path, self._settings_snapshot())
            except Exception as error:
                self.show_error(str(error))

    def load_current_preset(self):
        path, _ = self._open_workflow_file('Load preset', 'presets', 'Preset (*.json)', self.paths.presets_dir)
        if path:
            try:
                settings = load_preset(path)
                self._record_session()
                self._apply_settings(settings)
                self._record_session()
            except Exception as error:
                self.show_error(str(error))

    def save_diagnostics(self):
        path, _ = self._save_workflow_file('Save sanitized diagnostics', 'diagnostics', 'ZIP (*.zip)', self.paths.root / 'diagnostics.zip', 'zip')
        if path:
            try:
                export_diagnostics(path, self.paths)
                self.statusBar().showMessage('Sanitized diagnostics saved; source images excluded.')
            except Exception as error:
                self.show_error(str(error))

    def _start_generation(self, edge_only=False):
        if self.image is None:
            self.show_error('Please load an image first.')
            return
        if self._model_work_active():
            self.show_error('A job is already running. Cancel it or wait for completion.')
            return
        try:
            settings = self._settings_snapshot()
            if not edge_only and not self._ensure_model_license(settings['model']):
                return
            image = as_bgr(self.processed_image if edge_only or settings['use_processed_image_enabled'] else self.image)
            mask = None if self._subject_mask is None else self._subject_mask.copy()
            source = str(self.image_path)
            allow_download = self.download_action.isChecked()
            verbose = self.verbose
            store = self.model_store
            cancellation = self._jobs
            work_root = self.paths.work_dir.resolve()
            self._job_source_generation = self._source_generation
            self._record_session()
            def work(check, progress):
                folder = Path(tempfile.mkdtemp(prefix='job-', dir=work_root)).resolve()
                keep = False
                try:
                    check()
                    if edge_only:
                        from mesh_generator import MeshGenerator
                        progress('Constructing contour mesh')
                        mesh = MeshGenerator({key: False for key in ('visualize_partitioning', 'visualize_clustering', 'visualize_edges', 'visualize_depth')}).generate(image)
                        check()
                        result = {'mesh': mesh, 'background': [0.04] * 3, 'labels': None, 'model_info': {}}
                    else:
                        from depth_to_3d import DepthTo3D, model_names
                        progress('Preparing model (offline unless downloads are enabled)')
                        pipeline = DepthTo3D(model_names[settings['model']], verbose=verbose, model_store=store, allow_download=allow_download,
                            cancelled=lambda: cancellation.token.event.is_set())
                        check()
                        target = (0, 0) if settings['resolution'] == 0 else proportional_shape(image.shape, settings['resolution'])
                        path, background = pipeline.process_image(source, image_data=image, subject_mask=mask,
                            output_dir=folder, progress=progress, cancel_check=check, target_size=target,
                            smoothing_method=settings['smoothing_method'], flat_back=settings['flat_back_enabled'],
                            grayscale_enabled=settings['grayscale_enabled'], edge_detection_enabled=settings['edge_detection_enabled'],
                            invert_colors_enabled=settings['invert_colors_enabled'], depth_amount=settings['depth_amount'],
                            depth_drop_percentage=settings['depth_drop_percentage'], project_on_original=settings['project_on_original'],
                            background_removal=settings['drop_background_enabled'], background_tolerance=settings['background_tolerance'],
                            background_color=settings['current_selected_color'] if settings['use_selected_color'] else None)
                        color = [v / 255 for v in background] if min(background) >= 0 else [0.04] * 3
                        result = {'path': path, 'background': color, 'labels': pipeline.depth_labels, 'model_info': pipeline.model_info}
                    check()
                    result['settings'] = settings
                    result['folder'] = str(folder)
                    keep = True
                    return result
                finally:
                    if not keep and folder.is_relative_to(work_root) and folder.name.startswith('job-'):
                        shutil.rmtree(folder)
            self.cancel_button.setEnabled(True)
            self.process_button.setEnabled(False)
            self.generate_mesh_button.setEnabled(False)
            self._jobs.start(work)
            if hasattr(self, '_refresh_workspace'):
                self._refresh_workspace()
        except Exception as error:
            self.show_error(str(error))
            self._generation_idle()

    def cancel_generation(self):
        if hasattr(self, '_jobs'):
            self._jobs.cancel()
            if self._jobs.busy:
                self.statusBar().showMessage('Cancelling after the current operation…')

    def _generation_succeeded(self, result):
        cancelled = self._jobs.token is not None and self._jobs.token.event.is_set()
        if cancelled or self._job_source_generation != self._source_generation or self._closing_after_job:
            self._discard_job_result(result)
            self.statusBar().showMessage('Outdated result discarded; previous mesh retained.')
            return
        previous = (self.mesh_from_2d, self.mesh_3d, self.depth_labels, self._last_model_info, self.background_color)
        self.mesh_from_2d = result.get('mesh')
        self.mesh_3d = result.get('path')
        self.depth_labels = result['labels']
        self._last_model_info = result['model_info']
        self.background_color = result['background']
        if self.update_3d_viewport(self.background_color) is False:
            self.mesh_from_2d, self.mesh_3d, self.depth_labels, self._last_model_info, self.background_color = previous
            self._discard_job_result(result)
            return
        old_folder = self._accepted_job_folder
        self._accepted_job_folder = result.get('folder')
        if old_folder:
            self._discard_job_result({'folder': old_folder})
        self._processing_history.append({'operation': 'contour' if result.get('mesh') is not None else 'depth',
            'settings': deepcopy(result.get('settings', {})), 'model_info': deepcopy(result['model_info'])})
        self._processing_history = self._processing_history[-30:]
        self._mesh_before_repair = None
        self._mesh_before_repair_provenance = None
        try:
            self._seal_accepted_mesh(result['model_info'],
                mask_info=self._subject_mask_info if result.get('mesh') is None and self._subject_mask is not None else None,
                settings=result.get('settings', {}))
        except Exception as error:
            self._accepted_provenance = None
            self._provenance_mesh_token = None
            self.show_error(f'Mesh generated, but provenance could not be recorded: {error}')
            self._refresh_mesh_provenance_badge()
        self._record_session()
        self.statusBar().showMessage('Mesh ready. Use Export Mesh to save it to your chosen location.')

    def _discard_job_result(self, result):
        try:
            folder = Path(result['folder']).resolve()
            if folder.is_relative_to(self.paths.work_dir.resolve()) and folder.name.startswith('job-'):
                shutil.rmtree(folder)
        except Exception as error:
            self.show_error(f'Could not clean up the cancelled job: {error}')

    def _generation_idle(self):
        self.cancel_button.setEnabled(False)
        self.process_button.setEnabled(True)
        self.generate_mesh_button.setEnabled(True)
        if hasattr(self, '_refresh_workspace'):
            self._refresh_workspace()
        if self._closing_after_job:
            QTimer.singleShot(0, self.close)

    def _poll_viewport(self):
        if self.three_d_viewport is None:
            self._viewport_timer.stop()
            return
        try:
            viewer = self.three_d_viewport.viewer
            if not viewer.poll_events():
                viewer.destroy_window()
                self.three_d_viewport = None
                self._viewport_timer.stop()
            else:
                viewer.update_renderer()
        except Exception as error:
            self._viewport_timer.stop()
            self.show_error(f'Viewport update failed: {error}')

    def unload_models(self):
        if self._model_work_active():
            self.show_error('Wait for the active job before unloading models.')
            return
        self.depth_to_3d = None
        self.model_store.clear()
        self.statusBar().showMessage('Cached model objects unloaded.')

    def register_midas(self):
        if self._model_work_active():
            self.show_error('Wait for the active job before changing model setup.')
            return
        from subject_mask import MiDaSSetupDialog
        dialog = MiDaSSetupDialog(self.model_store, parent=self)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def edit_subject_mask(self):
        if self.image is None or self._model_work_active():
            self.show_error('Load an image and wait for the active job before editing its mask.')
            return
        from subject_mask import SubjectMaskDialog
        dialog = None
        try:
            dialog = SubjectMaskDialog(self.image, self.model_store, parent=self,
                initial_mask=self._subject_mask, allow_download=self.download_action.isChecked())
            if dialog.exec() == QDialog.DialogCode.Accepted and dialog.accepted_mask is not None:
                self._record_session()
                self._subject_mask = dialog.accepted_mask.copy()
                self._subject_mask_info = deepcopy(getattr(dialog, "model_metadata", {}))
                self._subject_mask_provenance_trusted = bool(self._subject_mask_info)
                self._source_generation += 1
                self._record_session()
                self.statusBar().showMessage('Subject mask accepted. Generate a depth mesh to apply it.')
        except Exception as error:
            self.show_error(str(error))
        finally:
            if dialog is not None and hasattr(dialog, 'deleteLater'):
                dialog.deleteLater()

    def clear_subject_mask(self):
        self._record_session()
        self._subject_mask = None
        self._subject_mask_info = {}
        self._subject_mask_provenance_trusted = False
        self._source_generation += 1
        self.cancel_generation()
        self._record_session()
        self.statusBar().showMessage('Subject mask cleared.')

    def suggest_local_parameters(self):
        if self.image is None:
            self.show_error('Load an image before requesting suggestions.')
            return
        from parameter_suggestions import suggest_parameters
        dialog = None
        try:
            current = self._settings_snapshot()
            suggestions = suggest_parameters(self.image, current)
            proposed = dict(current)
            for suggestion in suggestions:
                proposed.update(suggestion.settings)
            proposed = validate_settings(proposed)
            dialog = QDialog(self)
            dialog.setWindowTitle('Local parameter suggestions — preview before applying')
            dialog.setObjectName('localParameterSuggestions')
            layout = QVBoxLayout(dialog)
            reasons = QLabel('\n\n'.join(suggestion.title + ': ' + suggestion.reason + '\n' + str(dict(suggestion.settings)) for suggestion in suggestions))
            reasons.setWordWrap(True)
            layout.addWidget(reasons)
            row = QHBoxLayout()
            for title, settings in (('Current preview', current), ('Suggested preview', proposed)):
                column = QVBoxLayout()
                column.addWidget(QLabel(title))
                label = QLabel()
                label.setPixmap(pixmap(preview_image(self.image, settings), 400, 300))
                column.addWidget(label)
                row.addLayout(column)
            layout.addLayout(row)
            layout.addWidget(QLabel('Depth/smoothing changes affect the next mesh generation; the image preview shows edge and color changes.'))
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
            buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(dialog.accept)
            buttons.button(QDialogButtonBox.StandardButton.Apply).setProperty('edgemeshAcceptButton', True)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._record_session()
                self._apply_settings(proposed)
                self._record_session()
        except Exception as error:
            self.show_error(str(error))
        finally:
            if dialog is not None:
                dialog.deleteLater()

    def _current_mesh(self):
        return self.three_d_viewport.mesh if self.three_d_viewport is not None else None

    def inspect_current_mesh(self, for_export=False):
        if self._jobs.busy and for_export:
            self.show_error("Wait for generation before checking a mesh for export.")
            return False
        from mesh_health import inspect_mesh
        if self._current_mesh() is None:
            self.show_error('No loaded mesh is available to inspect.')
            return False
        try:
            mesh = self._current_mesh()
            report = inspect_mesh(mesh)
            details = '\n'.join(f'{key}: {value}' for key, value in report.to_dict().items())
            box = QMessageBox(self)
            box.setWindowTitle('Optional mesh-health report')
            box.setObjectName('meshHealthReport')
            box.setText('Review the mesh report. Open surfaces may be intentional.')
            box.setDetailedText(details)
            box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel if for_export else QMessageBox.StandardButton.Ok)
            accepted = box.exec() == QMessageBox.StandardButton.Ok
            return accepted and self._current_mesh() is mesh
        except Exception as error:
            self.show_error(str(error))
            return False

    def preview_mesh_repair(self):
        if self._jobs.busy:
            self.show_error("Wait for generation before previewing a repair.")
            return
        from mesh_health import inspect_mesh, repair_preview
        mesh = self._current_mesh()
        if mesh is None:
            self.show_error('No mesh is loaded.')
            return
        try:
            candidate = repair_preview(mesh)
            dialog = QMessageBox(self)
            dialog.setWindowTitle('Preview conservative mesh repair')
            dialog.setObjectName('meshRepairPreview')
            dialog.setText('Remove invalid/duplicate/degenerate faces and unused vertices? Holes are not filled. You can undo this repair.')
            dialog.setDetailedText('Before:\n' + str(inspect_mesh(mesh).to_dict()) + '\n\nAfter:\n' + str(inspect_mesh(candidate).to_dict()))
            dialog.setStandardButtons(QMessageBox.StandardButton.Apply | QMessageBox.StandardButton.Cancel)
            dialog.button(QMessageBox.StandardButton.Apply).setProperty('edgemeshAcceptButton', True)
            if dialog.exec() == QMessageBox.StandardButton.Apply and self._current_mesh() is mesh:
                prior_mesh_source = self.mesh_from_2d
                prior_provenance = deepcopy(self._accepted_provenance)
                self.mesh_from_2d = candidate
                if self.update_3d_viewport(self.background_color) is False:
                    self.mesh_from_2d = prior_mesh_source
                    return
                self._mesh_before_repair = deepcopy(mesh)
                self._mesh_before_repair_provenance = prior_provenance
                try:
                    self._seal_mesh_repair(mesh, prior_provenance)
                finally:
                    self._record_session()
        except Exception as error:
            self.show_error(str(error))

    def undo_mesh_repair(self):
        if self._mesh_before_repair is not None:
            prior_mesh_source = self.mesh_from_2d
            self.mesh_from_2d = self._mesh_before_repair
            if self.update_3d_viewport(self.background_color) is False:
                self.mesh_from_2d = prior_mesh_source
                return
            self._accepted_provenance = self._mesh_before_repair_provenance
            self._provenance_mesh_token = None
            self._mesh_before_repair = None
            self._mesh_before_repair_provenance = None
            self._refresh_mesh_provenance_badge()
            self._record_session()
