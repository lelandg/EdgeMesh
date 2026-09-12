"""Transparent project persistence and desktop workflow integration."""
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import re
import tempfile

from PySide6.QtCore import QByteArray, QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMenu, QPushButton, QTextBrowser, QToolBar, QVBoxLayout,
)

from project_store import ProjectStore
from session_state import SessionDocument
from ui_persistence import DialogPersistence, SettingsStore


def _project_warning(message):
    try:
        from log_utils import get_logger
        get_logger().warning(message)
    except (OSError, RuntimeError):
        logging.getLogger(__name__).warning(message)


def _startup_project_store(configured_root, default_root):
    """Recover stale saved destinations without touching their existing files."""
    if configured_root is None or configured_root == '':
        return ProjectStore(default_root), None
    try:
        if not isinstance(configured_root, str) or not configured_root.strip():
            raise ValueError('The configured projects location is not a folder path')
        candidate = Path(configured_root).expanduser().resolve(strict=True)
        if not candidate.is_dir():
            raise ValueError('The configured projects location is not a directory')
        # Permission bits alone do not establish Windows ACL or volume access.
        with tempfile.TemporaryFile(prefix='.edgemesh-write-', dir=candidate) as probe:
            probe.write(b'1')
            probe.flush()
        return ProjectStore(candidate), None
    except (OSError, ValueError, TypeError) as error:
        message = (f'The configured projects folder is unavailable ({error}). '
                   f'New projects will use {default_root}. Choose another folder in Setup if needed.')
        _project_warning(message)
        return ProjectStore(default_root), message


class ProjectWorkflowMixin:
    def _init_product_state(self):
        self.ui_settings = SettingsStore(self.paths.root / 'ui-settings.json')
        self.dialogs = DialogPersistence(self.ui_settings, self)
        self.project_store, self._startup_project_warning = _startup_project_store(
            self.ui_settings.get('projects_root'), self.paths.root / 'projects')
        if self._startup_project_warning:
            try:
                self.ui_settings.set('projects_root', str(self.project_store.root))
            except (OSError, ValueError, TypeError):
                _project_warning('Could not remember the recovered projects folder.')
        self._product_ready = False
        self._restoring_project = False
        self._saved_project_token = None
        self._saved_project_mesh_token = None
        self._autosave_error = None
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._save_project_now)
        self._init_compliance()

    def _finish_product_workflow(self):
        from assistant_panel import AssistantPanel
        self.assistant_panel = AssistantPanel(self, context_provider=self._assistant_context,
            settings_path=self.paths.root / 'assistant-settings.json', state_dir=self.paths.root / 'agents')
        self.workspace_tabs.addTab(self.assistant_panel, 'Assistant')
        self.project_toolbar = QToolBar('Project', self)
        self.project_toolbar.setObjectName('projectToolbar')
        self.addToolBar(self.project_toolbar)
        self.project_status = QLabel('Project will save automatically.')
        self.project_status.setAccessibleName('Project autosave status and folder')
        self.project_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.project_toolbar.addWidget(self.project_status)
        self.project_toolbar.addSeparator()
        self.project_toolbar.addAction('Open project…', self.open_project)
        self.project_toolbar.addAction('Save copy…', self.save_project_as)
        self._default_workspace_state = self.saveState()
        self.project_menu = QMenu('&Project', self)
        actions = self.menuBar().actions()
        if actions:
            self.menuBar().insertMenu(actions[0], self.project_menu)
        else:
            self.menuBar().addMenu(self.project_menu)
        for title, callback, shortcut in (
            ('&New project', self.new_project, 'Ctrl+N'),
            ('&Open project…', self.open_project, 'Ctrl+Shift+O'),
            ('&Save project now', lambda: self._save_project_now(force=True), 'Ctrl+S'),
            ('Save project &copy…', self.save_project_as, 'Ctrl+Shift+S'),
            ('Choose folder for new projects…', self.choose_project_root, 'Ctrl+Alt+P'),
        ):
            action = self.project_menu.addAction(title, callback)
            action.setShortcut(shortcut)
        self.project_menu.addSeparator()
        self.project_menu.addAction('Inspect accepted mesh provenance…', self.show_mesh_provenance)
        self._install_accessibility_shortcuts()
        self.depth_method_dropdown.currentTextChanged.connect(self._refresh_license_indicators)
        self.mask_view_mode.currentTextChanged.connect(self._request_project_save)
        self.mask_opacity.valueChanged.connect(self._request_project_save)
        self.parameter_tabs.currentChanged.connect(self._request_project_save)
        self.workspace_tabs.currentChanged.connect(self._request_project_save)
        self._product_ready = True
        restored = self._restore_startup_project()
        if not restored:
            self._request_project_save()
        saved = getattr(self, '_pending_window_state', '')
        if saved:
            try:
                if not isinstance(saved, str) or not self.restoreState(QByteArray.fromHex(saved.encode('ascii'))):
                    _project_warning('Could not restore the saved workspace layout; keeping the available layout.')
            except (TypeError, ValueError, UnicodeError, RuntimeError):
                _project_warning('Could not restore the saved workspace layout; keeping the available layout.')
        try:
            active = self.config.getint('Workspace', 'active_tab', fallback=0)
        except (TypeError, ValueError):
            active = 0
            _project_warning('Could not restore the saved workspace page; opening Workspace.')
        if self.image is not None and 0 <= active < self.workspace_tabs.count():
            self.workspace_tabs.setCurrentIndex(active)
        self._refresh_product_state()
        warning = getattr(self, '_startup_project_warning', None)
        if warning:
            self._startup_project_warning = None
            self.show_error(warning)

    def _restore_startup_project(self):
        if (getattr(self, '_explicit_startup_image', False)
                or not getattr(self, '_restore_last_project', True)):
            return False
        previous = self.ui_settings.get('last_project')
        if not previous:
            return False
        try:
            return self._open_project_path(previous, save_previous=False)
        except Exception as error:
            self.show_error(f'Could not reopen the previous project: {error}')
            return False

    def _make_project_settings(self):
        group = QGroupBox('Projects and saved settings')
        layout = QVBoxLayout(group)
        label = QLabel('Projects save automatically with their source image, mask, settings, accepted mesh and model provenance. The last project reopens at startup.')
        label.setWordWrap(True)
        layout.addWidget(label)
        row = QHBoxLayout()
        self.project_root_display = QLineEdit(str(self.project_store.root))
        self.project_root_display.setReadOnly(True)
        self.project_root_display.setAccessibleName('Folder used for new projects')
        row.addWidget(self.project_root_display, 1)
        browse = QPushButton('&Change folder…')
        browse.clicked.connect(self.choose_project_root)
        row.addWidget(browse)
        layout.addLayout(row)
        note = QLabel('Changing this folder affects new projects. Use Save project copy to move your current work into a new folder. Model weights and credentials are not copied into projects.')
        note.setWordWrap(True)
        layout.addWidget(note)
        return group

    def _request_project_save(self, *args):
        if self._product_ready and not self._restoring_project and not self._restoring_session:
            self._autosave_timer.start(650)

    def _project_token(self):
        state = self.three_d_viewport.save_view_state() if self.three_d_viewport is not None else {}
        return (self.session_history.revision, id(self.image), id(self._current_mesh()),
                json.dumps(state, sort_keys=True, allow_nan=False),
                self.workspace_tabs.currentIndex(), self.parameter_tabs.currentIndex(),
                self.mask_view_mode.currentText(), self.mask_opacity.value())

    def _mesh_file_for_project(self):
        mesh = self._current_mesh()
        if mesh is None:
            return None, None
        saved = self.project_store.current_mesh_path
        if self._saved_project_mesh_token == id(mesh) and saved is not None and saved.is_file():
            return saved, None
        import open3d as o3d
        descriptor, name = tempfile.mkstemp(prefix='project-mesh-', suffix='.ply', dir=self.paths.work_dir)
        os.close(descriptor)
        path = Path(name)
        try:
            if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
                raise OSError('Could not store the accepted mesh in this project.')
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path, path

    def _save_project_now(self, force=False):
        if not self._product_ready or self._restoring_project:
            return True
        self._autosave_timer.stop()
        # A portable project starts with its first source image. An empty New
        # workspace has nothing to copy and must not fail startup or closing.
        if self.image is None and not self.image_path:
            return True
        temporary = None
        try:
            if self._history_timer.isActive():
                self._history_timer.stop()
                self._record_session()
            self._autosave_timer.stop()
            token = self._project_token()
            if not force and token == self._saved_project_token:
                return True
            try:
                document = self._document()
            except (ValueError, TypeError):
                # Only incomplete form input is transient. Store validation
                # failures below must be visible even during background saves.
                if not force:
                    return False
                raise
            mesh_path, temporary = self._mesh_file_for_project()
            path = self.project_store.save_current(document, mesh_path=mesh_path)
            managed = self.project_store.current_document
            if managed is not None and managed.source_path:
                if document.source_path and document.source_path != managed.source_path:
                    self.session_history.relocate_source(document.source_path, managed.source_path)
                self.image_path = managed.source_path
                self.config.set('Settings', 'last_used_image', self.image_path)
            self._saved_project_mesh_token = id(self._current_mesh())
            self._saved_project_token = self._project_token()
            self._autosave_error = None
            self.ui_settings.update({'last_project': str(path), 'projects_root': str(self.project_store.root)})
            self.save_ui_settings()
            self._refresh_product_state()
            return True
        except (ValueError, TypeError) as error:
            self._autosave_error = str(error)
            self.show_error(f'Project was not saved: {error}')
        except Exception as error:
            self._autosave_error = str(error)
            self.show_error(f'Project autosave failed: {error}')
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    _project_warning('Could not remove a temporary project mesh after saving.')
        self._refresh_product_state()
        return False

    def _before_new_source(self):
        if not self._product_ready or self._restoring_project:
            return True
        if not self._save_project_now(force=True):
            return False
        candidate = ProjectStore(self.project_store.root)
        self._history_timer.stop()
        self._clear_project_mesh()
        self.project_store = candidate
        self.session_history = type(self.session_history)(limit=self.session_history.limit)
        self._processing_history = []
        self._last_model_info = {}
        self._saved_project_token = None
        self._saved_project_mesh_token = None
        return True

    def _clear_project_mesh(self):
        viewport = self.three_d_viewport
        if viewport is not None:
            self._release_project_viewport(viewport)
        self.three_d_viewport = None
        self.mesh_3d = None
        self.mesh_from_2d = None
        self.depth_labels = None
        self._accepted_provenance = None
        self._provenance_mesh_token = None
        self._mesh_before_repair = None
        self._mesh_before_repair_provenance = None
        self.mesh_placeholder.show()

    def new_project(self):
        if self._model_work_active():
            self.show_error('Finish or cancel the active job before creating a project.')
            return
        try:
            document = SessionDocument('', self._settings_snapshot())
            if not self._before_new_source():
                return
            self._restoring_project = True
            try:
                self._restore_document(document)
            finally:
                self._restoring_project = False
            self.ui_settings.set('last_project', None)
            self.workspace_tabs.setCurrentWidget(self.setup_page)
            self._refresh_workspace()
        except Exception as error:
            self.show_error(str(error))

    def open_project(self):
        path, _ = self.dialogs.open_file(self, 'Open project or legacy session', 'projects',
            'EdgeMesh project or session (*.json)', initial_directory=str(self.project_store.root))
        if path:
            self.open_project_path(path)

    def open_project_path(self, path):
        try:
            return self._open_project_path(path)
        except Exception as error:
            self.show_error(f'Could not open project: {error}')
            return False

    def _open_project_path(self, path, save_previous=True):
        if self._model_work_active():
            raise ValueError('Finish or cancel the active job before opening a project.')
        if save_previous and not self._save_project_now(force=True):
            return False
        candidate_store = ProjectStore(self.project_store.root)
        snapshot = candidate_store.inspect(path)
        previous = self._capture_project_workspace()
        self._autosave_timer.stop()
        self._history_timer.stop()
        self._restoring_project = True
        committed = False
        candidate_history = type(self.session_history)(limit=self.session_history.limit)
        try:
            self.session_history = candidate_history
            # Keep the previous renderer alive and out of restore callbacks until
            # the entire candidate is accepted, preserving its mesh and camera.
            if previous['viewport'] is not None:
                previous['viewport'].hide()
            self.three_d_viewport = None
            if not self._restore_document(snapshot.document):
                self._restore_project_workspace(previous)
                return False
            metadata = snapshot.document.model_info
            stored = metadata.get('accepted_mesh', {}) if isinstance(metadata, dict) else {}
            if snapshot.mesh_path is not None:
                self.mesh_from_2d = None
                self.mesh_3d = str(snapshot.mesh_path)
                self.depth_labels = None
                if self.update_3d_viewport() is False:
                    raise ValueError('The project mesh could not be displayed; previous project retained.')
                self._accepted_provenance = deepcopy(stored.get('provenance')) if isinstance(stored, dict) else None
                if isinstance(stored, dict) and stored.get('view_state'):
                    if self.three_d_viewport.restore_view_state(stored['view_state']) is False:
                        _project_warning('Could not restore the project camera; keeping the available view.')
            else:
                self._clear_project_mesh()
            self._mesh_before_repair = None
            self._mesh_before_repair_provenance = None
            candidate_history.record(self._document())
            candidate_store.accept(snapshot)
            self.project_store = candidate_store
            self.session_history = candidate_history
            self._provenance_mesh_token = None
            self._saved_project_mesh_token = id(self._current_mesh())
            self._saved_project_token = self._project_token()
            self._autosave_error = None
            self._refresh_workspace()
            if isinstance(metadata, dict) and metadata.get('backend') == 'huggingface':
                self.model_store.pin_revision(metadata.get('model_type'), metadata.get('revision'))
            try:
                self.ui_settings.set('last_project', str(snapshot.path))
            except Exception:
                _project_warning('The project opened, but its location could not be remembered for startup.')
            committed = True
            return True
        except Exception:
            discarded = self.three_d_viewport
            try:
                self._restore_project_workspace(previous)
            except Exception:
                _project_warning('Could not completely refresh the previous project after a failed open.')
            finally:
                if discarded is not None and discarded is not previous['viewport']:
                    self._release_project_viewport(discarded)
            raise
        finally:
            try:
                if committed and previous['viewport'] is not None:
                    self._release_project_viewport(previous['viewport'])
            finally:
                self._restoring_project = False

    def _capture_project_workspace(self):
        values = {
            name: getattr(self, name, None) for name in (
                'image_path', 'image', 'processed_image', '_subject_mask',
                'mesh_from_2d', 'mesh_3d', 'depth_labels', '_mesh_before_repair',
                '_provenance_mesh_token', '_saved_project_token', '_saved_project_mesh_token',
                '_autosave_error', '_subject_mask_provenance_trusted',
            )
        }
        values.update({
            name: deepcopy(getattr(self, name, None)) for name in (
                '_last_model_info', '_subject_mask_info', '_processing_history',
                '_accepted_provenance', '_accepted_provenance_check', '_mesh_before_repair_provenance',
            )
        })
        viewport = self.three_d_viewport
        return {
            'store': self.project_store, 'history': self.session_history,
            'viewport': viewport, 'viewport_hidden': viewport.isHidden() if viewport is not None else True,
            'settings': self._settings_snapshot(), 'values': values,
            'page': self.workspace_tabs.currentIndex(), 'parameters_page': self.parameter_tabs.currentIndex(),
            'mask_mode': self.mask_view_mode.currentText(), 'mask_opacity': self.mask_opacity.value(),
        }

    def _restore_project_workspace(self, previous):
        self.project_store = previous['store']
        self.session_history = previous['history']
        self.three_d_viewport = previous['viewport']
        for name, value in previous['values'].items():
            setattr(self, name, value)
        restoring_session = self._restoring_session
        try:
            self._apply_settings(previous['settings'])
            self._restoring_session = True
            # Restore the live buffers, not files which could have moved while
            # the project picker was open. Neither the old mask nor mesh changed.
            self.processed_image = previous['values']['processed_image']
            if self.image is None:
                self.original_label.clear()
                self.preview_label.clear()
            else:
                self.display_original_image()
                if self.processed_image is not None:
                    self.display_processed_image()
            self.workspace_tabs.setCurrentIndex(previous['page'])
            self.parameter_tabs.setCurrentIndex(previous['parameters_page'])
            self.mask_view_mode.setCurrentText(previous['mask_mode'])
            self.mask_opacity.setValue(previous['mask_opacity'])
            viewport = self.three_d_viewport
            if viewport is None:
                self.mesh_placeholder.show()
            else:
                self.mesh_placeholder.hide()
                viewport.setVisible(not previous['viewport_hidden'])
            self._refresh_workspace()
        finally:
            self._restoring_session = restoring_session

    def _release_project_viewport(self, viewport):
        try:
            viewport.shutdown()
        except Exception:
            _project_warning('Could not completely release an inactive project preview.')
        try:
            self.mesh_preview_layout.removeWidget(viewport)
            viewport.deleteLater()
        except Exception:
            _project_warning('Could not remove an inactive project preview.')

    def save_project_as(self):
        if not self._save_project_now(force=True):
            return
        directory = self.dialogs.choose_directory(self, 'Choose parent folder for a project copy',
            'project_copy', initial_directory=str(self.project_store.root))
        if not directory:
            return
        try:
            copy_store = ProjectStore(directory)
            name = self._project_name() or 'Untitled'
            path = copy_store.create(self._document(), name, mesh_path=self.project_store.current_mesh_path)
            self._open_project_path(path, save_previous=False)
        except Exception as error:
            self.show_error(f'Could not save project copy: {error}')

    def choose_project_root(self):
        directory = self.dialogs.choose_directory(self, 'Choose folder for new projects', 'project_root',
            initial_directory=str(self.project_store.root))
        if not directory:
            return
        try:
            self.project_store.set_root(directory)
            self.ui_settings.set('projects_root', str(self.project_store.root))
            self._refresh_product_state()
            self.statusBar().showMessage('New projects will use the selected folder. Current project was not moved.')
        except Exception as error:
            self.show_error(str(error))

    def _refresh_product_state(self):
        if not getattr(self, '_product_ready', False):
            return
        if hasattr(self, 'project_root_display'):
            self.project_root_display.setText(str(self.project_store.root))
        path = self.project_store.current_path
        text = 'Autosave failed — use Save project copy' if self._autosave_error else (
            f'Saved project: {self._project_name() or "Untitled"}' if path else 'New project · saves automatically')
        self.project_status.setText(text)
        self.project_status.setToolTip(str(path.parent) if path else str(self.project_store.root))
        self._refresh_license_indicators()

    def _project_name(self):
        metadata = self.project_store.current_metadata
        name = metadata.get('name') if isinstance(metadata, dict) else None
        if isinstance(name, str) and name.strip():
            return name.strip()
        return Path(self.image_path).stem if self.image_path else ''

    def _project_title(self):
        """Human project name, including the separator used after EdgeMesh."""
        name = self._project_name()
        return f' — {name}' if name else ''

    def _assistant_context(self):
        from model_licensing import policy_for

        mesh = self._current_mesh()
        self._refresh_mesh_provenance_badge()
        checked = getattr(self, '_accepted_provenance_check', None)
        trust = checked.get('status') if isinstance(checked, dict) else None
        if trust not in ('verified', 'tampered', 'unverified'):
            trust = 'unverified'
        envelope = self._accepted_provenance
        payload = envelope.get('payload') if isinstance(envelope, dict) else None
        payload = payload if isinstance(payload, dict) else {}
        parameters = payload.get('parameters')
        if trust == 'verified' and isinstance(parameters, dict) and parameters.get('input_mask_provenance') == 'unverified':
            trust = 'unverified'
        identities = payload.get('models')
        models = []
        for identity in identities[:100] if isinstance(identities, list) else []:
            if not isinstance(identity, dict):
                continue
            policy = policy_for(identity['model_id'] if 'model_id' in identity else identity.get('model_type'))
            revision = identity.get('revision')
            models.append({
                'name': policy.display_name if policy.usage_class != 'unknown' else 'Unverified model',
                'revision': revision if isinstance(revision, str) and re.fullmatch(r'[0-9a-f]{40}', revision) else None,
            })
        return {'application': 'EdgeMesh', 'settings': self._settings_snapshot(),
                'image': {'width': int(self.image.shape[1]), 'height': int(self.image.shape[0])} if self.image is not None else None,
                'mask': {'foreground_percent': round(float(self._subject_mask.mean()) * 100, 2)} if self._subject_mask is not None else None,
                'mesh': {'vertices': len(mesh.vertices), 'triangles': len(mesh.triangles)} if mesh is not None else None,
                'model_provenance': {'trust': trust, 'models': models},
                'available_actions': ['Explain settings', 'Suggest settings for user review', 'Explain mesh diagnostics'],
                'instruction': 'Give advice. Do not run shell commands, change files, or claim to have edited the application. No image pixels or project paths are included in this context.'}

    def _install_accessibility_shortcuts(self):
        self._workflow_shortcuts = []
        self._generation_shortcuts = []
        commands = [
            ('Open image', 'Ctrl+O', self.load_image),
            ('Export mesh', 'Ctrl+E', self.export_mesh),
            ('Edit subject mask', 'Ctrl+M', self.edit_subject_mask),
            ('Clear subject mask', 'Ctrl+Shift+M', self.clear_subject_mask),
            ('Save processed image', 'Ctrl+Alt+S', self.save_image),
            ('Generate depth mesh', 'Shift+Return', self._keyboard_generate),
            ('Generate depth mesh keypad', 'Shift+Enter', self._keyboard_generate),
            ('Cancel current job', 'Escape', self.cancel_generation),
            ('Keyboard shortcuts', 'F1', self.show_keyboard_shortcuts),
        ]
        for index, title in enumerate(('Workspace', 'History', 'Setup', 'Assistant')):
            commands.append((title, f'Ctrl+{index + 1}', lambda checked=False, i=index: self.workspace_tabs.setCurrentIndex(i)))
        for title, key, callback in commands:
            action = QAction(title, self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(callback)
            self.addAction(action)
            self._workflow_shortcuts.append(action)
            if key in ('Shift+Return', 'Shift+Enter'):
                self._generation_shortcuts.append(action)
        self.workspace_tabs.currentChanged.connect(self._update_generation_shortcuts)
        self._update_generation_shortcuts()
        self.project_menu.addAction('Keyboard shortcuts…', self.show_keyboard_shortcuts)
        for button, name in ((self.load_button, 'Open image'), (self.save_button, 'Save processed image'),
                             (self.export_mesh_button, 'Export accepted mesh'), (self.mask_edit_button, 'Edit subject mask'),
                             (self.mask_clear_button, 'Clear subject mask')):
            button.setAccessibleName(name)

    def _update_generation_shortcuts(self, *args):
        enabled = self.workspace_tabs.currentIndex() == 0
        for action in self._generation_shortcuts:
            action.setEnabled(enabled)

    def _keyboard_generate(self):
        if self.workspace_tabs.currentIndex() == 0 and self.process_button.isEnabled():
            self.process_button.click()

    def show_keyboard_shortcuts(self):
        dialog = QDialog(self)
        dialog.setObjectName('keyboardShortcuts')
        dialog.setWindowTitle('Keyboard and mouse controls')
        dialog.resize(700, 560)
        layout = QVBoxLayout(dialog)
        text = QTextBrowser()
        text.setAccessibleName('Keyboard shortcut reference')
        text.setPlainText('Projects: Ctrl+N new · Ctrl+Shift+O open · Ctrl+S save · Ctrl+Shift+S save copy\n'
            'Source: Ctrl+O open image · Ctrl+Alt+S save processed image\n'
            'Mesh: Alt+D or Shift+Enter depth mesh · Alt+M contour mesh · Ctrl+E export\n'
            'Mask: Ctrl+M edit · Ctrl+Shift+M clear · Ctrl+Z undo · Ctrl+Shift+Z redo\n'
            'Pages: Ctrl+1 workspace · Ctrl+2 history · Ctrl+3 setup · Ctrl+4 assistant\n'
            'Tab / Shift+Tab move focus. Space activates buttons/check boxes. Arrow keys adjust selections and sliders.\n'
            'Dialogs: Shift+Enter clicks an enabled confirmation action. Required acknowledgments still apply. Escape cancels.\n\n'
            'Inside the 3D viewer:\n'
            'Wheel zooms; Shift+wheel translates world X; Ctrl+wheel world Y; Alt+wheel world Z.\n'
            'Ctrl+Shift+wheel rotates about world X; Alt+Shift+wheel world Y; Ctrl+Alt+wheel world Z.\n'
            'Ctrl+Left/Right: translate X · Ctrl+Down/Up: Y · Ctrl+PageDown/PageUp: Z\n'
            'Alt+Down/Up: rotate X · Alt+Left/Right: Y · Alt+PageDown/PageUp: Z\n'
            'F fit · O projection · W display mode · A axes · 1–7 standard views\n'
            'Camera controls navigate the view; they do not modify exported mesh geometry.')
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()
