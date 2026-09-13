"""Desktop workspace, proportional previews and focused getting-started controls."""
from importlib.metadata import PackageNotFoundError, version as package_version

import cv2
from PySide6.QtCore import QByteArray, Qt, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter,
    QTabWidget, QToolBar, QVBoxLayout, QWidget,
)


class ImagePreviewLabel(QLabel):
    """Keep the original pixmap when splitters resize a letterboxed preview."""

    def __init__(self, text='', parent=None):
        super().__init__(text, parent)
        self._source_pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(120, 90)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def sizeHint(self):
        return QSize(320, 220)

    def setPixmap(self, pixmap):
        self._source_pixmap = QPixmap(pixmap)
        self._fit()

    def clear(self):
        self._source_pixmap = QPixmap()
        super().clear()

    def _fit(self):
        if not self._source_pixmap.isNull():
            super().setPixmap(self._source_pixmap.scaled(
                self.contentsRect().size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


def _text(value):
    label = QLabel(value)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


def _scroll(widget):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setWidget(widget)
    return area


class WorkspaceMixin:
    def _build_workspace(self, image_controls, mesh_controls):
        self._workspace_ready = False
        # Loading the initial image can save settings before construction ends.
        # Keep the user's original layout until the final restoration.
        self._startup_window_layout = {
            key: self.config.get('UI_Settings', key, fallback='')
            for key in ('windowGeometry', 'windowState')
        }
        self.setDockOptions(self.DockOption.AllowNestedDocks | self.DockOption.AllowTabbedDocks)
        self.workspace_tabs = QTabWidget()
        self.workspace_tabs.setObjectName('workspaceTabs')
        self.setCentralWidget(self.workspace_tabs)
        self.central_widget = self.workspace_tabs

        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.setObjectName('workspaceSplitter')
        self.workspace_splitter.setChildrenCollapsible(False)
        self.image_splitter = QSplitter(Qt.Orientation.Vertical)
        self.image_splitter.setObjectName('imageSplitter')
        self.image_splitter.setChildrenCollapsible(False)
        original_panel = QGroupBox('Original source')
        original_layout = QVBoxLayout(original_panel)
        original_layout.addWidget(self.original_label)
        self.original_label.setAccessibleName('Original source image')
        for label in (self.original_label, self.preview_label):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            label.setStyleSheet('background: #202829; color: #dce8e4; border-radius: 4px;')
        self.image_splitter.addWidget(original_panel)

        mask_panel = QGroupBox('Processed image + subject mask')
        mask_layout = QVBoxLayout(mask_panel)
        self.mask_preview = self.preview_label
        self.mask_preview.setAccessibleName('Processed image with accepted subject mask')
        self.mask_preview.setStyleSheet('background: #202829; color: #dce8e4; border-radius: 4px;')
        mask_layout.addWidget(self.mask_preview, 1)
        self.mask_status = _text('No mask. All pixels are available to the depth mesh.')
        mask_layout.addWidget(self.mask_status)
        mask_tools = QHBoxLayout()
        self.mask_edit_button = QPushButton('Edit mask…')
        self.mask_edit_button.setToolTip('Edit the foreground selection used by Depth Mesh.')
        self.mask_edit_button.clicked.connect(self.edit_subject_mask)
        self.mask_clear_button = QPushButton('Clear')
        self.mask_clear_button.clicked.connect(self.clear_subject_mask)
        self.mask_view_mode = QComboBox()
        self.mask_view_mode.addItems(['Overlay', 'Mask', 'Processed'])
        self.mask_view_mode.setAccessibleName('Mask display mode')
        self.mask_view_mode.currentTextChanged.connect(self._refresh_mask_preview)
        for widget in (self.mask_edit_button, self.mask_clear_button, self.mask_view_mode):
            mask_tools.addWidget(widget)
        mask_layout.addLayout(mask_tools)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel('Overlay'))
        self.mask_opacity = QSlider(Qt.Orientation.Horizontal)
        self.mask_opacity.setRange(0, 100)
        self.mask_opacity.setValue(40)
        self.mask_opacity.setAccessibleName('Mask overlay opacity')
        self.mask_opacity.valueChanged.connect(self._refresh_mask_preview)
        opacity_row.addWidget(self.mask_opacity)
        mask_layout.addLayout(opacity_row)
        self.image_splitter.addWidget(mask_panel)
        self.workspace_splitter.addWidget(self.image_splitter)

        mesh_panel = QWidget()
        self.mesh_preview_layout = QVBoxLayout(mesh_panel)
        self.mesh_preview_layout.setContentsMargins(6, 6, 6, 6)
        self.mesh_context = _text('1  Load an image    →    2  Select the subject    →    3  Generate a mesh')
        self.mesh_preview_layout.addWidget(self.mesh_context)
        self.mesh_placeholder = _text(
            'Your 3D mesh will appear here.\n\n'
            'Depth Mesh creates a relief from a photograph. Contour Mesh uses image edges without downloading an AI model.\n\n'
            'Use Setup for model choices and a quick start.'
        )
        self.mesh_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mesh_preview_layout.addWidget(self.mesh_placeholder, 1)
        mesh_panel.setMinimumSize(250, 180)
        self.workspace_splitter.addWidget(mesh_panel)
        self.workspace_splitter.setStretchFactor(0, 2)
        self.workspace_splitter.setStretchFactor(1, 3)
        self.workspace_tabs.addTab(self.workspace_splitter, 'Workspace')

        controls = QTabWidget()
        self.parameter_tabs = controls
        controls.setAccessibleName('Mesh and image parameters')
        controls.addTab(_scroll(mesh_controls), 'Mesh')
        controls.addTab(_scroll(image_controls), 'Image')
        self.controls_dock = QDockWidget('Parameters', self)
        self.controls_dock.setObjectName('parametersDock')
        self.controls_dock.setWidget(controls)
        self.controls_dock.setMinimumWidth(260)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.controls_dock)

        self.workspace_toolbar = QToolBar('Workflow', self)
        self.workspace_toolbar.setObjectName('workflowToolbar')
        self.addToolBar(self.workspace_toolbar)
        self.load_button.setText('Open image…')
        self.generate_mesh_button.setText('Contour Mesh')
        self.process_button.setText('Depth Mesh')
        self.process_button.setStyleSheet('QPushButton { background: #176b55; color: white; padding: 6px 12px; font-weight: 600; }')
        for button in (self.load_button, self.save_button, self.process_button,
                       self.generate_mesh_button, self.export_mesh_button):
            self.workspace_toolbar.addWidget(button)
        self.workspace_toolbar.addSeparator()
        self.workspace_toolbar.addAction('Setup', lambda: self.workspace_tabs.setCurrentWidget(self.setup_page))
        self.workspace_toolbar.addAction('Reset parameters', self.reset_defaults)
        # The old layout created this button, but its action now lives in the toolbar.
        self.reset_defaults_button.hide()
        self.workspace_tabs.currentChanged.connect(self._workspace_tab_changed)
        self.workspace_splitter.setSizes([420, 650])
        self.image_splitter.setSizes([280, 300])
        self._default_workspace_state = self.saveState()

    def _finish_workspace(self):
        from history_panel import HistoryPanel
        self.use_processed_image_checkbox.setToolTip(
            'Use the processed image as the depth model input. Photographs usually provide stronger depth cues than an edge-only canvas. Leave this off to infer from the original photograph.'
        )
        self.history_panel = HistoryPanel()
        self.history_panel.restore_requested.connect(self._restore_history_index)
        self.workspace_tabs.addTab(self.history_panel, 'History')
        self.setup_page = self._make_setup_page()
        self.workspace_tabs.addTab(self.setup_page, 'Setup')
        view_menu = self.menuBar().addMenu('&View')
        view_menu.addAction(self.controls_dock.toggleViewAction())
        view_menu.addAction(self._error_dock.toggleViewAction())
        view_menu.addAction(self.workspace_toolbar.toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction('Reset workspace layout', self.reset_workspace_layout)
        for label, index in [('Workspace', 0), ('History', 1), ('Setup', 2)]:
            view_menu.addAction(label, lambda checked=False, i=index: self.workspace_tabs.setCurrentIndex(i))
        for name, splitter in [('workspace_splitter', self.workspace_splitter), ('image_splitter', self.image_splitter)]:
            saved = self.config.get('Workspace', name, fallback='')
            if saved:
                splitter.restoreState(QByteArray.fromHex(saved.encode()))
        mode = self.config.get('Workspace', 'mask_view_mode', fallback='Overlay')
        if mode == 'Source':
            mode = 'Processed'
        self.mask_view_mode.setCurrentText(mode)
        self.mask_opacity.setValue(self.config.getint('Workspace', 'mask_opacity', fallback=40))
        self.parameter_tabs.setCurrentIndex(max(0, min(1, self.config.getint('Workspace', 'parameter_tab', fallback=0))))
        self._workspace_ready = True
        self._refresh_workspace()
        if self.image is None:
            self.workspace_tabs.setCurrentWidget(self.setup_page)

    def _make_setup_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel('Start with one image')
        title.setStyleSheet('font-size: 22px; font-weight: 600;')
        layout.addWidget(title)
        layout.addWidget(_text(
            'Open a photograph, edit the subject mask, then generate a depth mesh. '
            'Green marks the pixels kept in the mesh. A single photo creates a relief; the hidden side is not reconstructed.'
        ))
        row = QHBoxLayout()
        for label, callback in [('Open image…', lambda: self.load_image()),
                                ('Open project…', self.open_project),
                                ('Try example', self._load_workspace_example)]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        local = QGroupBox('Local models')
        form = QVBoxLayout(local)
        form.addWidget(_text(
            'Depth Anything 3 Small, Base, Mono-Large and Metric-Large are available with Apache 2.0 weights after optional runtime setup. '
            'DepthAnythingV2 creates detailed photo reliefs under noncommercial terms. Depth Pro weights are restricted to research use. '
            'SAM2 selects subjects from clicks and boxes; manual brushes work immediately without model weights. '
            'MiDaS and DPT require registered local source and checkpoints.'
        ))
        allow = QCheckBox('Allow model downloads when I request inference')
        allow.setChecked(self.download_action.isChecked())
        allow.toggled.connect(self.download_action.setChecked)
        self.download_action.toggled.connect(allow.setChecked)
        form.addWidget(allow)
        self.model_license_status = QLabel()
        self.model_license_status.setWordWrap(True)
        self.model_license_status.setAccessibleName('Selected model use restrictions')
        form.addWidget(self.model_license_status)
        form.addWidget(_text('Model files can be large. Downloads contact the model host; local inference keeps the image on this computer.'))
        from edgemesh_bootstrap.resources import resource_path
        da3_setup_url = resource_path('docs/Depth_Anything_3.html').resolve().as_uri()
        depth_help = QLabel(f'<a href="{da3_setup_url}">Set up Depth Anything 3</a> · After setup, choose a Depth Anything 3 model and generate a depth mesh. Metric-Large depth is normalized to relief scale.')
        depth_help.setWordWrap(True)
        depth_help.setOpenExternalLinks(True)
        depth_help.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        depth_help.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        form.addWidget(depth_help)
        actions = QHBoxLayout()
        for label, callback in [('Model details and licenses…', self.model_details),
                                ('Register MiDaS / DPT…', self.register_midas),
                                ('Unload models', self.unload_models)]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            actions.addWidget(button)
        form.addLayout(actions)
        layout.addWidget(local)
        layout.addWidget(self._make_project_settings())
        hardware = QGroupBox('This computer')
        hardware_layout = QFormLayout(hardware)
        try:
            torch_version = package_version('torch')
            runtime = f'PyTorch {torch_version} · ' + (
                'CPU build; large models may run slowly' if '+cpu' in torch_version
                else 'CPU / accelerator selection occurs when inference starts'
            )
        except PackageNotFoundError:
            runtime = 'PyTorch is not installed. Manual masks and contour tools remain available.'
        hardware_layout.addRow('Inference', _text(runtime))
        hardware_layout.addRow('3D preview', _text('Embedded VTK preview initializes when the first mesh is ready.'))
        hardware_layout.addRow('Saved state', _text('Projects save automatically to the chosen project folder. App settings, logs and credentials stay in your user profile.'))
        layout.addWidget(hardware)
        layout.addWidget(_text(
            'Quick offline start: Open image → Contour Mesh. For a photo relief, allow a model download when needed, choose DepthAnythingV2 and click Depth Mesh. '
            'History restores settings and masks; regenerate to update geometry. Projects retain the last accepted mesh between launches.'
        ))
        layout.addStretch()
        return _scroll(page)

    def _load_workspace_example(self):
        from edgemesh_bootstrap.resources import resource_path
        example = resource_path('Images/example.png')
        if not example.is_file():
            self.show_error('The example image is not installed. Open one of your own images to start.')
            return
        self.load_image(str(example))

    def _workspace_tab_changed(self, index):
        if getattr(self, '_workspace_ready', False):
            self.controls_dock.setEnabled(index == 0)
            if index == 1:
                self._history_timer.stop()
                self._record_session()
            if hasattr(self, 'history_panel'):
                self.history_panel.refresh(self.session_history)

    def reset_workspace_layout(self):
        self.restoreState(self._default_workspace_state)
        self.controls_dock.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.controls_dock)
        self.controls_dock.show()
        self._error_dock.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._error_dock)
        self._error_dock.hide()
        self.workspace_toolbar.show()
        self.workspace_splitter.setSizes([420, 650])
        self.image_splitter.setSizes([280, 300])
        self.workspace_tabs.setCurrentIndex(0)

    def _save_workspace_settings(self):
        if not getattr(self, '_workspace_ready', False):
            return
        if not self.config.has_section('Workspace'):
            self.config.add_section('Workspace')
        for name in ('workspace_splitter', 'image_splitter'):
            self.config.set('Workspace', name, bytes(getattr(self, name).saveState().toHex()).decode())
        self.config.set('Workspace', 'mask_view_mode', self.mask_view_mode.currentText())
        self.config.set('Workspace', 'mask_opacity', str(self.mask_opacity.value()))
        self.config.set('Workspace', 'parameter_tab', str(self.parameter_tabs.currentIndex()))
        self.config.set('Workspace', 'active_tab', str(self.workspace_tabs.currentIndex()))

    def _refresh_mask_preview(self, *args):
        if not hasattr(self, 'mask_preview'):
            return
        has_image = self.image is not None
        has_mask = self._subject_mask is not None
        self.mask_edit_button.setEnabled(has_image and not self._model_work_active())
        self.mask_clear_button.setEnabled(has_mask and not self._model_work_active())
        if not has_image:
            self._mask_preview_key = None
            self.mask_preview.clear()
            self.mask_preview.setText('Load an image to select its subject.')
            self.mask_status.setText('No image loaded.')
            return
        base = self.processed_image if self.processed_image is not None else self.image
        key = (id(self.image), id(base), id(self._subject_mask), self.mask_view_mode.currentText(), self.mask_opacity.value())
        if key == getattr(self, '_mask_preview_key', None):
            return
        from subject_mask import mask_preview_bgr
        mode = 'Source' if self.mask_view_mode.currentText() == 'Processed' else self.mask_view_mode.currentText()
        preview = mask_preview_bgr(base, self._subject_mask,
            mode=mode, opacity=self.mask_opacity.value() / 100)
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.mask_preview.setPixmap(QPixmap.fromImage(image))
        self.mask_status.setText(
            f'Foreground kept: {100 * self._subject_mask.mean():.1f}% · Applies to Depth Mesh.'
            if has_mask else 'No mask. Use Edit mask to select a subject with SAM2 or brushes.'
        )
        self._mask_preview_key = key

    def _refresh_workspace(self):
        if not getattr(self, '_workspace_ready', False):
            return
        self._refresh_mask_preview()
        self.history_panel.refresh(self.session_history)
        if hasattr(self, '_refresh_product_state'):
            self._refresh_product_state()
        self.setWindowTitle('EdgeMesh' + self._project_title())
        has_image = self.image is not None
        self.save_button.setEnabled(has_image)
        self.process_button.setEnabled(has_image and not self._jobs.busy)
        self.generate_mesh_button.setEnabled(has_image and not self._jobs.busy)
        self.export_mesh_button.setEnabled(self._current_mesh() is not None)
        if self._current_mesh() is not None:
            self.mesh_context.setText('Displayed mesh is the last generated result. Regenerate after changing the image, settings or mask.')

    def _restore_history_index(self, index):
        self._history_timer.stop()
        try:
            document = self.session_history.snapshot_at(index)
            if self._restore_document(document):
                self.session_history.select(index)
                self.undo_action.setEnabled(self.session_history.can_undo)
                self.redo_action.setEnabled(self.session_history.can_redo)
                self._refresh_workspace()
                self.workspace_tabs.setCurrentIndex(0)
                self._request_project_save()
        except Exception as error:
            self.show_error(str(error))

    def _install_embedded_viewport(self):
        from embedded_viewport import EmbeddedMeshViewport
        viewport = EmbeddedMeshViewport(parent=self)
        viewport.error_occurred.connect(self.show_error)
        viewport.view_state_changed.connect(self._request_project_save)
        self.mesh_preview_layout.addWidget(viewport, 1)
        return viewport
