"""Model-specific notices and provenance for the mesh actually accepted by the UI."""
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QLabel,
    QPlainTextEdit, QVBoxLayout,
)

from model_licensing import ProvenanceStore, canonical_model_identity, policy_for
from user_state import atomic_write


def _policy_text(policy):
    """Only verified catalog URLs are placed into clickable markup."""
    def esc(value):
        return html.escape(str(value or ''))
    sections = [f"<b>{esc(policy.get('display_name'))}</b>",
                f"{esc(policy.get('badge'))} · {esc(policy.get('license'))}",
                esc(policy.get('guidance'))]
    if policy.get('license_url'):
        sections.append(f'<a href="{esc(policy["license_url"])}">Read the official model license</a>')
    if policy.get('source_url') and policy['source_url'] != policy.get('license_url'):
        sections.append(f'<a href="{esc(policy["source_url"])}">Official model source</a>')
    sections.append(esc(policy.get('commercial_info')))
    if policy.get('commercial_url'):
        sections.append(f'<a href="{esc(policy["commercial_url"])}">Commercial licensing information</a>')
    sections.append(esc(policy.get('notice')))
    return '<p>' + '</p><p>'.join(value for value in sections if value) + '</p>'


class ModelLicenseDialog(QDialog):
    def __init__(self, policy, parent=None):
        super().__init__(parent)
        self.setObjectName('modelLicenseNotice')
        self.setWindowTitle('Review this model before preparation')
        self.resize(600, 460)
        layout = QVBoxLayout(self)
        details = QLabel(_policy_text(policy))
        details.setWordWrap(True)
        details.setOpenExternalLinks(True)
        details.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        details.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(details)
        revision = policy.get('revision')
        note = QLabel(
            f'This acknowledgment applies to model revision {revision}.' if revision else
            'This acknowledgment applies once to the revision selected for this preparation. Later revisions require review.'
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.acknowledgment = QCheckBox('&I understand these restrictions and will use the model accordingly.')
        self.acknowledgment.setAccessibleName('Acknowledge the model use restrictions')
        layout.addWidget(self.acknowledgment)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.continue_button = self.buttons.addButton('Acknowledge and &continue', QDialogButtonBox.ButtonRole.AcceptRole)
        self.continue_button.setDefault(True)
        self.continue_button.setEnabled(False)
        self.acknowledgment.toggled.connect(self.continue_button.setEnabled)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)


class ModelComplianceMixin:
    def _init_compliance(self):
        self.provenance_store = ProvenanceStore(self.paths.root)
        self._accepted_provenance = None
        self._accepted_provenance_check = {'status': 'unverified', 'reason': 'No generated mesh.'}
        self._provenance_mesh_token = None
        self._mesh_before_repair_provenance = None
        self._subject_mask_provenance_trusted = False

    def _ensure_model_license(self, model_type):
        download_action = getattr(self, 'download_action', None)
        if download_action is not None and download_action.isChecked() is False:
            return True
        request = self.model_store.consent_request(model_type)
        if not request.get('consent_needed'):
            return True
        dialog = ModelLicenseDialog(request, self)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self.statusBar().showMessage('Model preparation cancelled. No download was started.')
                return False
            self.model_store.accept_license(model_type, revision=request.get('revision'))
            return True
        finally:
            dialog.deleteLater()

    def _planned_policies(self):
        result = [policy_for(self.depth_method_dropdown.currentText()).as_dict()]
        if self._subject_mask is not None and self._subject_mask_info:
            identity = canonical_model_identity(self._subject_mask_info)
            result.append(identity.get('license_policy') or policy_for(identity.get('model_id', '')).as_dict())
        return result

    def _refresh_license_indicators(self):
        if not hasattr(self, 'depth_method_dropdown'):
            return
        for index in range(self.depth_method_dropdown.count()):
            policy = policy_for(self.depth_method_dropdown.itemText(index)).as_dict()
            usage = policy.get('usage_class', 'unknown')
            color = '#8a4b00' if usage == 'noncommercial' else '#68468b' if usage in ('custom', 'unknown') else '#176b55'
            self.depth_method_dropdown.setItemData(index, QBrush(QColor(color)), Qt.ItemDataRole.ForegroundRole)
            self.depth_method_dropdown.setItemData(index, f"{policy.get('badge')}: {policy.get('guidance')}", Qt.ItemDataRole.ToolTipRole)
        policies = self._planned_policies()
        nc = any(policy.get('usage_class') == 'noncommercial' for policy in policies)
        research = any(policy.get('restriction') == 'research-only' for policy in policies)
        uncertain = any(policy.get('usage_class') in ('custom', 'unknown') for policy in policies)
        color = '#8a5200' if nc else '#68468b' if uncertain else '#176b55'
        suffix = ' · NC research' if research else ' · NC' if nc else ' · Check terms' if uncertain else ''
        self.process_button.setText('Depth Mesh' + suffix)
        self.process_button.setStyleSheet(f'QPushButton {{ background: {color}; color: white; padding: 6px 12px; font-weight: 600; }}')
        self.process_button.setAccessibleName('Generate depth mesh' + suffix)
        description = '\n\n'.join(str(p.get('guidance', 'Review the model terms.')) for p in policies)
        self.process_button.setToolTip('Generate using the selected depth model. Alt+D or Shift+Enter.\n' + description)
        self.depth_method_dropdown.setToolTip(description)
        if hasattr(self, 'model_license_status'):
            self.model_license_status.setText(' · '.join(str(p.get('badge', 'Unverified terms')) for p in policies))
            self.model_license_status.setStyleSheet(f'color: {color}; font-weight: 600;')
        # Contour generation does not invoke the depth or subject-mask models.
        self.generate_mesh_button.setText('Contour Mesh')
        self.generate_mesh_button.setToolTip('Generate from image edges without a learned model. Alt+M.')
        self._refresh_mesh_provenance_badge()

    def _seal_accepted_mesh(self, model_info, mask_info=None, settings=None):
        mesh = self._current_mesh()
        if mesh is None:
            return
        import numpy as np
        identities = []
        if model_info:
            identities.append(canonical_model_identity(model_info))
        if mask_info:
            identities.append(canonical_model_identity(mask_info))
        parameters = deepcopy(settings or {})
        if mask_info and not self._subject_mask_provenance_trusted:
            parameters['input_mask_provenance'] = 'unverified'
        self._accepted_provenance = self.provenance_store.seal(
            np.asarray(mesh.vertices), np.asarray(mesh.triangles), identities,
            parameters=parameters,
        )
        self._provenance_mesh_token = None
        self._refresh_mesh_provenance_badge()

    def _refresh_mesh_provenance_badge(self):
        viewport = getattr(self, 'three_d_viewport', None)
        mesh = self._current_mesh() if viewport is not None else None
        if mesh is None or not hasattr(viewport, 'set_provenance_status'):
            return
        token = (id(mesh), id(self._accepted_provenance))
        if token != self._provenance_mesh_token:
            import numpy as np
            self._accepted_provenance_check = self.provenance_store.verify(
                self._accepted_provenance, np.asarray(mesh.vertices), np.asarray(mesh.triangles))
            self._provenance_mesh_token = token
        check = self._accepted_provenance_check
        record = self._accepted_provenance if isinstance(self._accepted_provenance, dict) else {}
        payload = record.get('payload', {})
        payload = payload if isinstance(payload, dict) else {}
        models = payload.get('models', [])
        models = models if isinstance(models, list) else []
        policies = [canonical_model_identity(value).get('license_policy', {}) for value in models if isinstance(value, dict)]
        nc = any(p.get('usage_class') == 'noncommercial' for p in policies)
        research = any(p.get('restriction') == 'research-only' for p in policies)
        parameters = payload.get('parameters', {})
        input_unverified = isinstance(parameters, dict) and parameters.get('input_mask_provenance') == 'unverified'
        verified = check.get('status') == 'verified' and not input_unverified
        usage = 'noncommercial' if nc else 'permissive' if verified and all(p.get('usage_class') == 'permissive' for p in policies) else 'unknown'
        label = 'NC · research only' if research else 'NC model used' if nc else 'Model terms recorded' if models else 'No learned model used' if verified else 'Model provenance unavailable'
        names = ', '.join(str(p.get('display_name', 'Unknown model')) for p in policies)
        trust = 'Verified on this computer' if verified else 'Input mask model unverified' if input_unverified else 'Changed metadata or geometry' if check.get('status') == 'tampered' else 'Unverified provenance'
        viewport.set_provenance_status(' · '.join(part for part in (label, names, trust) if part), usage_class=usage, verified=verified)

    def _mesh_metadata_for_project(self):
        if self._current_mesh() is None:
            return None
        save_view = getattr(self.three_d_viewport, 'save_view_state', None)
        return {'provenance': deepcopy(self._accepted_provenance),
                'view_state': save_view() if save_view is not None else {}}

    def _seal_mesh_repair(self, previous_mesh, previous_provenance):
        """Only a verified parent can establish model lineage for a local edit."""
        import numpy as np
        checked = self.provenance_store.verify(previous_provenance,
            np.asarray(previous_mesh.vertices), np.asarray(previous_mesh.triangles))
        self._accepted_provenance = deepcopy(previous_provenance)
        if checked.get('status') == 'verified':
            payload = previous_provenance['payload']
            parameters = deepcopy(payload.get('parameters', {}))
            parameters['local_edit'] = {'operation': 'mesh_repair', 'parent_signature': previous_provenance.get('signature')}
            mesh = self._current_mesh()
            self._accepted_provenance = self.provenance_store.seal(
                np.asarray(mesh.vertices), np.asarray(mesh.triangles), payload['models'], parameters=parameters)
        self._provenance_mesh_token = None
        self._refresh_mesh_provenance_badge()

    def _write_export_provenance(self, path):
        target = Path(path)
        digest = hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        file_digest = digest.hexdigest()
        provenance = deepcopy(self._accepted_provenance)
        binding = 'unverified'
        mesh = self._current_mesh()
        if mesh is not None:
            import numpy as np
            checked = self.provenance_store.verify(provenance,
                np.asarray(mesh.vertices), np.asarray(mesh.triangles))
            if checked.get('status') == 'verified':
                parameters = deepcopy(provenance['payload'].get('parameters', {}))
                parameters.update({'export_file_sha256': file_digest,
                    'export_format': target.suffix.lower(),
                    'geometry_reference': 'accepted_mesh_before_format_conversion',
                    'parent_signature': provenance.get('signature')})
                provenance = self.provenance_store.seal(np.asarray(mesh.vertices),
                    np.asarray(mesh.triangles), provenance['payload']['models'], parameters=parameters)
                if self.provenance_store.verify(provenance).get('status') == 'verified':
                    binding = 'signed'
        record = {'format': 'edgemesh-export-provenance', 'version': 2,
                  'mesh_file': target.name, 'mesh_file_sha256': file_digest,
                  'export_binding': binding, 'provenance': provenance,
                  'notice': 'When export_binding is signed, the exact exported file hash is authenticated inside provenance.payload.parameters.export_file_sha256. The geometry digest refers to the accepted mesh before format conversion; OBJ/STL may round or reorder coordinates. Unverified ancestry remains unverified. This is not DRM.'}
        atomic_write(target.with_name(target.name + '.edgemesh.json'), json.dumps(record, indent=2, allow_nan=False).encode('utf-8'))

    def show_mesh_provenance(self):
        self._refresh_mesh_provenance_badge()
        dialog = QDialog(self)
        dialog.setObjectName('meshProvenance')
        dialog.setWindowTitle('Accepted mesh provenance')
        dialog.resize(700, 550)
        layout = QVBoxLayout(dialog)
        status = QLabel(str(self._accepted_provenance_check.get('reason', '')) + '\nLocal signatures detect changes. They cannot prevent edits by someone who controls this computer.')
        status.setWordWrap(True)
        layout.addWidget(status)
        details = QPlainTextEdit()
        details.setReadOnly(True)
        details.setAccessibleName('Mesh model provenance JSON')
        details.setPlainText(json.dumps(self._accepted_provenance, indent=2, ensure_ascii=False))
        layout.addWidget(details)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def model_details(self):
        from model_store import HF_MODELS
        dialog = QDialog(self)
        dialog.setObjectName('modelLicenseCatalog')
        dialog.setWindowTitle('Model licenses and preparation')
        dialog.resize(650, 500)
        layout = QVBoxLayout(dialog)
        selector = QComboBox()
        selector.setAccessibleName('Model to inspect')
        keys = list(HF_MODELS) + ['midas', 'dpt']
        for key in keys:
            policy = policy_for(key).as_dict()
            selector.addItem(f"{policy.get('display_name', key)} · {policy.get('badge', 'Check terms')}", key)
        layout.addWidget(selector)
        details = QLabel()
        details.setWordWrap(True)
        details.setOpenExternalLinks(True)
        details.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        details.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(details, 1)
        selector.currentIndexChanged.connect(lambda index: details.setText(_policy_text(policy_for(selector.itemData(index)).as_dict())))
        details.setText(_policy_text(policy_for(selector.currentData()).as_dict()))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()
