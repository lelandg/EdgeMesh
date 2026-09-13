import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from project_store import ProjectStore
from session_state import SessionDocument, load_session, save_session


class ProjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.source = self.directory / "source.png"
        self.source.write_bytes(b"image fixture")
        self.document = SessionDocument(str(self.source), {"depth_amount": 0.8},
            {"model_id": "fixture", "license": "non-commercial", "sha256": "verified-model-hash"},
            np.array([[True, False], [False, True]]), [{"action": "accepted mask"}])
        self.store = ProjectStore(self.directory / "projects")

    def test_project_is_portable_and_provenance_is_plaintext(self):
        project = self.store.create(self.document, "Example")
        payload = json.loads(project.read_text(encoding="utf-8"))
        self.assertEqual(payload["session"]["model_info"], self.document.model_info)
        self.assertFalse(Path(payload["session"]["source_path"]).is_absolute())
        moved = self.directory / "moved"
        shutil.copytree(project.parent, moved)
        self.source.unlink()
        loaded = self.store.load(moved)
        self.assertTrue(Path(loaded.source_path).is_relative_to(moved))
        self.assertEqual(Path(loaded.source_path).read_bytes(), b"image fixture")
        np.testing.assert_array_equal(loaded.mask, self.document.mask)
        self.assertEqual(loaded.history, self.document.history)

    def test_failed_save_preserves_manifest_and_accepted_state(self):
        project = self.store.create(self.document)
        before = project.read_bytes()
        changed = self.store.current_document
        changed.settings["depth_amount"] = 9.0
        with patch("project_store.atomic_write", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.save_current(changed)
        self.assertEqual(project.read_bytes(), before)
        self.assertEqual(self.store.current_document.settings["depth_amount"], 0.8)
        self.assertEqual(self.store.current_path, project)

    def test_failed_create_preserves_current_project_and_existing_folder(self):
        accepted = self.store.create(self.document)
        selected = self.directory / "existing"
        selected.mkdir()
        preserved = selected / "important.txt"
        preserved.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.store.create(self.document, directory=selected)
        self.assertEqual(preserved.read_text(), "keep")
        with patch("project_store.atomic_write", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.create(self.document)
        self.assertEqual(self.store.current_path, accepted)

    def test_inspect_allows_ui_restore_before_acceptance(self):
        first = self.store.create(self.document)
        another = ProjectStore(self.store.root)
        changed = copy.deepcopy(self.document)
        changed.settings["depth_amount"] = 4.0
        second = another.create(changed)
        candidate = self.store.inspect(second)
        self.assertEqual(self.store.current_path, first)
        self.assertEqual(candidate.document.settings["depth_amount"], 4.0)
        self.store.accept(candidate)
        candidate.document.settings["depth_amount"] = 99.0
        self.assertEqual(self.store.current_document.settings["depth_amount"], 4.0)

    def test_invalid_or_missing_assets_never_replace_current_project(self):
        accepted = self.store.create(self.document)
        other = ProjectStore(self.directory / "other").create(self.document)
        payload = json.loads(other.read_text())
        asset = other.parent / payload["assets"]["source"]["path"]
        asset.unlink()
        with self.assertRaises(FileNotFoundError):
            self.store.load(other)
        self.assertEqual(self.store.current_path, accepted)
        asset.write_bytes(b"corrupt image fixture")
        with self.assertRaisesRegex(ValueError, "changed or is corrupt"):
            self.store.load(other)
        self.assertEqual(self.store.current_path, accepted)

    def test_malformed_json_and_future_formats_do_not_replace_state(self):
        accepted = self.store.create(self.document)
        invalid = self.directory / "broken.json"
        for content in ("{", "null", '{"format": "edgemesh-project", "version": 999}'):
            invalid.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.store.load(invalid)
            self.assertEqual(self.store.current_path, accepted)

    def test_asset_traversal_and_foreign_drive_rejected(self):
        project = self.store.create(self.document)
        original = json.loads(project.read_text())
        for reference in ("../source.png", "/source.png", "C:\\source.png", "C:source.png", "\\\\server\\share\\image.png"):
            payload = copy.deepcopy(original)
            payload["assets"]["source"]["path"] = reference
            payload["session"]["source_path"] = reference
            project.write_text(json.dumps(payload), encoding="utf-8")
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                self.store.inspect(project)

    def test_model_source_and_mask_validation_precede_save(self):
        project = self.store.create(self.document)
        before = project.read_bytes()
        invalid = self.store.current_document
        invalid.settings["depth_amount"] = float("nan")
        with self.assertRaises(ValueError):
            self.store.save_current(invalid)
        self.assertEqual(project.read_bytes(), before)
        invalid = self.store.current_document
        invalid.source_path = str(self.directory / "missing.png")
        with self.assertRaises(FileNotFoundError):
            self.store.save_current(invalid)
        self.assertEqual(project.read_bytes(), before)

    def test_mesh_copy_hash_and_invalidation(self):
        mesh = self.directory / "mesh.ply"
        mesh.write_bytes(b"ply fixture")
        project = self.store.create(self.document, mesh_path=mesh)
        copied = self.store.current_mesh_path
        self.assertEqual(copied.read_bytes(), mesh.read_bytes())
        self.assertTrue(copied.is_relative_to(project.parent))
        loaded = ProjectStore(self.store.root)
        loaded.load(project)
        self.assertEqual(loaded.current_mesh_path, copied)
        self.store.save_current(self.store.current_document, mesh_path=copied)
        self.assertEqual(self.store.current_mesh_path, copied)
        self.store.save_current(self.store.current_document)
        self.assertIsNone(self.store.current_mesh_path)
        self.assertTrue(copied.is_file())  # Existing artifacts are never deleted by autosave.

    def test_legacy_explicit_sessions_remain_compatible(self):
        legacy = self.directory / "legacy.json"
        save_session(legacy, self.document)
        loaded = self.store.load(legacy)
        self.assertEqual(loaded.settings, self.document.settings)
        self.assertTrue(self.store.is_legacy)
        loaded.settings["depth_amount"] = 3.0
        self.store.save_current(loaded)
        self.assertEqual(load_session(legacy).settings["depth_amount"], 3.0)
        self.assertEqual(json.loads(legacy.read_text())["format"], "edgemesh-session")

    def test_new_project_root_does_not_move_current_project(self):
        first = self.store.create(self.document)
        changed_root = self.directory / "new projects"
        self.store.set_root(changed_root)
        self.assertEqual(self.store.current_path, first)
        second = self.store.create(self.document)
        self.assertEqual(second.parent.parent, changed_root)
        self.assertTrue(first.exists())
        self.assertEqual(len(self.store.list_projects()), 1)

    def test_credentials_are_rejected_without_replacing_accepted_state(self):
        project = self.store.create(self.document)
        before = project.read_bytes()
        for field in ("api_key", "OpenAI-API-Key", "access_token", "password", "private_key"):
            document = self.store.current_document
            document.model_info["provider"] = {field: "fixture-not-a-real-credential"}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "credentials"):
                self.store.save_current(document)
            self.assertEqual(project.read_bytes(), before)
        imported = json.loads(project.read_text())
        imported["session"]["history"].append({"refresh_token": "fixture"})
        candidate = project.parent / "imported.json"
        candidate.write_text(json.dumps(imported), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "credentials"):
            self.store.inspect(candidate)
        self.assertEqual(self.store.current_path, project)

    def test_changed_managed_asset_is_not_silently_accepted_on_save(self):
        project = self.store.create(self.document)
        before = project.read_bytes()
        document = self.store.current_document
        Path(document.source_path).write_bytes(b"changed after project creation")
        with self.assertRaisesRegex(ValueError, "changed or is corrupt"):
            self.store.save_current(document)
        self.assertEqual(project.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
