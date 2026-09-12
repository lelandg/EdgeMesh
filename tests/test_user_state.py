import configparser
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from user_state import UserPaths, export_diagnostics, migrate_config, save_config
from log_utils import setup_logger


class UserStateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.environment = patch.dict("os.environ", {"EDGEMESH_DATA_DIR": str(self.root / "user")})
        self.environment.start()
        self.paths = UserPaths.discover()

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def test_discovery_separates_user_state_from_install(self):
        self.assertEqual(self.paths.root, self.root / "user")
        for path in (self.paths.presets_dir, self.paths.work_dir, self.paths.logs_dir):
            self.assertTrue(path.is_dir())

    def test_valid_legacy_copied_once_and_original_preserved(self):
        legacy = self.root / "legacy.ini"
        original = "[Settings]\nlast_used_image = source.png\n"
        legacy.write_text(original)
        config = migrate_config(legacy, self.paths)
        self.assertEqual(config["Settings"]["last_used_image"], "source.png")
        self.assertEqual(legacy.read_text(), original)
        legacy.write_text("[Settings]\nlast_used_image = newer.png\n")
        self.assertEqual(migrate_config(legacy, self.paths)["Settings"]["last_used_image"], "source.png")

    def test_bad_legacy_can_retry_after_repair(self):
        legacy = self.root / "broken.ini"
        legacy.write_text("not an ini file")
        with patch("user_state._error") as error:
            result = migrate_config(legacy, self.paths)
        self.assertTrue(result.has_section("Settings"))
        self.assertFalse(self.paths.config_file.exists())
        error.assert_called_once()
        legacy.write_text("[Settings]\nlast_used_image = repaired.png\n")
        self.assertEqual(migrate_config(legacy, self.paths)["Settings"]["last_used_image"], "repaired.png")

    def test_atomic_failure_preserves_previous_config(self):
        self.paths.config_file.write_text("old content")
        config = configparser.ConfigParser()
        config["Settings"] = {"last_used_image": "new.png"}
        with patch("user_state.os.replace", side_effect=PermissionError("read only")), patch("user_state._error") as error:
            with self.assertRaises(PermissionError):
                save_config(config, self.paths.config_file)
        self.assertEqual(self.paths.config_file.read_text(), "old content")
        self.assertEqual(list(self.paths.root.glob(".config.ini.*")), [])
        error.assert_called_once()

    def test_default_config_valid_and_export_contains_no_private_content(self):
        config = migrate_config(None, self.paths)
        config["Settings"]["api_key"] = "very-private-key"
        config["Settings"]["last_used_image"] = "C:/Private/person.jpg"
        save_config(config, self.paths.config_file)
        (self.paths.logs_dir / "private-user.log").write_text("2026-01-01 - app - ERROR - password=very-private-key C:/Private/person.jpg")
        (self.paths.work_dir / "image.png").write_bytes(b"private image")
        target = self.root / "diagnostics.zip"
        self.assertEqual(export_diagnostics(target, self.paths), target)
        with zipfile.ZipFile(target) as archive:
            content = " ".join(archive.read(name).decode() for name in archive.namelist())
            self.assertNotIn("very-private-key", content)
            self.assertNotIn("person.jpg", content)
            self.assertNotIn("private-user", " ".join(archive.namelist()))
            self.assertNotIn("image.png", " ".join(archive.namelist()))
            summary = json.loads(archive.read("logs/log-000.json"))
            self.assertEqual(summary["levels"]["ERROR"], 1)

    def test_failed_support_export_is_logged(self):
        with patch("user_state.atomic_write", side_effect=PermissionError("read only")), patch("user_state._error") as error:
            with self.assertRaises(PermissionError):
                export_diagnostics(self.root / "support.zip", self.paths)
        error.assert_called_once()

    def test_logs_rotate_with_bounded_backups(self):
        path = self.paths.logs_dir / "rotation.log"
        logger = setup_logger("test.user-state.rotation", path)
        handler = logger.handlers[0]
        handler.maxBytes = 100
        handler.backupCount = 2
        try:
            for index in range(10):
                logger.error("test error number %s", index)
            self.assertTrue(path.is_file())
            self.assertTrue(Path(str(path) + ".1").is_file())
            self.assertTrue(Path(str(path) + ".2").is_file())
            self.assertFalse(Path(str(path) + ".3").exists())
        finally:
            handler.close()
            logger.removeHandler(handler)

    def test_named_default_loggers_have_distinct_safe_paths(self):
        loggers = [setup_logger(name) for name in ("test.paths.one", "test.paths/two", "test.paths_two")]
        try:
            targets = [Path(logger.handlers[0].baseFilename) for logger in loggers]
            self.assertEqual(len(set(targets)), 3)
            self.assertTrue(all(path.parent == self.paths.logs_dir for path in targets))
        finally:
            for logger in loggers:
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
