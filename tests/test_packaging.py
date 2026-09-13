"""Packaging contracts that can run before a distribution build."""
import contextlib
import io
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import types
import unittest
from unittest.mock import Mock, patch

from edgemesh_bootstrap.resources import resource_path

ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_help_and_version_do_not_import_desktop_or_depth_runtime(self):
        code = """import sys
from edgemesh_bootstrap.cli import main
try:
    main(sys.argv[1:])
except SystemExit as exc:
    assert exc.code == 0
assert not any(name.split('.')[0] in {'torch', 'PySide6', 'transformers'} for name in sys.modules)
"""
        for argument, expected in (("--help", "Create 3D meshes"), ("--version", "EdgeMesh ")):
            with self.subTest(argument=argument):
                result = subprocess.run([sys.executable, "-c", code, argument], cwd=ROOT,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)

    def test_resources_exist_and_reject_escape_paths(self):
        self.assertTrue(resource_path("Images/example.png").is_file())
        self.assertTrue(resource_path("EdgeMesh.ico").is_file())
        self.assertTrue(resource_path("docs/Depth_Anything_3.html").is_file())
        with self.assertRaises(ValueError):
            resource_path("../version.py")
        with self.assertRaises(ValueError):
            resource_path(str(ROOT / "version.py"))
        if sys.platform == "win32":
            with self.assertRaises(ValueError):
                resource_path("C:outside.png")
        with self.assertRaises(FileNotFoundError):
            resource_path("missing-resource.png")

    def test_da3_setup_page_has_portable_copyable_commands(self):
        class Commands(HTMLParser):
            def __init__(self):
                super().__init__()
                self.commands = {}
                self.buttons = set()
                self.current = None

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "pre":
                    self.current = attrs["id"]
                    self.commands[self.current] = ""
                elif tag == "button" and "data-copy" in attrs:
                    self.buttons.add(attrs["data-copy"])

            def handle_data(self, data):
                if self.current is not None:
                    self.commands[self.current] += data

            def handle_endtag(self, tag):
                if tag == "pre":
                    self.current = None

        page = Commands()
        page.feed(resource_path("docs/Depth_Anything_3.html").read_text(encoding="utf-8"))
        self.assertEqual(set(page.commands), page.buttons)
        self.assertIn("python -m da3_setup", page.commands.values())
        self.assertIn("python -m edgemesh_bootstrap", page.commands.values())
        for command in page.commands.values():
            with self.subTest(command=command):
                self.assertNotRegex(command, r"[A-Za-z]:[\\/]|/Users/|/home/|\.worktrees|<[^>]+>")

    def test_metadata_keeps_depth_optional_and_version_single_sourced(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = metadata["project"]
        self.assertEqual(project["scripts"]["edgemesh"], "edgemesh_bootstrap.cli:main")
        self.assertEqual(project["requires-python"], ">=3.12,<3.13")
        self.assertNotIn("version", project)
        self.assertEqual(metadata["tool"]["setuptools"]["dynamic"]["version"],
                         {"attr": "version.__version__"})
        self.assertTrue(all("https://" not in item for item in project["dependencies"]))
        self.assertFalse(any(item.startswith(("torch", "transformers")) for item in project["dependencies"]))
        constraints = (ROOT / "constraints" / "windows-py312.txt").read_text(encoding="utf-8").splitlines()
        depth_dependencies = project["optional-dependencies"]["depth"]
        for package in ("torch", "torchvision"):
            with self.subTest(package=package):
                expected = [line.strip() for line in constraints if line.strip().startswith(f"{package}==")]
                self.assertEqual(len(expected), 1, f"Expected one exact {package} constraint")
                self.assertEqual([item for item in depth_dependencies if item.startswith(f"{package}==")],
                                 expected)
        self.assertIn("google-antigravity==0.1.16", project["optional-dependencies"]["assistants"])

    def test_setup_metadata_does_not_require_source_on_sys_path(self):
        result = subprocess.run(
            [sys.executable, "-I", str(ROOT / "setup.py"), "--name"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "edgemesh")

    def test_source_manifest_preserves_readme_case_and_excludes_tests(self):
        with tempfile.TemporaryDirectory(prefix="edgemesh-metadata-") as directory:
            result = subprocess.run(
                [sys.executable, "-I", str(ROOT / "setup.py"), "egg_info", "--egg-base", directory],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = (Path(directory) / "edgemesh.egg-info" / "SOURCES.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            manifest = [Path(name).as_posix() for name in manifest]
        self.assertIn("README.md", manifest)
        self.assertNotIn("ReadMe.md", manifest)
        self.assertFalse(any(name.startswith("tests/") for name in manifest))
        self.assertIn("MeshTools/LICENSE", manifest)
        self.assertIn("Images/example.png", manifest)
        self.assertIn("docs/Depth_Anything_3.html", manifest)

    def test_packaging_excludes_developer_scripts(self):
        from _build_support import application_modules
        modules = application_modules()
        self.assertIn("edge_mesh", modules)
        self.assertIn("model_store", modules)
        self.assertTrue({"da3_backend", "da3_worker", "da3_setup"}.issubset(modules))
        self.assertIn("depth_diagnostics", modules)
        self.assertNotIn("setup", modules)
        self.assertNotIn("torch_test", modules)
        self.assertNotIn("install_open3d", modules)

    def test_depth_command_forwards_arguments_and_exit_code(self):
        from edgemesh_bootstrap.cli import main
        diagnostics = types.ModuleType("depth_diagnostics")
        diagnostics.main = Mock(return_value=7)
        with patch.dict(sys.modules, {"depth_diagnostics": diagnostics}):
            result = main(["diagnose-depth", "--width", "64"])
        self.assertEqual(result, 7)
        diagnostics.main.assert_called_once_with(["--width", "64"])

    def test_missing_depth_runtime_is_logged_without_stdout_noise(self):
        from edgemesh_bootstrap.cli import main
        diagnostics = types.ModuleType("depth_diagnostics")
        diagnostics.main = Mock(side_effect=ModuleNotFoundError("No module named torch", name="torch"))
        logger_module = types.ModuleType("log_utils")
        logger = Mock()
        logger_module.get_logger = Mock(return_value=logger)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {"depth_diagnostics": diagnostics, "log_utils": logger_module}):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = main(["diagnose-depth"])
        self.assertEqual(result, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("[depth]", stderr.getvalue())
        logger.exception.assert_called_once()

    def test_explicit_project_prevents_restoring_the_previous_project(self):
        from edgemesh_bootstrap.cli import main
        project = ROOT / "sample-project"
        for arguments, restore_last in (([], True), ([str(project)], False)):
            with self.subTest(arguments=arguments):
                qt_widgets = types.ModuleType("PySide6.QtWidgets")
                application = Mock()
                application.exec.return_value = 0
                qt_widgets.QApplication = Mock(return_value=application)
                desktop = types.ModuleType("edge_mesh")
                window = Mock()
                desktop.MainWindowImageProcessing = Mock(return_value=window)
                modules = {"PySide6.QtWidgets": qt_widgets, "edge_mesh": desktop}
                with patch.dict(sys.modules, modules):
                    result = main(arguments)
                self.assertEqual(result, 0)
                desktop.MainWindowImageProcessing.assert_called_once_with(
                    verbose=False, restore_last_project=restore_last
                )
                if arguments:
                    window.open_project_path.assert_called_once_with(str(project.absolute()))
                else:
                    window.open_project_path.assert_not_called()
                window.show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
