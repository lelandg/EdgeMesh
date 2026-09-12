"""CLI regressions that stop before model preparation or inference."""
import builtins
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import depth_diagnostics


class StopBeforeInference(RuntimeError):
    pass


def runtime_stubs():
    constructor = Mock(side_effect=StopBeforeInference("Diagnostic test boundary"))

    class FakeDepthTo3D:
        def __init__(self, **kwargs):
            constructor(**kwargs)

    store = object()
    logger = Mock()
    store_factory = Mock(return_value=store)
    modules = {
        "cv2": SimpleNamespace(
            IMREAD_UNCHANGED=-1,
            imread=Mock(return_value=SimpleNamespace(shape=(4, 6, 3))),
        ),
        "torch": SimpleNamespace(set_num_threads=Mock()),
        "trimesh": SimpleNamespace(),
        "data_contracts": SimpleNamespace(proportional_shape=Mock(return_value=(4, 6))),
        "depth_to_3d": SimpleNamespace(DepthTo3D=FakeDepthTo3D),
        "log_utils": SimpleNamespace(get_logger=Mock(return_value=logger)),
        "model_store": SimpleNamespace(ModelStore=store_factory),
    }
    return SimpleNamespace(
        modules=modules, constructor=constructor, store=store,
        store_factory=store_factory, logger=logger,
    )


class DepthDiagnosticsCliTests(unittest.TestCase):
    def run_until_pipeline(self, output_dir, *arguments):
        runtime = runtime_stubs()
        with patch.dict(sys.modules, runtime.modules), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(StopBeforeInference, "Diagnostic test boundary"):
                depth_diagnostics.main([
                    "input.png", "--output-dir", str(output_dir), *arguments,
                ])
        runtime.logger.exception.assert_called_once_with("Offline depth diagnostic failed")
        return runtime

    def test_invalid_amounts_fail_before_runtime_imports(self):
        real_import = builtins.__import__
        runtime_imports = []

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"cv2", "torch", "trimesh", "depth_to_3d", "model_store"}:
                runtime_imports.append(name)
                raise AssertionError("Invalid input reached the depth runtime")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            for amount in ("nan", "inf", "-inf", "-0.1", "100.01"):
                with self.subTest(amount=amount):
                    errors = io.StringIO()
                    with patch("builtins.__import__", side_effect=guarded_import), redirect_stderr(errors):
                        with self.assertRaises(SystemExit) as raised:
                            depth_diagnostics.main([
                                "input.png", "--output-dir", folder,
                                f"--depth-amount={amount}",
                            ])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn("depth-amount", errors.getvalue())
                    self.assertIn("finite", errors.getvalue())
        self.assertEqual(runtime_imports, [])

    def test_boundary_amounts_are_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            for amount in ("0", "100"):
                with self.subTest(amount=amount):
                    self.run_until_pipeline(folder, f"--depth-amount={amount}")

    def test_default_state_uses_application_registrations_and_cpu(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = self.run_until_pipeline(folder, "--model", "midas")
        runtime.store_factory.assert_called_once_with(root=None, cache_dir=None)
        runtime.constructor.assert_called_once_with(
            model_type="midas", verbose=False, model_store=runtime.store, device="cpu",
        )

    def test_explicit_state_cache_and_device_are_forwarded(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "registered-models"
            cache = Path(folder) / "hf-cache"
            runtime = self.run_until_pipeline(
                Path(folder) / "output", "--model", "dpt",
                "--model-state", str(state), "--cache-dir", str(cache),
                "--device", "cuda",
            )
        runtime.store_factory.assert_called_once_with(root=state, cache_dir=cache)
        runtime.constructor.assert_called_once_with(
            model_type="dpt", verbose=False, model_store=runtime.store, device="cuda",
        )


if __name__ == "__main__":
    unittest.main()
