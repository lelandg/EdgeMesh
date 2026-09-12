"""Model store contracts without model weights, Hub access, or GPU allocation."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from model_store import (
    HF_MODELS,
    ModelPreparationCancelled,
    ModelLicenseConsentRequired,
    ModelSetupError,
    ModelStore,
)


PIN = "a" * 40
OTHER_PIN = "b" * 40


class ModelStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ModelStore(self.temp.name)
        self.model = Mock()
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        self.model_loader = Mock(return_value=self.model)
        self.processor = Mock()
        self.processor_loader = Mock(return_value=self.processor)
        self.api = Mock()
        self.api.return_value.model_info.return_value = SimpleNamespace(sha=PIN)
        self.cached = Mock(return_value=None)
        fake_model = SimpleNamespace(from_pretrained=self.model_loader)
        fake_processor = SimpleNamespace(from_pretrained=self.processor_loader)
        self.patches = patch.dict(
            sys.modules,
            {
                "transformers": SimpleNamespace(
                    AutoModelForDepthEstimation=fake_model,
                    AutoImageProcessor=fake_processor,
                    Sam2Model=fake_model,
                    Sam2Processor=fake_processor,
                ),
                "huggingface_hub": SimpleNamespace(
                    HfApi=self.api, try_to_load_from_cache=self.cached
                ),
            },
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def test_offline_miss_never_calls_hub(self):
        with self.assertRaisesRegex(ModelSetupError, "No pinned cached revision"):
            self.store.get_depth("depth_anything_v2", "cpu")
        self.api.assert_not_called()
        self.model_loader.assert_not_called()

    def test_explicit_download_records_same_sha_for_model_and_processor(self):
        self.store.consent_callback = lambda request: True
        first = self.store.get_depth("DepthAnythingV2", "cpu", allow_download=True)
        second = self.store.get_depth("depth_anything_v2", "cpu")
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        self.assertEqual(self.model_loader.call_count, 1)
        for loader in (self.model_loader, self.processor_loader):
            self.assertEqual(loader.call_args.kwargs["revision"], PIN)
            self.assertFalse(loader.call_args.kwargs["local_files_only"])
            self.assertFalse(loader.call_args.kwargs["trust_remote_code"])
        manifest = json.loads(self.store.manifest_path.read_text())
        self.assertEqual(
            manifest["hf"][HF_MODELS["depth_anything_v2"][0]]["revision"], PIN
        )
        self.assertEqual(first[2]["revision"], PIN)
        first[2]["preprocessing"]["bad"] = True
        self.assertEqual(second[2]["preprocessing"], {})

    def test_recorded_revision_loads_offline_after_restart(self):
        self.store.pin_revision("depth_pro", PIN)
        restarted = ModelStore(self.temp.name)
        restarted.get_depth("depth_pro", "cpu")
        self.assertTrue(self.model_loader.call_args.kwargs["local_files_only"])
        self.assertEqual(self.model_loader.call_args.kwargs["revision"], PIN)
        self.api.assert_not_called()
        self.assertTrue(all(call.kwargs["revision"] == PIN for call in self.cached.call_args_list))

    def test_existing_hf_snapshot_supplies_pin_without_network(self):
        self.cached.return_value = str(
            Path(self.temp.name) / "models--test" / "snapshots" / PIN / "config.json"
        )
        self.assertEqual(self.store.get_depth("depth_pro", "cpu")[2]["revision"], PIN)
        self.api.assert_not_called()

    def test_revision_device_preprocessing_and_clear_invalidate_pairs(self):
        self.store.pin_revision("depth_pro", PIN)
        self.store.get_depth("depth_pro", "cpu")
        self.store.get_depth("depth_pro", "cuda:0")
        self.store.get_depth("depth_pro", "cpu", preprocessing={"do_resize": False})
        self.store.pin_revision("depth_pro", OTHER_PIN)
        self.store.get_depth("depth_pro", "cpu")
        self.assertEqual(self.model_loader.call_count, 4)
        self.assertEqual(len(self.store.identities()), 4)
        self.store.clear()
        self.assertEqual(self.store.identities(), [])
        self.store.get_depth("depth_pro", "cpu")
        self.assertEqual(self.model_loader.call_count, 5)

    def test_preprocessing_cannot_override_offline_or_identity(self):
        self.store.pin_revision("depth_pro", PIN)
        for field in ("revision", "local_files_only", "cache_dir", "trust_remote_code"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.store.get_depth("depth_pro", "cpu", preprocessing={field: "bad"})
        self.model_loader.assert_not_called()

    def test_cancellation_between_processor_and_weights_prevents_model_load(self):
        self.store.pin_revision("depth_pro", PIN)
        cancelled = [False]

        def processor(*args, **kwargs):
            cancelled[0] = True
            return self.processor

        self.processor_loader.side_effect = processor
        with self.assertRaises(ModelPreparationCancelled):
            self.store.get_depth("depth_pro", "cpu", cancelled=lambda: cancelled[0])
        self.model_loader.assert_not_called()
        self.assertEqual(self.store.identities(), [])

    def test_cancellation_before_resolution_is_side_effect_free(self):
        with self.assertRaises(ModelPreparationCancelled):
            self.store.get_sam2("cpu", allow_download=True, cancelled=lambda: True)
        self.cached.assert_not_called()
        self.api.assert_not_called()
        self.assertFalse(self.store.manifest_path.exists())

    def test_sam2_is_optional_and_uses_its_own_identity(self):
        self.assertEqual(self.store.identities(), [])
        self.store.get_sam2("cpu", allow_download=True)
        self.assertEqual(self.model_loader.call_args.args[0], HF_MODELS["sam2"][0])
        self.assertEqual(self.store.identities()[0]["model_type"], "sam2")

    def test_unpinned_midas_fails_with_setup_guidance(self):
        with self.assertRaisesRegex(ModelSetupError, "Register local MiDaS"):
            self.store.get_depth("midas", "cpu", allow_download=True)

    def test_midas_registration_and_tamper_detection(self):
        checkpoint = Path(self.temp.name) / "checkpoint.pt"
        checkpoint.write_bytes(b"original checkpoint")
        with patch.object(self.store, "_source_revision", return_value=PIN):
            self.store.register_midas("dpt", self.temp.name, checkpoint)
            checkpoint.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ModelSetupError, "checksum changed"):
                self.store.get_depth("dpt", "cpu")

    def test_midas_local_backbone_avoids_nested_hub_and_uses_correct_transform(self):
        source = Path(self.temp.name)
        original = Mock(
            side_effect=AssertionError("Unpinned backbone must not be called")
        )
        blocks = SimpleNamespace(
            __file__=str(source / "midas" / "blocks.py"),
            _edgemesh_revision=PIN,
            _make_pretrained_resnext101_wsl=original,
            _make_resnet_backbone=Mock(return_value="backbone"),
        )
        transforms = SimpleNamespace(
            default_transform="midas-transform", dpt_transform="dpt-transform"
        )

        def hub_load(directory, name, **kwargs):
            self.assertEqual(kwargs["source"], "local")
            if name == "transforms":
                return transforms
            self.assertFalse(kwargs["pretrained"])
            self.assertEqual(blocks._make_pretrained_resnext101_wsl(False), "backbone")
            return self.model

        builder = Mock(return_value="resnext")
        with patch.dict(
            sys.modules,
            {
                "torch": SimpleNamespace(hub=SimpleNamespace(load=hub_load)),
                "torchvision": SimpleNamespace(
                    __version__="test", models=SimpleNamespace(resnext101_32x8d=builder)
                ),
                "midas.blocks": blocks,
            },
        ):
            model, processor, adapter = self.store._load_legacy_architecture(
                {"source_dir": str(source), "revision": PIN}, "midas"
            )
        self.assertIs(model, self.model)
        self.assertEqual(processor, "midas-transform")
        self.assertIn("resnext101_32x8d", adapter)
        builder.assert_called_once_with(weights=None)
        self.assertIs(blocks._make_pretrained_resnext101_wsl, original)

    def test_cancelled_midas_registration_does_not_write(self):
        checkpoint = Path(self.temp.name) / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        with patch.object(self.store, "_source_revision", return_value=PIN):
            with self.assertRaises(ModelPreparationCancelled):
                self.store.register_midas(
                    "dpt", self.temp.name, checkpoint, cancelled=lambda: True
                )
        self.assertFalse(self.store.manifest_path.exists())

    def test_invalid_manifest_is_preserved_and_reported(self):
        self.store.manifest_path.parent.mkdir(parents=True)
        self.store.manifest_path.write_text('{"schema_version": 99}')
        with self.assertRaises(ModelSetupError):
            ModelStore(self.temp.name)
        self.assertEqual(self.store.manifest_path.read_text(), '{"schema_version": 99}')

    def test_download_permission_is_not_nc_license_consent(self):
        with self.assertRaises(ModelLicenseConsentRequired):
            self.store.get_depth("depth_anything_v2", "cpu", allow_download=True)
        self.processor_loader.assert_not_called()
        self.model_loader.assert_not_called()
        self.assertFalse(self.store.has_license_consent("depth_anything_v2", PIN))

    def test_consent_callback_requires_literal_true(self):
        for response in (False, None, "yes", 1):
            self.store.consent_callback = lambda request, value=response: value
            with self.subTest(response=response), self.assertRaises(ModelLicenseConsentRequired):
                self.store.get_depth("depth_anything_v2", "cpu", allow_download=True)
        self.model_loader.assert_not_called()

    def test_license_consent_is_revision_specific_and_survives_restart(self):
        self.store.pin_revision("depth_anything_v2", PIN)
        request = self.store.consent_request("depth_anything_v2")
        self.assertTrue(request["consent_needed"])
        self.assertEqual(request["revision"], PIN)
        self.store.accept_license("depth_anything_v2", request["revision"])
        restarted = ModelStore(self.temp.name)
        self.assertTrue(restarted.has_license_consent("depth_anything_v2", PIN))
        self.assertFalse(restarted.has_license_consent("depth_anything_v2", OTHER_PIN))
        restarted.pin_revision("depth_anything_v2", OTHER_PIN)
        with self.assertRaises(ModelLicenseConsentRequired):
            restarted.get_depth("depth_anything_v2", "cpu", allow_download=True)

    def test_first_run_acknowledgment_is_bound_when_revision_resolves(self):
        self.assertIsNone(self.store.consent_request("depth_anything_v2")["revision"])
        self.api.assert_not_called()
        self.store.accept_license("depth_anything_v2")
        self.assertFalse(self.store.manifest_path.exists())
        metadata = self.store.get_depth("depth_anything_v2", "cpu", allow_download=True)[2]
        self.assertTrue(self.store.has_license_consent("depth_anything_v2", PIN))
        self.assertEqual(metadata["license_policy"]["usage_class"], "noncommercial")
        self.assertEqual(self.store._pending_license_acceptance, {})
        self.store.pin_revision("depth_anything_v2", OTHER_PIN)
        with self.assertRaises(ModelLicenseConsentRequired):
            self.store.get_depth("depth_anything_v2", "cpu", allow_download=True)

    def test_offline_nc_load_needs_no_new_download_acknowledgment(self):
        self.store.pin_revision("depth_anything_v2", PIN)
        self.store.consent_callback = Mock(side_effect=AssertionError("Offline mode must not ask to download"))
        metadata = self.store.get_depth("depth_anything_v2", "cpu")[2]
        self.assertEqual(metadata["license_policy"]["usage_class"], "noncommercial")
        self.api.assert_not_called()

    def test_depth_pro_notice_is_research_only(self):
        request = self.store.consent_request("depth_pro")
        self.assertEqual(request["restriction"], "research-only")
        self.assertTrue(request["consent_needed"])

    def test_local_weight_terms_do_not_require_an_unsupported_download_acknowledgment(self):
        for model_type in ("MiDaS", "DPT"):
            with self.subTest(model_type=model_type):
                request = self.store.consent_request(model_type)
                self.assertEqual(request["usage_class"], "unknown")
                self.assertFalse(request["consent_needed"])
        self.api.assert_not_called()

    def test_cached_v1_comparison_uses_its_own_permissive_identity(self):
        self.store.pin_revision("depth_anything_v1", PIN)
        metadata = self.store.get_depth("DepthAnythingV1", "cpu")[2]
        self.assertEqual(metadata["model_id"], "LiheYoung/depth-anything-large-hf")
        self.assertEqual(metadata["license"], "apache-2.0")
        self.assertEqual(metadata["license_policy"]["usage_class"], "permissive")
        self.assertTrue(self.model_loader.call_args.kwargs["local_files_only"])
        self.assertTrue(self.processor_loader.call_args.kwargs["local_files_only"])
        self.api.assert_not_called()

    def test_policy_change_requires_a_new_acknowledgment(self):
        from dataclasses import replace
        from model_licensing import policy_for

        self.store.pin_revision("depth_anything_v2", PIN)
        self.store.accept_license("depth_anything_v2", PIN)
        changed = replace(policy_for("depth_anything_v2"), policy_version=2)
        with patch("model_store.policy_for", return_value=changed):
            self.assertFalse(self.store.has_license_consent("depth_anything_v2", PIN))
            with self.assertRaises(ModelLicenseConsentRequired):
                self.store.get_depth("depth_anything_v2", "cpu", allow_download=True)
        self.model_loader.assert_not_called()

    def test_weight_hashes_reject_anchored_or_traversing_shard_names(self):
        invalid = ("D:private.safetensors", r"\private.safetensors", "../outside.bin", r"..\outside.bin")
        valid = "shards/model-00001.safetensors"
        index = Path(self.temp.name) / "index.json"
        index.write_text(json.dumps({"weight_map": {str(i): name for i, name in enumerate((*invalid, valid))}}))
        weights = Path(self.temp.name) / "valid.safetensors"
        weights.write_bytes(b"valid cached shard")

        def cached(model, filename, **kwargs):
            if filename == "model.safetensors.index.json":
                return str(index)
            if filename in invalid:
                self.fail("An unsafe shard name reached the cache lookup")
            return str(weights) if filename == valid else None

        self.cached.side_effect = cached
        self.store.pin_revision("sam2", PIN)
        metadata = self.store.get_sam2("cpu")[2]
        self.assertEqual(set(metadata["weight_files_sha256"]), {valid})
        self.api.assert_not_called()

    def test_weight_hashes_use_exact_cached_revision(self):
        import hashlib
        weights = Path(self.temp.name) / "weights.safetensors"
        weights.write_bytes(b"cached test weights")
        self.store.pin_revision("sam2", PIN)
        self.cached.side_effect = lambda model, filename, **kwargs: str(weights) if filename == "model.safetensors" else None
        metadata = self.store.get_sam2("cpu")[2]
        self.assertEqual(metadata["weight_files_sha256"], {"model.safetensors": hashlib.sha256(weights.read_bytes()).hexdigest()})
        self.assertTrue(all(call.kwargs["revision"] == PIN for call in self.cached.call_args_list))


if __name__ == "__main__":
    unittest.main()
