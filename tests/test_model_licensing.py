"""Policy authority and provenance integrity without model downloads or GPUs."""

import copy
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from model_licensing import (
    CATALOG,
    ProvenanceStore,
    canonical_model_identity,
    geometry_sha256,
    policy_for,
)


class LicensingTests(unittest.TestCase):
    def test_catalog_identity_wins_over_editable_license(self):
        identity = canonical_model_identity({
            "model_id": "depth-anything/Depth-Anything-V2-Large-hf",
            "model_type": "sam2", "license": "apache-2.0",
            "license_policy": {"usage_class": "permissive"}, "revision": "a" * 40,
        })
        self.assertEqual(identity["license"], "cc-by-nc-4.0")
        self.assertEqual(identity["model_type"], "depth_anything_v2")
        self.assertEqual(identity["license_policy"]["usage_class"], "noncommercial")

    def test_unknown_checkpoint_cannot_borrow_known_model_type_license(self):
        for model_id in ("custom/weights", "", None, 0, "sam2", "SAM2", "DepthAnythingV1", "Depth Pro"):
            with self.subTest(model_id=model_id):
                identity = canonical_model_identity({"model_id": model_id, "model_type": "sam2", "license": "apache-2.0"})
                self.assertEqual(identity["license_policy"]["usage_class"], "unknown")
                self.assertEqual(identity["license_policy"]["commercial_status"], "unverified")

    def test_model_families_keep_their_checkpoint_specific_terms(self):
        self.assertEqual(policy_for("LiheYoung/depth-anything-large-hf").license, "apache-2.0")
        self.assertEqual(policy_for("DepthAnythingV1").usage_class, "permissive")
        self.assertEqual(policy_for("depth-anything/DA3-LARGE-1.1").usage_class, "noncommercial")
        self.assertEqual(policy_for("depth-anything/DA3METRIC-LARGE").usage_class, "permissive")
        self.assertEqual(policy_for("depth-anything/DA3MONO-LARGE").usage_class, "permissive")
        self.assertEqual(policy_for("depth-anything/DA3-BASE").usage_class, "permissive")
        self.assertEqual(policy_for("Depth Pro").restriction, "research-only")
        self.assertTrue(policy_for("Depth Pro").requires_acceptance)
        self.assertFalse(policy_for("sam2").requires_acceptance)
        with self.assertRaises(TypeError):
            CATALOG["sam2"] = policy_for("depth_pro")


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        logger_patch = patch("log_utils.get_logger", return_value=logging.getLogger("model_licensing"))
        logger_patch.start()
        self.addCleanup(logger_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ProvenanceStore(self.temp.name)
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.5]], dtype=np.float64)
        self.faces = np.array([[0, 1, 2]], dtype=np.int64)
        self.models = [{"model_type": "depth_anything_v2", "revision": "a" * 40}, {"model_type": "sam2", "revision": "b" * 40}]

    def seal(self):
        return self.store.seal(self.vertices, self.faces, self.models, {"depth_amount": 1.0})

    def test_round_trip_is_plain_json_and_binds_both_model_stages(self):
        envelope = json.loads(json.dumps(self.seal()))
        self.assertEqual(self.store.verify(envelope, self.vertices, self.faces)["status"], "verified")
        self.assertEqual(len(envelope["payload"]["models"]), 2)
        self.assertEqual(envelope["payload"]["models"][0]["license"], "cc-by-nc-4.0")
        self.assertEqual(ProvenanceStore(self.temp.name).verify(envelope)["status"], "verified")

    def test_editing_license_or_geometry_breaks_binding(self):
        original = self.seal()
        edited = copy.deepcopy(original)
        edited["payload"]["models"][0]["license"] = "apache-2.0"
        with self.assertLogs("model_licensing", "WARNING"):
            self.assertEqual(self.store.verify(edited)["status"], "tampered")
        moved = self.vertices.copy()
        moved[2, 2] += 0.5
        with self.assertLogs("model_licensing", "WARNING"):
            self.assertEqual(self.store.verify(original, moved, self.faces)["status"], "tampered")

    def test_missing_or_different_profile_key_is_unverified(self):
        envelope = self.seal()
        other = ProvenanceStore(Path(self.temp.name) / "other-profile")
        self.assertEqual(other.verify(envelope)["status"], "unverified")
        self.assertFalse(other.key_path.exists(), "Verification must not create keys")
        other.seal(self.vertices, self.faces, self.models)
        self.assertEqual(other.verify(envelope)["status"], "unverified")

    def test_key_storage_failure_retains_unsigned_unverified_metadata(self):
        with patch.object(self.store, "_key", side_effect=OSError("test unavailable storage")), self.assertLogs("model_licensing", "ERROR"):
            envelope = self.seal()
        self.assertEqual(envelope["integrity"], "unverified")
        self.assertEqual(envelope["signature"], "")
        self.assertEqual(self.store.verify(envelope)["status"], "unverified")

    def test_hash_is_stable_across_array_endian_and_memory_layout(self):
        digest = geometry_sha256(self.vertices, self.faces)
        self.assertEqual(digest, geometry_sha256(np.asfortranarray(self.vertices), self.faces.astype(">i8")))
        self.assertEqual(digest, geometry_sha256(self.vertices.astype(">f8"), self.faces.astype(np.uint32)))

    def test_malformed_geometry_and_metadata_return_unverified(self):
        envelope = self.seal()
        for vertices in ([[10 ** 400, 0, 0]], [[float("nan"), 0, 0]], [1, 2, 3]):
            with self.subTest(vertices=str(vertices)[:40]), self.assertLogs("model_licensing", "ERROR"):
                self.assertEqual(self.store.verify(envelope, vertices, self.faces)["status"], "unverified")
        malformed = copy.deepcopy(envelope)
        malformed["payload"] = []
        with self.assertLogs("model_licensing", "ERROR"):
            self.assertEqual(self.store.verify(malformed)["status"], "unverified")

    def test_concurrent_first_seals_share_one_complete_key(self):
        code = (
            "import json,sys; from model_licensing import ProvenanceStore; "
            "s=ProvenanceStore(sys.argv[1]); "
            "e=s.seal([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]],[]); "
            "print(json.dumps(e))"
        )
        processes = [subprocess.Popen([sys.executable, "-c", code, self.temp.name],
                                     cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True) for _ in range(3)]
        try:
            envelopes = []
            for process in processes:
                output, error = process.communicate(timeout=60)
                self.assertEqual(process.returncode, 0, error)
                envelopes.append(json.loads(output))
            self.assertEqual(len({envelope["key_id"] for envelope in envelopes}), 1)
            for envelope in envelopes:
                self.assertEqual(self.store.verify(envelope)["status"], "verified")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    unittest.main()
