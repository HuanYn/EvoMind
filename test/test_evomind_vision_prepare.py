"""Lightweight tests for the CPU/GPU scope boundary and data readiness contract."""
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evomind_vision_prepare as prep
import evomind_continue as continuation


class PreparationTests(unittest.TestCase):
    def test_cpu_permission_does_not_open_gpu_gate(self):
        prep.require_cpu_scope()
        with self.assertRaisesRegex(RuntimeError, "VISION_HELD"):
            continuation.require_vision_enabled()

    def test_child_environment_masks_gpu_and_bounds_threads(self):
        env = prep.cpu_environment()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")

    def test_memory_probe_is_read_only_and_positive(self):
        self.assertGreater(prep.available_memory(), 0)
        self.assertNotIn("torch", sys.modules)

    def test_partial_or_capped_data_is_not_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.json"
            for value in ({"status": "running"}, {"status": "complete", "seed": 42, "val_fraction": .02, "test_fraction": .02, "max_rows": 100}):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    prep.validate_summary(path)

    def test_complete_summary_requires_full_scan_and_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "summary.json"
            train = root / "train.parquet"
            (root / "manifest.jsonl").touch()
            train.touch()
            value = {"format_version": 1, "status": "complete", "seed": 42, "val_fraction": .02, "test_fraction": .02,
                     "max_rows": None, "grouping": "sha256_original_image_bytes", "counts": {"scanned": 10},
                     "sources": [{"total_rows": 10}], "official_train_parquet": str(train)}
            path.write_text(json.dumps(value))
            prep.validate_summary(path)
            value["counts"]["scanned"] = 9
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "Full source"):
                prep.validate_summary(path)

    def test_direct_gpu_actions_are_gated_too(self):
        for action in ("probe", "parity", "cache", "train", "official", "evaluate"):
            with self.subTest(action=action):
                result = subprocess.run([sys.executable, "-B", "scripts/evomind_pipeline_step.py", action],
                                        cwd=prep.ROOT, env=prep.cpu_environment(), capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("VISION_HELD_FOR_FULL_TEXT_ALIGNMENT", result.stderr)


if __name__ == "__main__":
    unittest.main()
