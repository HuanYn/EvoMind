"""CPU fixtures for evaluation recovery; no trained-model score is fabricated."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import datasets  # Windows DLL order
import torch
import evomind_harness_data as data
import evomind_harness_eval as harness
import evomind_posttrain_evaluate as evaluation


class RecoveryTests(unittest.TestCase):
    def test_dtype_and_numeric_serialization(self):
        import numpy as np
        value = {"dtype": torch.float16, "device": torch.device("cpu"), "score": np.float32(.5)}
        result = json.loads(json.dumps(value, default=harness.serializable, allow_nan=False))
        self.assertEqual(result, {"dtype": "torch.float16", "device": "cpu", "score": .5})

    def test_unknown_metadata_rejected(self):
        with self.assertRaises(TypeError):
            json.dumps({"unknown": object()}, default=harness.serializable)
        with self.assertRaises(ValueError):
            json.dumps({"score": float("nan")}, default=harness.serializable, allow_nan=False)

    def test_scope_restores_loader_on_error(self):
        original = datasets.load_dataset
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with data.local_chinese_datasets("cmmlu"):
                self.assertIsNot(datasets.load_dataset, original)
                raise RuntimeError("fixture")
        self.assertIs(datasets.load_dataset, original)

    def test_no_network_for_pinned_chinese(self):
        with patch.object(datasets, "load_dataset", side_effect=AssertionError("network forbidden")):
            with data.local_chinese_datasets("ceval-valid"):
                ds = datasets.load_dataset("ceval/ceval-exam", name="computer_network")
                self.assertEqual(len(ds["val"]), 19)
                self.assertEqual(len(ds["dev"]), 5)
                self.assertEqual(ds["val"][0]["answer"], data.subject_rows("ceval", "computer_network", "val")[0]["answer"])
            with data.local_chinese_datasets("cmmlu"):
                ds = datasets.load_dataset("haonan-li/cmmlu", name="agronomy")
                self.assertEqual(len(ds["dev"]), 5)
                self.assertEqual(set(ds["test"].column_names), {"Question", "A", "B", "C", "D", "Answer"})

    def test_unrelated_dataset_not_intercepted(self):
        with patch.object(datasets, "load_dataset", return_value="fixture") as original:
            with data.local_chinese_datasets("cmmlu"):
                self.assertEqual(datasets.load_dataset("unrelated", name="abc"), "fixture")
            original.assert_called_once_with("unrelated", name="abc")

    def test_unknown_options_and_bad_subject_rejected(self):
        with data.local_chinese_datasets("cmmlu"):
            with self.assertRaises(ValueError):
                datasets.load_dataset("haonan-li/cmmlu", name="agronomy", split="test")
        with self.assertRaises(ValueError):
            data.subject_rows("cmmlu", "../agronomy", "test")

    def test_recovery_uses_new_harness_directories_only(self):
        jobs = evaluation.jobs("full_sft", {"checkpoint": "fixture.pth"})
        self.assertEqual(len(jobs), 10)
        for key, argv, directory, result in jobs:
            if key.startswith("harness_"):
                self.assertTrue(directory.name.endswith("_v2"))
            else:
                self.assertEqual(directory.name, key)


if __name__ == "__main__":
    unittest.main()
