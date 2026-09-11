"""CPU-only checks for public inference checkpoint loading."""
import datasets  # noqa: F401 -- Windows DLL ordering
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.evomind_load_text_model import extract_model_state


class CheckpointLoaderTests(unittest.TestCase):
    def test_pure_and_training_wrapped_state_load_identically(self):
        model = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            pure = Path(directory) / "model.pth"
            wrapped = Path(directory) / "resume.pth"
            torch.save(model.state_dict(), pure)
            torch.save({"model": model.state_dict(), "optimizer": {}, "epoch": 1, "step": 10}, wrapped)
            for path in (pure, wrapped):
                loaded = torch.nn.Linear(3, 2)
                loaded.load_state_dict(extract_model_state(path), strict=True)
                for expected, actual in zip(model.parameters(), loaded.parameters()):
                    torch.testing.assert_close(expected, actual)

    def test_metadata_only_or_malformed_wrapper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pth"
            for value in ({}, {"step": 10}, {"model": {}}, {"model": {"weight": "not a tensor"}}):
                with self.subTest(value=value):
                    torch.save(value, path)
                    with self.assertRaises(ValueError):
                        extract_model_state(path)

    def test_wrong_architecture_remains_a_strict_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            torch.save({"model": torch.nn.Linear(3, 2).state_dict()}, path)
            with self.assertRaises(RuntimeError):
                torch.nn.Linear(4, 2).load_state_dict(extract_model_state(path), strict=True)


if __name__ == "__main__":
    unittest.main()
