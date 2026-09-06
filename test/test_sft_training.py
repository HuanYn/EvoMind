import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from trainer.train_full_sft import (
    cosine_learning_rate,
    sha256_file,
    shifted_sft_nll,
    trim_batch_right_padding,
)
from model.chat_template import IGNORE_INDEX


class TestSFTTrainingHelpers(unittest.TestCase):
    def test_shifted_loss_uses_next_token_and_ignores_prompt(self):
        logits = torch.tensor(
            [
                [
                    [4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 0.0],
                    [0.0, 6.0, 0.0, 0.0],
                ]
            ]
        )
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 2, 1]])

        nll_sum, supervised = shifted_sft_nll(logits, labels)
        expected = F.cross_entropy(
            logits[:, 1:, :].reshape(-1, 4),
            torch.tensor([2, 1]),
            reduction="sum",
        )

        self.assertEqual(supervised, 2)
        self.assertTrue(torch.allclose(nll_sum, expected))
        self.assertLess((nll_sum / supervised).item(), 0.1)

    def test_shifted_loss_rejects_no_assistant_target(self):
        logits = torch.zeros(1, 2, 4)
        labels = torch.full((1, 3), IGNORE_INDEX)
        with self.assertRaisesRegex(ValueError, "no supervised assistant"):
            shifted_sft_nll(logits, labels)

    def test_trim_removes_only_all_padding_suffix_columns(self):
        input_ids = torch.tensor(
            [
                [1, 8, 2, 0, 0],
                [1, 9, 7, 2, 0],
            ]
        )
        labels = torch.tensor(
            [
                [IGNORE_INDEX, IGNORE_INDEX, 2, IGNORE_INDEX, IGNORE_INDEX],
                [IGNORE_INDEX, IGNORE_INDEX, 7, 2, IGNORE_INDEX],
            ]
        )
        trimmed_ids, trimmed_labels = trim_batch_right_padding(
            input_ids,
            labels,
            pad_id=0,
        )
        self.assertEqual(tuple(trimmed_ids.shape), (2, 4))
        self.assertEqual(tuple(trimmed_labels.shape), (2, 4))
        self.assertTrue(torch.equal(trimmed_ids[:, -1], torch.tensor([0, 2])))

    def test_cosine_schedule_matches_boundaries(self):
        self.assertAlmostEqual(
            cosine_learning_rate(1, 10, 2, 1e-4, 1e-5),
            5e-5,
        )
        self.assertAlmostEqual(
            cosine_learning_rate(2, 10, 2, 1e-4, 1e-5),
            1e-4,
        )
        self.assertAlmostEqual(
            cosine_learning_rate(10, 10, 2, 1e-4, 1e-5),
            1e-5,
        )

    def test_sha256_file_is_streamed_and_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"MiniMind SFT")
            self.assertEqual(
                sha256_file(path, chunk_size=3),
                "318790485d484ceb87495ff68cff37e4f4a511aec8b2460589d62af1b0fc2b0a",
            )


if __name__ == "__main__":
    unittest.main()
