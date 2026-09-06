import unittest

import torch

from model.chat_template import IGNORE_INDEX
from model.dpo_loss import assistant_sequence_logps, dpo_loss


class TestDPOLoss(unittest.TestCase):
    def test_sequence_logps_use_only_shifted_non_ignored_labels(self):
        logits = torch.tensor([[[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]])
        labels = torch.tensor([[IGNORE_INDEX, 1, IGNORE_INDEX]])
        actual = assistant_sequence_logps(logits, labels)
        expected = torch.log_softmax(logits[0, 0], dim=-1)[1]
        self.assertTrue(torch.allclose(actual, expected.unsqueeze(0)))

    def test_dpo_prefers_larger_chosen_relative_reward(self):
        policy_chosen = torch.tensor([5.0])
        policy_rejected = torch.tensor([2.0])
        reference_chosen = torch.tensor([4.0])
        reference_rejected = torch.tensor([2.5])
        loss, metrics = dpo_loss(
            policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1
        )
        self.assertLess(loss.item(), torch.log(torch.tensor(2.0)).item())
        self.assertGreater(metrics["margin"].item(), 0.0)
        self.assertEqual(metrics["preference_accuracy"].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
