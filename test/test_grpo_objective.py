import unittest

import torch

from model.grpo import completion_logps, group_advantages, grpo_policy_loss


class TestGRPOObjective(unittest.TestCase):
    def test_group_advantages_are_zero_mean_per_prompt(self):
        advantages = group_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 3)
        self.assertTrue(torch.allclose(advantages.reshape(2, 3).mean(dim=1), torch.zeros(2), atol=1e-6))
        self.assertGreater(advantages[0].item(), 0)
        self.assertGreater(advantages[4].item(), 0)

    def test_completion_logps_uses_causal_shift(self):
        logits = torch.tensor([[[0.0, 6.0, 0.0], [7.0, 0.0, 0.0], [0.0, 0.0, 5.0]]])
        ids = torch.tensor([[2, 1, 0]])  # prompt=[2], completion=[1, 0]
        logps = completion_logps(logits, ids, prompt_length=1, completion_length=2)
        self.assertEqual(tuple(logps.shape), (1, 2))
        self.assertGreater(logps[0, 0].item(), -0.1)
        self.assertGreater(logps[0, 1].item(), -0.1)

    def test_old_policy_equal_to_current_has_ratio_one_and_finite_loss(self):
        logps = torch.tensor([[-1.0, -1.2], [-0.8, -1.5]], requires_grad=True)
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.ones_like(logps, dtype=torch.bool)
        loss, values = grpo_policy_loss(logps, logps.detach(), logps.detach(), advantages, mask, epsilon=0.2, beta=0.04)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(values["ratio"].item(), 1.0, places=6)
        self.assertAlmostEqual(values["kl"].item(), 0.0, places=6)
        loss.backward()
        self.assertIsNotNone(logps.grad)


if __name__ == "__main__":
    unittest.main()
