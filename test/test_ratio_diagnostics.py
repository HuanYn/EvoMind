import datasets  # noqa: F401 -- Windows DLL ordering before torch
import math
import unittest

import torch

from trainer.evomind_rl_runtime import grpo_cispo_ratio_statistics


class TestRatioDiagnostics(unittest.TestCase):
    def test_reports_grpo_directional_suppression_and_cispo_cap(self):
        # Both valid tokens are GRPO-suppressed for a positive advantage;
        # only ratio 6.0 is CISPO-capped.
        logps = torch.log(torch.tensor([[1.3, 0.7, 6.0]]))
        old = torch.zeros_like(logps)
        ref = torch.zeros_like(logps)
        advantages = torch.tensor([1.0])
        mask = torch.tensor([[1.0, 0.0, 1.0]])
        stats = grpo_cispo_ratio_statistics(logps, old, ref, advantages, mask, epsilon=0.2, epsilon_high=5.0)
        self.assertAlmostEqual(stats["grpo_suppressed_rate"], 1.0, places=6)
        self.assertAlmostEqual(stats["cispo_capped_rate"], 0.5, places=6)

    def test_signs_and_exact_thresholds_use_strict_directional_comparisons(self):
        # Boundaries 0.5/1.5 and 2.0 are not interventions. Wrong-side ratios
        # and zero advantages do not suppress GRPO's advantage term.
        ratios = torch.tensor([[.25, .5, 1.5, 2., 4.]] * 3, dtype=torch.float64)
        logps = ratios.log()
        stats = grpo_cispo_ratio_statistics(
            logps, torch.zeros_like(logps), logps, torch.tensor([1., -1., 0.]),
            torch.ones_like(logps), epsilon=.5, epsilon_high=2.)
        self.assertAlmostEqual(stats["grpo_suppressed_rate"], 3 / 15, places=7)
        # CISPO's ratio cap is counted independently of the advantage sign.
        self.assertAlmostEqual(stats["cispo_capped_rate"], 3 / 15, places=7)
        self.assertAlmostEqual(stats["ratio_mean"], 1.65, places=6)
        self.assertEqual(stats["ratio_p95"], 4.)
        self.assertEqual(stats["ratio_max"], 4.)
        self.assertEqual(stats["reference_logp_gap"], 0.)
        self.assertEqual(stats["kl_penalty"], 0.)

    def test_padding_and_empty_rows_do_not_change_token_statistics(self):
        selected = torch.tensor([[.5, 2.]], dtype=torch.float64).log()
        expected = grpo_cispo_ratio_statistics(
            selected, torch.zeros_like(selected), selected + .25, torch.tensor([-1.]),
            torch.ones_like(selected), epsilon=.2, epsilon_high=1.5)
        # Large/nonfinite padding values must be excluded before exp/reductions.
        padded = torch.tensor([[selected[0, 0], 1000., selected[0, 1]],
                               [float("nan"), float("inf"), -float("inf")]], dtype=torch.float64)
        mask = torch.tensor([[1., 0., 1.], [0., 0., 0.]])
        actual = grpo_cispo_ratio_statistics(
            padded, torch.zeros_like(padded), padded + .25, torch.tensor([-1., 1.]),
            mask, epsilon=.2, epsilon_high=1.5)
        self.assertEqual(actual, expected)
        self.assertAlmostEqual(actual["reference_logp_gap"], .25)
        self.assertAlmostEqual(actual["kl_penalty"], math.exp(.25) - .25 - 1)

    def test_empty_mask_has_an_explicit_error(self):
        logps = torch.zeros(2, 3)
        with self.assertRaisesRegex(ValueError, "at least one valid completion token"):
            grpo_cispo_ratio_statistics(logps, logps, logps, torch.ones(2), torch.zeros_like(logps),
                                       epsilon=.2, epsilon_high=5.)


if __name__ == "__main__":
    unittest.main()
