"""CPU regression checks for checkpoint timing in the two official trainers.

Uses their actual train_epoch functions, a one-parameter model, disabled scaling,
and a save spy. No MiniMind model, GPU workload, or training dataset is created.
"""
import contextlib
import importlib
import inspect
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import datasets  # Preserve upstream's Windows pyarrow-before-torch import order.
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class TinyLossModel(torch.nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, device="cpu"))
        self.nonfinite = nonfinite

    def forward(self, input_ids, labels):
        loss = ((self.weight * input_ids.float() - labels.float()) ** 2).mean()
        if self.nonfinite:
            loss = loss * float("nan")
        return SimpleNamespace(loss=loss, aux_loss=loss.new_zeros(()))


class CountedSGD(torch.optim.SGD):
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr=lr)
        self.completed_steps = 0

    def step(self, closure=None):
        result = super().step(closure)
        self.completed_steps += 1
        return result


class OfficialCheckpointBoundaryTests(unittest.TestCase):
    def run_epoch(self, module_name, count=3, nonfinite=False, start_step=0):
        module = importlib.import_module(module_name)
        self.assertTrue(hasattr(module, "save_training_checkpoint"),
                        f"{module_name} must use the shared checkpoint helper")
        model = TinyLossModel(nonfinite=nonfinite)
        optimizer = CountedSGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        args = SimpleNamespace(device="cpu", epochs=1, learning_rate=0.1,
                               accumulation_steps=2, grad_clip=1000.0,
                               log_interval=1, save_interval=1,
                               save_dir="unused-test-directory", save_weight="test")
        config = SimpleNamespace(hidden_size=768, use_moe=False)
        saves = []
        signature = inspect.signature(module.save_training_checkpoint)

        def capture(*positional, **keywords):
            bound = signature.bind_partial(*positional, **keywords).arguments
            saves.append({"step": bound["step"], "optimizer_steps": optimizer.completed_steps,
                          "state": {key: value.detach().clone() for key, value in model.state_dict().items()},
                          "gradients_cleared": all(parameter.grad is None for parameter in model.parameters())})

        loader = [(torch.tensor([[float(index)]]), torch.zeros(1, 1))
                  for index in range(start_step + 1, count + 1)]
        replacements = dict(model=model, optimizer=optimizer, scaler=scaler, args=args,
                            lm_config=config, autocast_ctx=contextlib.nullcontext(),
                            get_lr=lambda current, total, lr: lr,
                            save_training_checkpoint=capture, is_main_process=lambda: True)
        with patch.multiple(module, create=True, **replacements), contextlib.redirect_stdout(io.StringIO()):
            if nonfinite:
                with self.assertRaises((FloatingPointError, RuntimeError, ValueError)):
                    module.train_epoch(0, loader, count, start_step=start_step)
            else:
                module.train_epoch(0, loader, count, start_step=start_step)
        return model, optimizer, saves

    def test_final_partial_update_precedes_save(self):
        for module_name in ("trainer.train_pretrain", "trainer.train_full_sft"):
            with self.subTest(trainer=module_name):
                model, optimizer, saves = self.run_epoch(module_name)
                self.assertEqual(optimizer.completed_steps, 2)
                self.assertEqual([entry["step"] for entry in saves], [2, 3])
                self.assertEqual([entry["optimizer_steps"] for entry in saves], [1, 2])
                self.assertTrue(all(entry["gradients_cleared"] for entry in saves))
                self.assertAlmostEqual(model.weight.item(), 0.05, places=6,
                                       msg="Keep the upstream partial-accumulation scaling unchanged")
                torch.testing.assert_close(saves[-1]["state"]["weight"], model.state_dict()["weight"],
                                           rtol=0, atol=0)

    def test_periodic_checkpoints_only_at_optimizer_boundaries(self):
        for module_name in ("trainer.train_pretrain", "trainer.train_full_sft"):
            with self.subTest(trainer=module_name):
                model, optimizer, saves = self.run_epoch(module_name, count=4)
                self.assertEqual(optimizer.completed_steps, 2)
                self.assertEqual([entry["step"] for entry in saves], [2, 4])
                self.assertTrue(all(entry["gradients_cleared"] for entry in saves))
                torch.testing.assert_close(saves[-1]["state"]["weight"], model.state_dict()["weight"],
                                           rtol=0, atol=0)

    def test_nonfinite_stops_before_backward_update_or_save(self):
        for module_name in ("trainer.train_pretrain", "trainer.train_full_sft"):
            with self.subTest(trainer=module_name):
                model, optimizer, saves = self.run_epoch(module_name, nonfinite=True)
                self.assertEqual(optimizer.completed_steps, 0)
                self.assertEqual(saves, [])
                self.assertIsNone(model.weight.grad)
                self.assertEqual(model.weight.item(), 1.0)

    def test_finished_epoch_resume_does_not_duplicate_step_or_save(self):
        for module_name in ("trainer.train_pretrain", "trainer.train_full_sft"):
            with self.subTest(trainer=module_name):
                _, optimizer, saves = self.run_epoch(module_name, start_step=3)
                self.assertEqual(optimizer.completed_steps, 0)
                self.assertEqual(saves, [])


if __name__ == "__main__":
    unittest.main()
