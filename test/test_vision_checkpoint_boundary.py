"""CPU checks execute the real V trainer functions without importing its model.

AST extraction avoids collisions between the sibling text/vision packages.
"""
import ast
import contextlib
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import datasets  # Windows DLL import order.
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TinyVLM(torch.nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.vision_encoder = torch.nn.Linear(1, 1)
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False
        self.nonfinite = nonfinite

    def forward(self, input_ids, labels, pixel_values):
        assert pixel_values["pixel_values"].device.type == "cpu"
        loss = ((self.weight * input_ids - labels) ** 2).mean()
        if self.nonfinite:
            loss = loss * float("nan")
        return SimpleNamespace(loss=loss, aux_loss=loss.new_zeros(()))


class VisionCheckpointTests(unittest.TestCase):
    def run_epoch(self, count=3, accumulation=2, nonfinite=False, start_step=0):
        model = TinyVLM(nonfinite)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        args = SimpleNamespace(device="cpu", epochs=1, learning_rate=0.1,
                               accumulation_steps=accumulation, grad_clip=1000.,
                               log_interval=100, save_interval=1)
        saves = []

        def record(model, optimizer, scaler, config, args, epoch, step, wandb=None):
            saves.append({"step": step, "weight": model.weight.detach().clone(),
                          "cleared": all(parameter.grad is None for parameter in model.parameters())})

        namespace = {"torch": torch, "time": time, "args": args, "model": model,
                     "optimizer": optimizer, "scaler": scaler, "vlm_config": SimpleNamespace(),
                     "autocast_ctx": contextlib.nullcontext(), "Logger": lambda message: None,
                     "get_lr": lambda step, total, lr: lr, "is_main_process": lambda: True,
                     "save_vlm_training_checkpoint": record}
        function = load_function(ROOT / "vision/trainer/train_sft_vlm.py", "train_epoch", namespace)
        loader = [(torch.tensor([[float(index)]]), torch.zeros(1, 1), {"pixel_values": torch.zeros(1, 1)})
                  for index in range(start_step + 1, count + 1)]
        if nonfinite:
            with self.assertRaises(FloatingPointError):
                function(0, loader, count, start_step=start_step)
        else:
            function(0, loader, count, start_step=start_step)
        return model, saves

    def test_residual_update_is_exported(self):
        model, saves = self.run_epoch()
        self.assertEqual([item["step"] for item in saves], [2, 3])
        self.assertTrue(all(item["cleared"] for item in saves))
        self.assertAlmostEqual(model.weight.item(), 0.05, places=6)
        torch.testing.assert_close(saves[-1]["weight"], model.weight, rtol=0, atol=0)

    def test_b1_acc4_tail_and_default_acc1(self):
        for count, accumulation, expected in ((5, 4, [4, 5]), (3, 1, [1, 2, 3]), (4, 2, [2, 4])):
            with self.subTest(accumulation=accumulation):
                model, saves = self.run_epoch(count, accumulation)
                self.assertEqual([item["step"] for item in saves], expected)
                self.assertTrue(all(item["cleared"] for item in saves))
                torch.testing.assert_close(saves[-1]["weight"], model.weight, rtol=0, atol=0)

    def test_nonfinite_does_not_update_or_save(self):
        model, saves = self.run_epoch(nonfinite=True)
        self.assertEqual(saves, [])
        self.assertEqual(model.weight.item(), 1.0)
        self.assertIsNone(model.weight.grad)

    def test_completed_epoch_has_no_duplicate_save(self):
        _, saves = self.run_epoch(start_step=3)
        self.assertEqual(saves, [])

    def test_atomic_half_export_excludes_encoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = TinyVLM()
            calls = []
            namespace = {"torch": torch, "os": os,
                         "DistributedDataParallel": torch.nn.parallel.DistributedDataParallel,
                         "is_main_process": lambda: True, "Logger": lambda message: None,
                         "vlm_checkpoint": lambda config, **kwargs: calls.append(kwargs)}
            save = load_function(ROOT / "vision/trainer/trainer_utils.py", "save_vlm_training_checkpoint", namespace)
            config = SimpleNamespace(hidden_size=768, use_moe=False)
            args = SimpleNamespace(save_dir=temporary, save_weight="sft_vlm")
            save(model, None, None, config, args, 0, 3)
            destination = Path(temporary) / "sft_vlm_768.pth"
            result = torch.load(destination, map_location="cpu", weights_only=True)
            self.assertEqual(set(result), {"weight"})
            self.assertEqual(result["weight"].dtype, torch.float16)
            self.assertFalse(Path(str(destination) + ".tmp").exists())
            self.assertEqual(calls[0]["step"], 3)


if __name__ == "__main__":
    unittest.main()
