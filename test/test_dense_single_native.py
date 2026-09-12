"""Exercise launcher storage hooks through the actual native trainer on CPU."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dense_single_native_launcher_under_test", ROOT / "scripts" / "launch_dense_single.py"
)
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


class NativeWorkerIntegrationTests(unittest.TestCase):
    def test_two_native_epoch_tail_snapshots_and_completed_resume(self):
        # Native modules are imported only for this integration test. Keep the
        # Windows Arrow-before-Torch import order and never allocate on CUDA.
        import datasets
        import torch

        original_path = sys.path[:]
        sys.path.insert(0, str(ROOT))
        self.addCleanup(setattr, sys, "path", original_path)
        import dataset.lm_dataset as native_data
        from model.model_vlm import VLMConfig
        import trainer.trainer_utils as native

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.25))
                self.vision_encoder = torch.nn.Linear(1, 1)
                self.vision_encoder.requires_grad_(False)
                self.forward_calls = 0

            def forward(self, input_ids, labels, pixel_values):
                self.forward_calls += 1
                return SimpleNamespace(
                    loss=((self.weight * input_ids - labels) ** 2).mean(),
                    aux_loss=self.weight.new_zeros(()),
                )

        class ThreeRows(torch.utils.data.Dataset):
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 3

            def __getitem__(self, index):
                value = torch.tensor([float(index + 1)])
                return value, value * 1.5, {"pixel_values": torch.zeros(1)}

        models = []

        def tiny_runtime(contract):
            model = TinyModel()
            models.append(model)
            return model, None, None, VLMConfig(hidden_size=768, num_hidden_layers=8)

        real_argv = launcher.native_argv

        def cpu_argv(run_dir, selection, resume=False):
            argv = real_argv(run_dir, selection, resume=resume)
            argv[argv.index("--device") + 1] = "cpu"
            return argv

        original_initialize = native.init_vlm_model
        original_checkpoint = native.vlm_checkpoint

        def invoke(run_dir, contract, resume):
            original_argv = sys.argv[:]
            try:
                return launcher.train_worker(SimpleNamespace(run_dir=run_dir, resume=resume), contract)
            finally:
                # Production invokes each worker in a fresh subprocess. Restore
                # those module-level hooks between invocations in this process.
                native.init_vlm_model = original_initialize
                native.vlm_checkpoint = original_checkpoint
                sys.argv = original_argv

        with tempfile.TemporaryDirectory(prefix="dense-native-", dir=ROOT / "test") as temporary:
            run_dir = Path(temporary)
            contract = {"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001",
                        "data": {"path": str(run_dir / "fixture.parquet")}}
            (run_dir / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
            (run_dir / "preflight.json").write_text(
                json.dumps({"selected": {"batch_size": 1, "accumulation_steps": 4}}),
                encoding="utf-8",
            )
            console = io.StringIO()
            with (
                patch.object(launcher, "gpu_guard", return_value={}),
                patch.object(launcher, "load_native_runtime", side_effect=tiny_runtime),
                patch.object(launcher, "OFFICIAL_ROWS", 3),
                patch.object(launcher, "native_argv", side_effect=cpu_argv),
                patch.object(launcher.runpy, "run_path", wraps=launcher.runpy.run_path) as run_path,
                patch.object(native_data, "VLMDataset", ThreeRows),
                patch.object(torch.cuda, "is_available", return_value=False),
                patch.object(torch.cuda, "manual_seed"),
                patch.object(torch.cuda, "manual_seed_all"),
                patch.object(torch.cuda, "empty_cache"),
                patch.dict(os.environ, {"RANK": "-1"}),
                redirect_stdout(console),
            ):
                self.assertEqual(invoke(run_dir, contract, resume=False), 0)
                self.assertEqual(models[0].forward_calls, 6)
                snapshot_dirs = sorted(path.name for path in (run_dir / "snapshots").iterdir())
                self.assertEqual(snapshot_dirs, ["epoch1_step3", "epoch2_step3"])
                history_before = (run_dir / "checkpoints.jsonl").read_bytes()
                snapshot_hashes = {
                    str(path.relative_to(run_dir)): launcher.sha256(path)
                    for path in (run_dir / "snapshots").rglob("*") if path.is_file()
                }
                first = torch.load(run_dir / "snapshots/epoch1_step3/resume.pth", weights_only=True)
                second = torch.load(run_dir / "snapshots/epoch2_step3/resume.pth", weights_only=True)
                self.assertEqual((first["epoch"], first["step"]), (0, 3))
                self.assertEqual((second["epoch"], second["step"]), (1, 3))
                self.assertEqual(next(iter(first["optimizer"]["state"].values()))["step"].item(), 1)
                self.assertEqual(next(iter(second["optimizer"]["state"].values()))["step"].item(), 2)
                self.assertNotEqual(first["model"]["weight"].item(), second["model"]["weight"].item())
                export = torch.load(run_dir / "exports/dense_single_768.pth", weights_only=True)
                self.assertEqual(set(export), {"weight"})
                self.assertEqual(export["weight"].dtype, torch.float16)
                completion = json.loads((run_dir / "training_completed.json").read_text(encoding="utf-8"))
                self.assertEqual(completion["status"], "training_complete_evaluation_pending")
                self.assertEqual(completion["epochs"], 2)
                self.assertEqual(completion["optimizer_updates"], 2)
                self.assertEqual(completion["weights"]["sha256"], launcher.sha256(completion["weights"]["path"]))

                self.assertEqual(invoke(run_dir, contract, resume=True), 0)
                self.assertEqual(models[1].forward_calls, 0)
                torch.testing.assert_close(models[1].state_dict()["weight"], second["model"]["weight"], rtol=0, atol=0)
                self.assertEqual((run_dir / "checkpoints.jsonl").read_bytes(), history_before)
                self.assertEqual(
                    {str(path.relative_to(run_dir)): launcher.sha256(path)
                     for path in (run_dir / "snapshots").rglob("*") if path.is_file()},
                    snapshot_hashes,
                )
                self.assertEqual(run_path.call_count, 2)
                for call in run_path.call_args_list:
                    self.assertEqual(Path(call.args[0]), ROOT / "trainer/train_sft_vlm.py")
                    self.assertEqual(call.kwargs["run_name"], "__main__")
            self.assertIn("Checkpoint saved after optimizer update: epoch=2, microstep=3", console.getvalue())
            self.assertIs(native.init_vlm_model, original_initialize)
            self.assertIs(native.vlm_checkpoint, original_checkpoint)


if __name__ == "__main__":
    unittest.main()
