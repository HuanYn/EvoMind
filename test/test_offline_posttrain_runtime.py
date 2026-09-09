"""Tiny CPU-only checks of real DPO/LoRA/KD loops and their disk checkpoints."""
import contextlib
import copy
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import datasets  # Windows import ordering; never load a real training dataset.
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer.evomind_offline_runtime import OfflineRuntime, OfflineAdamW, atomic_write, sha256


class TinyModel(torch.nn.Module):
    def __init__(self, weight=0.8, nonfinite=False):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.layer = torch.nn.Module()
        self.layer.lora = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.layer.lora.weight.fill_(weight)
        self.nonfinite = nonfinite

    def forward(self, input_ids, labels=None):
        value = input_ids.float() * self.layer.lora.weight.squeeze() + self.base
        if self.nonfinite:
            value = value * float("nan")
        logits = torch.stack((value, -value, value * 0, value * 0.5), dim=-1)
        return SimpleNamespace(logits=logits, loss=value.square().mean(), aux_loss=value.new_zeros(()))


class OfflinePosttrainTests(unittest.TestCase):
    def setUp(self):
        guard = patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA initialization forbidden in CPU tests"))
        guard.start()
        self.addCleanup(guard.stop)
        parent = ROOT / "artifacts" / "test_tmp"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="offline_posttrain_", dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old_cwd = Path.cwd()
        (self.root / "trainer").mkdir()
        (self.root / "model").mkdir()
        (self.root / "model" / "tokenizer.json").write_text("{}", encoding="utf-8")
        os.chdir(self.root / "trainer")
        self.addCleanup(os.chdir, self.old_cwd)

    def fixture(self, branch, *, accumulation=2, max_steps=0, nonfinite=False, count=3, folder=None, teacher_autocast=0):
        path = self.root / (folder or branch)
        path.mkdir()
        init_dir = path / "init"
        init_dir.mkdir()
        (init_dir / "full_sft_768.pth").write_bytes(b"hashed fixture base; not loaded as a model")
        data_path = path / "data.jsonl"
        data_path.write_text('{}\n', encoding="utf-8")
        args = SimpleNamespace(device="cpu", epochs=1, batch_size=1, learning_rate=0.03,
            accumulation_steps=accumulation, grad_clip=1.0, log_interval=1, save_interval=1,
            max_steps=max_steps, max_seq_len=4, num_workers=0, dtype="bfloat16", seed=42,
            from_resume=0, from_weight="full_sft", save_weight=branch, lora_name=branch,
            save_dir=str(path / "out"), resume_dir=str(path / "checkpoints"),
            init_dir=str(init_dir), data_path=str(data_path), beta=0.15, alpha=0.5, temperature=1.5,
            teacher_autocast=teacher_autocast)
        config = SimpleNamespace(hidden_size=768, use_moe=False)
        runtime = OfflineRuntime(args, config, branch, branch, [("base", config, "full_sft")])
        model = TinyModel(nonfinite=nonfinite)
        optimizer = OfflineAdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        runtime.restore(model, optimizer, scaler, None, count)
        runtime.start_training()
        return SimpleNamespace(args=args, config=config, runtime=runtime, model=model, optimizer=optimizer,
                               scaler=scaler, count=count, branch=branch)

    def run_epoch(self, case, *, count=None, iters=None, start_step=0):
        module = importlib.import_module(f"trainer.train_{case.branch}")
        count = case.count if count is None else count
        iters = case.count if iters is None else iters
        pair = (torch.tensor([[1, 2, 1]]), torch.tensor([[-100, 0, 1]]))
        if case.branch == "dpo":
            batch = {"x_chosen": pair[0], "x_rejected": pair[0],
                     "y_chosen": torch.tensor([[0, 0, 0]]), "y_rejected": torch.tensor([[1, 1, 1]]),
                     "mask_chosen": torch.ones(1, 3), "mask_rejected": torch.ones(1, 3)}
            loader = [batch] * count
        else:
            loader = [pair] * count
        replacements = dict(args=case.args, model=case.model, optimizer=case.optimizer, scaler=case.scaler,
                            runtime=case.runtime, lm_config=case.config, autocast_ctx=contextlib.nullcontext())
        with patch.multiple(module, create=True, **replacements), contextlib.redirect_stdout(io.StringIO()):
            if case.branch == "dpo":
                reference = TinyModel(weight=1.2).eval().requires_grad_(False)
                module.train_epoch(0, loader, iters, reference, case.config, start_step, beta=0.15)
            elif case.branch == "lora":
                module.train_epoch(0, loader, iters, [p for p in case.model.parameters() if p.requires_grad], start_step)
            else:
                teacher = TinyModel(weight=1.2).eval().requires_grad_(False)
                module.train_epoch(0, loader, iters, teacher, case.config, start_step, alpha=0.5, temperature=1.5)

    def test_real_loops_tail_update_saved_and_only_boundaries(self):
        for branch in ("dpo", "lora", "distillation"):
            with self.subTest(branch=branch):
                case = self.fixture(branch)
                with patch.object(case.runtime, "save", wraps=case.runtime.save) as save:
                    self.run_epoch(case)
                case.runtime.finish()
                self.assertEqual([call.args[4] for call in save.call_args_list], [2, 3])
                saved = torch.load(case.runtime.resume_path, weights_only=True)
                self.assertEqual(saved["step"], 3)
                self.assertEqual(saved["evomind"]["optimizer_updates"], 2)
                self.assertEqual(saved["scaler"], {})
                for key, value in case.model.state_dict().items():
                    torch.testing.assert_close(saved["model"][key], value, rtol=0, atol=0)
                for state in saved["optimizer"]["state"].values():
                    self.assertEqual(state["step"].item(), 2)
                export = torch.load(case.runtime.export_path, weights_only=True)
                self.assertTrue(all(value.dtype == torch.float16 for value in export.values()))
                if branch == "lora":
                    self.assertTrue(all(".lora." in key for key in export))
                    self.assertNotIn("base", export)
                report = json.loads(case.runtime.report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "completed")
                self.assertFalse(report["probe_only"])
                self.assertEqual(report["export_path"], str(case.runtime.export_path))
                self.assertEqual(report["resume_path"], str(case.runtime.resume_path))
                self.assertEqual(report["export_sha256"], sha256(case.runtime.export_path))
                self.assertEqual(report["resume_sha256"], sha256(case.runtime.resume_path))

    def test_probe_two_complete_adam_updates_requires_eight_microbatches(self):
        for branch in ("dpo", "lora", "distillation"):
            with self.subTest(branch=branch):
                case = self.fixture(branch, accumulation=4, max_steps=2, count=10)
                self.run_epoch(case)
                case.runtime.finish()
                report = json.loads(case.runtime.report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "probe_complete")
                self.assertEqual(report["stop_reason"], "max_steps")
                self.assertEqual(report["optimizer_updates_this_invocation"], 2)
                self.assertEqual(report["microbatches_this_invocation"], 8)
                self.assertEqual(report["step"], 8)
                self.assertTrue(report["probe_only"])

    def test_nonfinite_loss_stops_without_update_or_save(self):
        for branch in ("dpo", "lora", "distillation"):
            with self.subTest(branch=branch):
                case = self.fixture(branch, nonfinite=True)
                with self.assertRaises(FloatingPointError):
                    self.run_epoch(case)
                self.assertEqual(case.runtime.updates, 0)
                self.assertFalse(case.runtime.resume_path.exists())
                self.assertTrue(all(p.grad is None for p in case.model.parameters()))

    def test_strict_resume_restores_native_model_adam_and_tail_position(self):
        case = self.fixture("dpo")
        self.run_epoch(case)
        args = copy.deepcopy(case.args)
        args.from_resume = 1
        resumed = OfflineRuntime(args, case.config, "dpo", "dpo", [("base", case.config, "full_sft")])
        data = resumed.load_resume()
        model = TinyModel(weight=9.0)
        optimizer = OfflineAdamW((p for p in model.parameters() if p.requires_grad), lr=0.03)
        self.assertEqual(resumed.restore(model, optimizer, case.scaler, data, 3), (1, 0))
        torch.testing.assert_close(model.layer.lora.weight, case.model.layer.lora.weight, rtol=0, atol=0)
        self.assertEqual(next(iter(optimizer.state.values()))["step"].item(), 2)
        self.assertEqual(resumed.updates, 2)

    def resumed_fixture(self, case):
        args = copy.deepcopy(case.args)
        args.from_resume = 1
        runtime = OfflineRuntime(args, case.config, case.branch, case.branch, [("base", case.config, "full_sft")])
        model = TinyModel(weight=9.0)
        optimizer = OfflineAdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        return SimpleNamespace(runtime=runtime, model=model, optimizer=optimizer, scaler=scaler)

    def assert_final_export(self, original, restored):
        export = torch.load(restored.runtime.export_path, weights_only=True)
        expected = {key: value.half() for key, value in original.model.state_dict().items()
                    if original.branch != "lora" or ".lora." in key}
        self.assertEqual(set(export), set(expected))
        for key in expected:
            torch.testing.assert_close(export[key], expected[key], rtol=0, atol=0)
        report = json.loads(restored.runtime.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["stop_reason"], "epochs_complete")
        self.assertTrue(report["recovered_final_export"])
        self.assertEqual(report["optimizer_updates_this_invocation"], 0)
        self.assertEqual(report["microbatches_this_invocation"], 0)
        self.assertEqual(report["optimizer_updates"], 2)
        self.assertEqual(report["export_sha256"], sha256(restored.runtime.export_path))
        self.assertEqual(report["resume_sha256"], sha256(restored.runtime.resume_path))
        self.assertEqual(next(iter(restored.optimizer.state.values()))["step"].item(), 2)

    def test_final_resume_regenerates_stale_or_missing_export_for_all_branches(self):
        for branch in ("dpo", "lora", "distillation"):
            for missing in (False, True):
                with self.subTest(branch=branch, missing=missing):
                    case = self.fixture(branch, folder=f"{branch}_{missing}")
                    self.run_epoch(case)
                    resume_hash = sha256(case.runtime.resume_path)
                    if missing:
                        case.runtime.export_path.unlink()
                    else:
                        atomic_write(case.runtime.export_path, {"stale": torch.tensor(77.0)}, tensor=True)
                    restored = self.resumed_fixture(case)
                    position = restored.runtime.restore(restored.model, restored.optimizer, restored.scaler,
                                                        restored.runtime.load_resume(), case.count)
                    self.assertEqual(position, (case.args.epochs, 0))
                    restored.runtime.start_training()
                    restored.runtime.finish()  # No additional epoch or Adam update.
                    self.assertEqual(sha256(restored.runtime.resume_path), resume_hash)
                    self.assert_final_export(case, restored)

    def test_export_interruption_after_final_resume_can_fail_again_then_recover(self):
        case = self.fixture("lora")
        def interrupted_write(path, value, *, tensor=False):
            if Path(path) == case.runtime.export_path:
                saved = torch.load(case.runtime.resume_path, weights_only=True)
                if saved["step"] == case.count:
                    raise OSError("injected export interruption after final resume")
            return atomic_write(path, value, tensor=tensor)
        with patch("trainer.evomind_offline_runtime.atomic_write", side_effect=interrupted_write):
            with self.assertRaisesRegex(OSError, "export interruption"):
                self.run_epoch(case)
        final_resume = torch.load(case.runtime.resume_path, weights_only=True)
        self.assertEqual(final_resume["step"], case.count)
        self.assertEqual(final_resume["evomind"]["optimizer_updates"], 2)
        stale_hash = sha256(case.runtime.export_path)
        self.assertEqual(json.loads(case.runtime.report_path.read_text(encoding="utf-8"))["step"], 2)
        with self.assertRaisesRegex(RuntimeError, "No optimizer-boundary checkpoint"):
            case.runtime.finish()
        first = self.resumed_fixture(case)
        with patch("trainer.evomind_offline_runtime.atomic_write", side_effect=interrupted_write):
            with self.assertRaisesRegex(OSError, "export interruption"):
                first.runtime.restore(first.model, first.optimizer, first.scaler, first.runtime.load_resume(), case.count)
        self.assertEqual(sha256(case.runtime.export_path), stale_hash)
        with self.assertRaisesRegex(RuntimeError, "No optimizer-boundary checkpoint"):
            first.runtime.finish()
        restored = self.resumed_fixture(case)
        self.assertEqual(restored.runtime.restore(restored.model, restored.optimizer, restored.scaler,
                                                 restored.runtime.load_resume(), case.count), (1, 0))
        restored.runtime.finish()
        self.assert_final_export(case, restored)

    def test_completed_receipt_refuses_changed_export_or_resume_hash(self):
        for which in ("export_path", "resume_path"):
            with self.subTest(which=which):
                case = self.fixture("dpo", folder=which)
                self.run_epoch(case)
                getattr(case.runtime, which).write_bytes(b"changed after checkpoint publication")
                with self.assertRaisesRegex(RuntimeError, "changed or missing file binding"):
                    case.runtime.finish()
                self.assertNotEqual(json.loads(case.runtime.report_path.read_text(encoding="utf-8"))["status"], "completed")

    def test_finish_refuses_nonfinal_epoch_cursor(self):
        case = self.fixture("dpo", count=4)
        self.run_epoch(case, count=2)
        with self.assertRaisesRegex(RuntimeError, "final configured epoch tail"):
            case.runtime.finish()

    def test_resume_rejects_changed_accumulation_and_missing_scaler(self):
        case = self.fixture("dpo")
        self.run_epoch(case)
        args = copy.deepcopy(case.args)
        args.from_resume = 1
        args.accumulation_steps = 4
        resumed = OfflineRuntime(args, case.config, "dpo", "dpo", [("base", case.config, "full_sft")])
        with self.assertRaisesRegex(ValueError, "contract"):
            resumed.load_resume()
        args.accumulation_steps = 2
        resumed = OfflineRuntime(args, case.config, "dpo", "dpo", [("base", case.config, "full_sft")])
        saved = torch.load(case.runtime.resume_path, weights_only=True)
        del saved["scaler"]
        torch.save(saved, case.runtime.resume_path)
        with self.assertRaisesRegex(ValueError, "scaler"):
            resumed.load_resume()

    def test_resume_rejects_probe_promotion(self):
        case = self.fixture("dpo", max_steps=2, count=4)
        self.run_epoch(case)
        args = copy.deepcopy(case.args)
        args.max_steps = 0
        args.from_resume = 1
        resumed = OfflineRuntime(args, case.config, "dpo", "dpo", [("base", case.config, "full_sft")])
        with self.assertRaisesRegex(ValueError, "probe promotion"):
            resumed.load_resume()

    def test_missing_teacher_or_resume_is_not_random_fallback(self):
        case = self.fixture("distillation")
        teacher = SimpleNamespace(hidden_size=768, use_moe=True)
        with self.assertRaises(FileNotFoundError):
            OfflineRuntime(case.args, case.config, "distillation", "distillation", [("teacher", teacher, "full_sft")])
        args = copy.deepcopy(case.args)
        args.from_resume = 1
        runtime = OfflineRuntime(args, case.config, "distillation", "distillation", [("base", case.config, "full_sft")])
        with self.assertRaises(FileNotFoundError):
            runtime.load_resume()

    def test_nonfinite_gradient_blocks_optimizer(self):
        case = self.fixture("dpo")
        case.model.layer.lora.weight.grad = torch.full_like(case.model.layer.lora.weight, float("inf"))
        with self.assertRaises(RuntimeError):
            case.runtime.update(case.model, case.optimizer, case.scaler, case.model.parameters(), 0, 2, 3)
        self.assertEqual(case.runtime.updates, 0)
        self.assertEqual(case.optimizer.state, {})

    def test_probe_rejects_insufficient_complete_accumulations(self):
        with self.assertRaisesRegex(ValueError, "N\*accumulation_steps"):
            self.fixture("dpo", accumulation=4, max_steps=2, count=3)

    def test_mid_update_save_is_refused(self):
        case = self.fixture("dpo")
        with self.assertRaisesRegex(RuntimeError, "completed optimizer update"):
            case.runtime.save(case.model, case.optimizer, case.scaler, 0, 1, 3)

    def test_atomic_replace_failure_preserves_old_file(self):
        target = self.root / "atomic.pth"
        atomic_write(target, {"old": torch.tensor(1)}, tensor=True)
        original = target.read_bytes()
        with patch("trainer.evomind_offline_runtime.os.replace", side_effect=OSError("injected failure")):
            with self.assertRaises(OSError):
                atomic_write(target, {"new": torch.tensor(2)}, tensor=True)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp.*")), [])

    def test_mid_epoch_resume_matches_uninterrupted_tiny_adam(self):
        full = self.fixture("dpo", count=4, folder="full")
        self.run_epoch(full)
        partial = self.fixture("dpo", count=4, folder="partial")
        self.run_epoch(partial, count=2)
        args = copy.deepcopy(partial.args)
        args.from_resume = 1
        runtime = OfflineRuntime(args, partial.config, "dpo", "dpo", [("base", partial.config, "full_sft")])
        model = TinyModel(weight=9.0)
        optimizer = OfflineAdamW((p for p in model.parameters() if p.requires_grad), lr=0.03)
        position = runtime.restore(model, optimizer, partial.scaler, runtime.load_resume(), 4)
        self.assertEqual(position, (0, 2))
        resumed = SimpleNamespace(**{**vars(partial), "args": args, "runtime": runtime,
                                    "model": model, "optimizer": optimizer})
        self.run_epoch(resumed, count=2, iters=4, start_step=2)
        torch.testing.assert_close(model.layer.lora.weight, full.model.layer.lora.weight, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(next(iter(optimizer.state.values()))[key],
                                       next(iter(full.optimizer.state.values()))[key], rtol=0, atol=0)

    def test_distillation_refuses_teacher_vocab_truncation(self):
        case = self.fixture("distillation")
        module = importlib.import_module("trainer.train_distillation")
        teacher = TinyModel()
        original = teacher.forward

        def bad_forward(*args, **kwargs):
            result = original(*args, **kwargs)
            result.logits = torch.cat((result.logits, result.logits[..., :1]), dim=-1)
            return result

        teacher.forward = bad_forward
        values = dict(args=case.args, model=case.model, optimizer=case.optimizer, scaler=case.scaler,
                      runtime=case.runtime, autocast_ctx=contextlib.nullcontext())
        with patch.multiple(module, create=True, **values):
            with self.assertRaisesRegex(ValueError, "vocabulary dimensions differ"):
                module.train_epoch(0, [(torch.tensor([[1, 2, 1]]), torch.tensor([[-100, 0, 1]]))],
                                   1, teacher, case.config, alpha=0.5, temperature=1.5)
        self.assertFalse(case.runtime.resume_path.exists())

    def test_explicit_teacher_autocast_path_keeps_probe_contract(self):
        case = self.fixture("distillation", max_steps=2, count=4, teacher_autocast=1)
        self.run_epoch(case)
        case.runtime.finish()
        self.assertTrue(case.runtime.probe_complete)
        self.assertEqual(case.runtime.contract["parameters"]["teacher_autocast"], 1)

    def test_cpu_adam_does_not_initialize_cuda_even_when_gpu_is_reported_available(self):
        case = self.fixture("dpo")
        with patch("torch.cuda.is_available", return_value=True):
            self.run_epoch(case)
        self.assertEqual(case.runtime.updates, 2)


if __name__ == "__main__":
    unittest.main()
