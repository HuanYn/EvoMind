"""Portable launcher contract tests; standard library only, no training launches."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evomind_reproduce", ROOT / "scripts" / "evomind_reproduce.py")
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


class ReproduceTests(unittest.TestCase):
    def test_all_stages_dispatch_direct_trainers_with_valid_declared_options(self):
        for stage, filename in reproduce.TRAINERS.items():
            with self.subTest(stage=stage):
                plan = reproduce.build_command(stage)
                self.assertEqual(Path(plan["cwd"]), ROOT / "trainer")
                self.assertEqual(Path(plan["argv"][2]), ROOT / "trainer" / filename)
                self.assertEqual(reproduce.option_value(plan["argv"], "--device"), "cpu")
                self.assertEqual(reproduce.option_value(plan["argv"], "--num_workers"), "0")
                self.assertTrue(all(token in reproduce.trainer_options(stage) for token in plan["argv"][3:] if token.startswith("--")))

    def test_rl_loss_and_export_prefix_match_selected_stage(self):
        for stage in ("grpo", "cispo"):
            argv = reproduce.build_command(stage)["argv"]
            self.assertIn("--evomind_runtime", argv)
            self.assertEqual(reproduce.option_value(argv, "--loss_type"), stage)
            self.assertEqual(reproduce.option_value(argv, "--save_weight"), stage)
        with self.assertRaisesRegex(ValueError, "requires --loss_type"):
            reproduce.build_command("grpo", ["--loss_type=cispo"])

    def test_paths_are_repo_relative_and_values_with_spaces_are_preserved(self):
        argv = reproduce.build_command("dpo", ["--data_path", "local data/preference.jsonl", "--init_dir=parents", "--save_dir", "runs/new dpo"])["argv"]
        self.assertEqual(reproduce.option_value(argv, "--data_path"), str(ROOT / "local data" / "preference.jsonl"))
        self.assertEqual(reproduce.option_value(argv, "--init_dir"), str(ROOT / "parents"))
        self.assertEqual(reproduce.option_value(argv, "--save_dir"), str(ROOT / "runs" / "new dpo"))

    def test_unknown_options_and_missing_path_values_fail_before_launch(self):
        with self.assertRaisesRegex(ValueError, "Unknown trainer"):
            reproduce.build_command("sft", ["--init_dir", "parents"])
        with self.assertRaisesRegex(ValueError, "requires a (path|value)"):
            reproduce.build_command("pretrain", ["--data_path"])

    def test_default_plan_is_non_mutating_and_needs_no_assets(self):
        with mock.patch.object(reproduce.subprocess, "call", side_effect=AssertionError("must not launch")), mock.patch.object(reproduce, "validate_inputs", side_effect=AssertionError("must not inspect training assets")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(reproduce.main(["train", "cispo", "--dry-run"]), 0)
            self.assertFalse(json.loads(output.getvalue())["execute"])

    def test_missing_assets_and_existing_outputs_fail_before_execute(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            plan = {"stage": "pretrain", "argv": ["python", "-B", "train_pretrain.py", "--data_path", str(root / "data.jsonl"), "--save_dir", str(root / "out"), "--save_weight", "pretrain"]}
            with self.assertRaisesRegex(ValueError, "Missing local inputs"):
                reproduce.validate_inputs(plan, root=root)
            (root / "model").mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (root / "model" / name).touch()
            (root / "data.jsonl").touch()
            reproduce.validate_inputs(plan, root=root)
            (root / "out").mkdir()
            (root / "out" / "pretrain_768.pth").touch()
            with self.assertRaisesRegex(ValueError, "already exists"):
                reproduce.validate_inputs(plan, root=root)

    def test_trainer_help_only_passes_help(self):
        with mock.patch.object(reproduce.subprocess, "call", return_value=0) as call:
            self.assertEqual(reproduce.main(["trainer-help", "cispo"]), 0)
        self.assertEqual(call.call_args.args[0][-1], "--help")
        self.assertEqual(Path(call.call_args.args[0][-2]).name, "train_grpo.py")


if __name__ == "__main__":
    unittest.main()
