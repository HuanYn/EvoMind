"""CPU/stdlib orchestration contracts; no model imports or GPU discovery."""
import copy
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evomind_posttrain as pipeline
import evomind_continue as continuation
import evomind_posttrain_evaluate as evaluation
from evomind_run import atomic_json, sha256
from test_evomind_runtime_receipts import emit_receipt_fixture


def repository_relative_plan(plan):
    """Compare frozen recipes across checkouts without changing their targets."""
    # The checked-in historical plan has Windows paths even in a Linux clone.
    # Pure paths parse that syntax without consulting the host filesystem.
    path_type = PureWindowsPath if PureWindowsPath(plan["base_run"]).drive else PurePosixPath
    repository = path_type(plan["base_run"]).parents[2]  # artifacts/runs/<run>

    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str):
            path = path_type(value)
            if path.is_absolute() and path.is_relative_to(repository):
                return "<repository>/" + path.relative_to(repository).as_posix()
        return value

    return normalize(plan)


class PosttrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.branch = pipeline.build_plan()["branches"][0]
        self.base = self.root / "base/full_sft_768.pth"
        self.base.parent.mkdir()
        self.base.write_bytes(b"base fixture, not real weights")

    def tearDown(self):
        self.temp.cleanup()

    def emit(self, directory, *, probe, epochs=1, updates=2):
        emit_receipt_fixture(self.branch, directory, probe=probe, epochs=epochs, updates=updates)

    def test_all_branches_keep_official_epochs_and_groups(self):
        plan = pipeline.build_plan()
        branches = {b["name"]: b for b in plan["branches"]}
        self.assertEqual(set(branches), {"dpo", "grpo", "cispo", "agent_cispo", "lora", "distillation"})
        stored = json.loads(pipeline.PLAN.read_text(encoding="utf-8"))
        self.assertEqual(repository_relative_plan(stored), repository_relative_plan(plan))
        scope = json.loads((pipeline.ROOT / "configs/text_alignment_scope.json").read_text(encoding="utf-8"))
        self.assertNotIn("agent_grpo", scope["required_coverage"])
        self.assertEqual(branches["agent_cispo"]["options"]["loss_type"], "cispo")
        for b in branches.values():
            budgets = {r["batch_size"] * r["accumulation_steps"] for r in b["batch_candidates"]}
            self.assertEqual(len(budgets), 1)
            self.assertEqual(b["probe_optimizer_updates"], 2)
        self.assertEqual(branches["lora"]["options"]["epochs"], 10)
        self.assertEqual(branches["distillation"]["options"]["epochs"], 6)
        self.assertEqual(branches["grpo"]["options"]["num_generations"], 6)
        self.assertEqual(branches["agent_cispo"]["options"]["max_total_len"], 2500)
        self.assertNotIn("ppo", scope["required_coverage"])
        self.assertFalse(scope["human_review_required"])
        self.assertEqual(branches["grpo"]["data"], "dataset/rlaif.jsonl")
        self.assertEqual(branches["cispo"]["data"], "dataset/rlaif.jsonl")

    def test_plan_relocation_preserves_recipe_values_and_relative_targets(self):
        windows = {"base_run": r"E:\old\evomind\artifacts\runs\base",
                   "run_dir": r"E:\old\evomind\artifacts\runs\posttrain",
                   "branches": [{"data": "dataset/rlaif.jsonl", "options": {
                       "epochs": 1, "num_generations": 6,
                       "reward_model_path": r"E:\old\evomind\models\reward"}}]}
        linux = {"base_run": "/tmp/clone/artifacts/runs/base",
                 "run_dir": "/tmp/clone/artifacts/runs/posttrain",
                 "branches": [{"data": "dataset/rlaif.jsonl", "options": {
                     "epochs": 1, "num_generations": 6,
                     "reward_model_path": "/tmp/clone/models/reward"}}]}
        original = copy.deepcopy(windows)
        self.assertEqual(repository_relative_plan(windows), repository_relative_plan(linux))
        self.assertEqual(windows, original)
        for changed_value in ("/tmp/clone/models/other_reward", "/tmp/external/models/reward"):
            changed = copy.deepcopy(linux)
            changed["branches"][0]["options"]["reward_model_path"] = changed_value
            self.assertNotEqual(repository_relative_plan(windows), repository_relative_plan(changed))
        for key, value in (("epochs", 2), ("num_generations", 4)):
            changed = copy.deepcopy(linux)
            changed["branches"][0]["options"][key] = value
            self.assertNotEqual(repository_relative_plan(windows), repository_relative_plan(changed))
        changed = copy.deepcopy(linux)
        changed["run_dir"] = "/tmp/clone/artifacts/runs/other_run"
        self.assertNotEqual(repository_relative_plan(windows), repository_relative_plan(changed))

    def test_probe_and_full_commands_are_isolated_and_base_not_chained(self):
        b = self.branch
        probe = pipeline.command(b, b["batch_candidates"][0], self.root / "probe", self.base, probe=True)
        full = pipeline.command(b, b["batch_candidates"][0], self.root / "full", self.base, probe=False)
        self.assertEqual(probe[probe.index("--max_steps")+1], "2")
        self.assertEqual(full[full.index("--max_steps")+1], "0")
        self.assertEqual(probe[probe.index("--init_dir")+1], str(self.base.parent))
        self.assertNotEqual(probe[probe.index("--save_dir")+1], full[full.index("--save_dir")+1])

    def test_base_requires_both_completed_and_snapshot_hash(self):
        state = {"stages": {"pretrain": {"status": "completed"}, "full_sft": {"status": "running"}}}
        atomic_json(self.root / "state.json", state)
        with patch.object(pipeline, "BASE_RUN", self.root), self.assertRaisesRegex(RuntimeError, "full_sft"):
            pipeline.base_checkpoint()
        state["stages"]["full_sft"] = {"status": "completed", "snapshot": {"path": str(self.base), "sha256": sha256(self.base)}}
        atomic_json(self.root / "state.json", state)
        with patch.object(pipeline, "BASE_RUN", self.root):
            self.assertEqual(pipeline.base_checkpoint(), self.base)

    def test_probe_cannot_count_as_full_or_one_update(self):
        self.emit(self.root, probe=True)
        pipeline.check_receipt(self.branch, self.root, probe=True)
        with self.assertRaises(ValueError):
            pipeline.check_receipt(self.branch, self.root, probe=False)
        self.emit(self.root, probe=True, updates=1)
        with self.assertRaises(ValueError):
            pipeline.check_receipt(self.branch, self.root, probe=True)

    def test_full_receipt_requires_exact_budget_and_cursor(self):
        self.emit(self.root, probe=False, epochs=2)
        with self.assertRaises(ValueError):
            pipeline.check_receipt(self.branch, self.root, probe=False)
        self.emit(self.root, probe=False)
        path = self.root / "weights/dpo_768.runtime.json"
        value = json.loads(path.read_text()); value["step"] = 7; atomic_json(path, value)
        with self.assertRaises(ValueError):
            pipeline.check_receipt(self.branch, self.root, probe=False)

    def test_record_checks_receipt_as_well_as_weight(self):
        self.emit(self.root, probe=False)
        record = pipeline.check_receipt(self.branch, self.root, probe=False)
        pipeline.verify_record(record)
        Path(record["runtime_receipt"]).write_text("{}")
        with self.assertRaises(ValueError):
            pipeline.verify_record(record)

    def test_record_checks_complete_resume_hash(self):
        self.emit(self.root, probe=False)
        record = pipeline.check_receipt(self.branch, self.root, probe=False)
        Path(record["resume_checkpoint"]).write_bytes(b"changed complete resume fixture")
        with self.assertRaisesRegex(ValueError, "resume checkpoint"):
            pipeline.verify_record(record)

    def test_successful_probe_launches_full_without_promotion(self):
        calls = []
        def execute(argv, directory):
            calls.append(argv)
            self.emit(directory, probe=argv[argv.index("--max_steps")+1] == "2")
            log = directory / "console.log"; log.write_text("ok")
            return 0, log
        state = {"branches": {}}
        with patch.object(pipeline, "RUN", self.root / "run"), patch.object(pipeline, "validate_assets"):
            pipeline.run_branch(self.branch, state, lambda: None, self.base, execute)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][calls[1].index("--from_resume")+1], "0")
        self.assertEqual(state["branches"]["dpo"]["status"], "completed")

    def test_remote_dispatch_blocks_local_restart_before_any_work(self):
        for status in ("remote_pending", "training", "failed"):
            with self.subTest(status=status):
                state = {"branches": {"dpo": {"status": status, "probes": [],
                                               "remote_dispatch": {"run": "remote/formal"}}}}
                original = copy.deepcopy(state)
                execute, persist = Mock(), Mock()
                with patch.object(pipeline, "RUN", self.root / "run"), \
                     patch.object(pipeline, "validate_assets") as validate:
                    with self.assertRaisesRegex(RuntimeError, "remote supervisor"):
                        pipeline.run_branch(self.branch, state, persist, self.base, execute)
                validate.assert_not_called()
                execute.assert_not_called()
                persist.assert_not_called()
                self.assertEqual(state, original)
                self.assertFalse((self.root / "run").exists())

    def test_non_oom_does_not_trigger_batch_fallback(self):
        calls = []
        def execute(argv, directory):
            calls.append(argv); directory.mkdir(parents=True)
            log = directory / "console.log"; log.write_text("invalid data")
            return 1, log
        with patch.object(pipeline, "RUN", self.root / "run"), patch.object(pipeline, "validate_assets"):
            with self.assertRaisesRegex(RuntimeError, "not OOM"):
                pipeline.run_branch(self.branch, {"branches": {}}, lambda: None, self.base, execute)
        self.assertEqual(len(calls), 1)

    def test_full_failure_before_checkpoint_preserves_evidence_and_uses_fresh_directory(self):
        state = {"branches": {}}
        calls = []
        def first(argv, directory):
            directory.mkdir(parents=True, exist_ok=True)
            log = directory / "console.log"
            log.write_text("preserved attempted run")
            if argv[argv.index("--max_steps")+1] == "2":
                self.emit(directory, probe=True)
                return 0, log
            return 1, log
        def retry(argv, directory):
            calls.append((argv, directory))
            self.emit(directory, probe=False)
            log = directory / "console.log"; log.write_text("fresh formal attempt")
            return 0, log
        with patch.object(pipeline, "RUN", self.root / "run"), patch.object(pipeline, "validate_assets"):
            with self.assertRaisesRegex(RuntimeError, "full training exited"):
                pipeline.run_branch(self.branch, state, lambda: None, self.base, first)
            pipeline.run_branch(self.branch, state, lambda: None, self.base, retry)
        self.assertEqual(calls[0][1].name, "full_retry1")
        self.assertEqual(calls[0][0][calls[0][0].index("--from_resume")+1], "0")
        self.assertEqual((self.root / "run/dpo/full/console.log").read_text(), "preserved attempted run")
        self.assertEqual(len(state["branches"]["dpo"]["abandoned_full_attempts"]), 1)

    def test_full_retry_with_checkpoint_resumes_the_recorded_attempt(self):
        probe = self.root / "run/dpo/probe_0"
        self.emit(probe, probe=True)
        full = self.root / "run/dpo/full_retry2"
        resume = full / "checkpoints/dpo_768_resume.pth"
        resume.parent.mkdir(parents=True); resume.write_bytes(b"mock resume, not a real checkpoint")
        candidate = self.branch["batch_candidates"][0]
        state = {"branches": {"dpo": {"status": "failed", "selected": candidate, "full_directory": str(full),
            "probes": [{"candidate": candidate, "status": "passed", "directory": str(probe),
                        "output": pipeline.check_receipt(self.branch, probe, probe=True)}]}}}
        def execute(argv, directory):
            self.assertEqual(directory, full)
            self.assertEqual(argv[argv.index("--from_resume")+1], "1")
            self.emit(directory, probe=False)
            log = directory / "console.log"; log.write_text("resumed")
            return 0, log
        with patch.object(pipeline, "RUN", self.root / "run"), patch.object(pipeline, "validate_assets"):
            pipeline.run_branch(self.branch, state, lambda: None, self.base, execute)
        self.assertEqual(state["branches"]["dpo"]["status"], "completed")

    def test_evaluation_includes_thinking_seeds_benchmarks_tools_and_lora_base(self):
        model = {"checkpoint": str(self.base), "sha256": sha256(self.base), "lora": "adapter.pth"}
        jobs = evaluation.jobs("lora", model)
        self.assertEqual({j[0] for j in jobs}, {"thinking_off", "thinking_on", "tool_seed42", *(f"harness_{task}" for task in evaluation.TASKS)})
        for _, argv, _, _ in jobs:
            self.assertIn("--lora", argv)

    def test_pending_acceptance_cannot_enable_vision(self):
        atomic_json(self.root / "text_acceptance.json", {"status": "pending"})
        with patch.object(pipeline, "RUN", self.root), self.assertRaisesRegex(RuntimeError, "pending"):
            continuation.enable_vision_after_acceptance()


if __name__ == "__main__":
    unittest.main()
