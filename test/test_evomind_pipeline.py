"""No-model tests for the continuation contract and fail-closed dispatch."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evomind_continue as continuation
import evomind_pipeline_step as pipeline


class PipelineTests(unittest.TestCase):
    def test_vision_is_held_until_explicit_text_scope_completion(self):
        with self.assertRaisesRegex(RuntimeError, "VISION_HELD"):
            continuation.require_vision_enabled()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scope.json"
            with self.assertRaises(RuntimeError):
                continuation.require_vision_enabled(path)
            path.write_text(json.dumps({"vision_enabled": "true"}))
            with self.assertRaises(RuntimeError):
                continuation.require_vision_enabled(path)
            path.write_text(json.dumps({"vision_enabled": True}))
            continuation.require_vision_enabled(path)

    def test_plan_orders_gates_before_long_training(self):
        stages = continuation.build_manifest()["stages"]
        names = [row["name"] for row in stages]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 29)
        self.assertLess(names.index("real_encoder_cache_parity"), names.index("smoke_C"))
        self.assertLess(names.index("smoke_C"), names.index("official_v_two_epochs"))
        for seed in (42, 123, 2026):
            for variant in "ABC":
                self.assertLess(names.index(f"full_{variant}_seed{seed}"), names.index(f"test_{variant}_seed{seed}"))
        self.assertEqual(names[-1], "aggregate_report")

    def test_checked_in_manifest_matches_code(self):
        self.assertEqual(json.loads(continuation.MANIFEST.read_text(encoding="utf-8")), continuation.build_manifest())

    def test_controls_same_base_budget_except_cache_and_view(self):
        rows = {v: list(map(str, pipeline.common(v, 42, Path("unused")))) for v in "ABC"}
        for argv in rows.values():
            self.assertEqual(argv[argv.index("--init-weights") + 1], str(pipeline.BASE))
            self.assertEqual(argv[argv.index("--max-train-records") + 1], "20000")
            self.assertEqual(argv[argv.index("--max-seq-len") + 1], "768")
            self.assertEqual(argv[argv.index("--epochs") + 1], "2")
            self.assertEqual(argv[argv.index("--grad-accum") + 1], "4")
        self.assertNotIn("--cache-dir", rows["B"])
        self.assertIn("--cache-dir", rows["C"])

    def test_contract_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            pipeline.create_or_compare(path, {"seed": 42})
            pipeline.create_or_compare(path, {"seed": 42})
            with self.assertRaises(ValueError):
                pipeline.create_or_compare(path, {"seed": 123})
            self.assertEqual(json.loads(path.read_text()), {"seed": 42})

    def test_native_eval_not_mislabelled_custom(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "EVAL", Path(directory)), \
                patch.object(pipeline, "sha256", return_value="verified_test_hash"), patch.object(pipeline, "run") as invoke:
            pipeline.evaluate("official", 42)
        argv = invoke.call_args.args[0]
        self.assertIn("--official-weights", argv)
        self.assertNotIn("--checkpoint", argv)
        self.assertEqual(argv[argv.index("--split") + 1], "test")

    def test_probe_only_oom_can_trigger_single_fallback(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "ROOT", Path(directory)):
            output = Path(directory) / "artifacts/preflight"
            output.mkdir(parents=True)
            (output / "vision_official_b4.json").write_text(json.dumps({"status": "oom", "exit_code": 3}))
            (output / "vision_official_b1.json").write_text(json.dumps({"status": "success", "exit_code": 0}))
            pipeline.probe()
            result = json.loads((output / "vision_official.json").read_text())
            self.assertEqual(result["selected"], {"batch_size": 1, "accumulation_steps": 4})
            self.assertEqual(len(result["observations"]), 2)
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "ROOT", Path(directory)):
            output = Path(directory) / "artifacts/preflight"
            output.mkdir(parents=True)
            (output / "vision_official_b4.json").write_text(json.dumps({"status": "failed", "exit_code": 1}))
            with patch.object(pipeline.subprocess, "run") as invoke, self.assertRaises(RuntimeError):
                pipeline.probe()
            invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
