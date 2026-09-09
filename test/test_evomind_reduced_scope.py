"""Synthetic CPU-only scope/coverage tests, not trained-model evaluation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vision"))
import evomind_posttrain as train
import evomind_posttrain_evaluate as evaluate
import evomind_harness_eval as harness
from evomind_v.official_eval import examples


class ScopeTests(unittest.TestCase):
    def test_full_rlaif_and_six_branches(self):
        branches = {b["name"]: b for b in train.build_plan()["branches"]}
        self.assertEqual(set(branches), {"dpo", "lora", "grpo", "cispo", "agent_cispo", "distillation"})
        for name in ("grpo", "cispo"):
            self.assertEqual(branches[name]["data"], "dataset/rlaif.jsonl")
        with (ROOT / "dataset/rlaif.jsonl").open(encoding="utf-8") as stream:
            self.assertEqual(sum(bool(line.strip()) for line in stream), 19502)

    def test_exact_seven_official_benchmarks(self):
        self.assertEqual(harness.TASKS, ("ceval-valid", "cmmlu", "arc_easy", "piqa", "openbookqa", "hellaswag", "social_iqa"))
        keys = [j[0] for j in evaluate.jobs("full_sft", {"checkpoint": "full_sft_768.pth"})]
        self.assertEqual(len(keys), 10)
        self.assertNotIn("chinese", keys)
        self.assertNotIn("tool_seed123", keys)

    def test_six_vision_examples_have_no_fake_answers(self):
        rows, identity = examples(ROOT / "vision")
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(identity["images"]), 6)
        self.assertTrue(all(r["reference_answers"] == [] and r["task_type"] == "open" for r in rows))
        self.assertEqual(identity["prompt"], "<image>\n请描述这张图中的主要物体和场景。")

    def test_harness_full_coverage_not_smoke(self):
        fixture = {"results": {"synthetic": {"acc,none": .5}}, "config": {"limit": None},
                   "samples": {"synthetic": [{"doc_id": 0}, {"doc_id": 1}]},
                   "n-samples": {"synthetic": {"original": 2, "effective": 2}}}
        harness.validate_coverage(fixture)
        for mutation in ("limit", "count", "duplicate", "missing"):
            changed = copy.deepcopy(fixture)
            if mutation == "limit": changed["config"]["limit"] = 2
            if mutation == "count": changed["n-samples"]["synthetic"]["original"] = 3
            if mutation == "duplicate": changed["samples"]["synthetic"][1]["doc_id"] = 0
            if mutation == "missing": changed["samples"] = {}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                harness.validate_coverage(changed)

    def test_no_human_export_or_wait_but_failures_still_block(self):
        for success in (True, False):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                models = {"full_sft": {"checkpoint": "synthetic.pth", "sha256": "0" * 64}}
                def jobs(name, model):
                    return [("synthetic", [], root / "job", root / "job/result.json")]
                def execute(argv, dest):
                    dest.mkdir(parents=True, exist_ok=True)
                    result = dest / "result.json"
                    result.write_text('{"synthetic_test_only": true}')
                    return (0 if success else 1), result
                with patch.object(evaluate, "EVAL", root / "eval"), patch.object(evaluate, "RUN", root), \
                     patch.object(evaluate, "model_inputs", return_value=models), \
                     patch.object(evaluate, "build_plan", return_value={"branches": []}), \
                     patch.object(evaluate, "jobs", side_effect=jobs), patch.object(evaluate, "execute", side_effect=execute), \
                     patch.object(evaluate, "verify_evaluation_result"), patch.object(evaluate, "write_blind_package") as blind:
                    self.assertEqual(evaluate.run_evaluations(), 0 if success else 2)
                    blind.assert_not_called()
                receipt = json.loads((root / "text_acceptance.json").read_text())
                self.assertEqual(receipt["status"], "accepted" if success else "pending")
                self.assertEqual(receipt["human_review"]["status"], "cancelled_by_user")


if __name__ == "__main__":
    unittest.main()
