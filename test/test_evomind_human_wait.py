"""Deterministic waiting tests; injected sleep, synthetic annotations, no GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evomind_continue import wait_for_human_annotations
from evomind_run import atomic_json


class HumanWaitTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "artifacts/test_tmp"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name)
        self.annotations = self.root / "eval/blind_review/annotations.csv"
        self.annotations.parent.mkdir(parents=True)
        self.annotations.write_text("empty synthetic test fixture\n", encoding="utf-8")
        self.receipt = {"status": "pending", "machine_evaluations_complete": True,
            "evaluation_state": str(self.root / "eval/state.json"), "human_review": {"status": "awaiting_human_annotations"}}
        self.persist()

    def tearDown(self):
        self.temp.cleanup()

    def persist(self):
        atomic_json(self.root / "text_acceptance.json", self.receipt)

    def test_wait_metadata_then_auto_validate_and_continue(self):
        sleeps, evaluations = [], []
        def sleep(seconds):
            sleeps.append(seconds)
            self.annotations.write_text("changed fixture, NOT genuine annotations\n", encoding="utf-8")
        def evaluate():
            evaluations.append(True)
            self.receipt.update(status="accepted", human_review={"status": "annotated"})
            self.persist()
            return 0
        self.assertEqual(wait_for_human_annotations(self.root, evaluate, sleep), 0)
        self.assertEqual(sleeps, [30])
        self.assertEqual(len(evaluations), 1)
        self.assertEqual(json.loads((self.root / "human_wait.json").read_text())["status"], "accepted")

    def test_machine_failure_does_not_wait_or_rerun_gpu_jobs(self):
        self.receipt["machine_evaluations_complete"] = False
        self.persist()
        def forbidden(*args):
            self.fail("Must not wait/evaluate on machine failure")
        self.assertEqual(wait_for_human_annotations(self.root, forbidden, forbidden), 2)

    def test_unchanged_annotations_do_not_rerun_evaluations(self):
        attempts = []
        def sleep(seconds):
            attempts.append(seconds)
            if len(attempts) == 3:
                raise InterruptedError("test stops wait; no actual external annotation")
        def forbidden():
            self.fail("Unchanged annotation must not trigger re-evaluation")
        with self.assertRaises(InterruptedError):
            wait_for_human_annotations(self.root, forbidden, sleep)
        self.assertEqual(len(attempts), 3)


if __name__ == "__main__":
    unittest.main()
