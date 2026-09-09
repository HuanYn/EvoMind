"""Tiny offline tests: no real benchmark download, model checkpoint or GPU."""
import argparse
import csv
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evomind_chinese_eval as evaluator


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "artifacts/test_tmp").mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "artifacts/test_tmp")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.data = self.directory / "data"
        self.sources = {}
        for name in ("ceval", "cmmlu"):
            spec = dict(evaluator.SOURCES[name], subjects=2, rows=3)
            source = evaluator.archive_path(self.data, name, spec)
            source.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(source, "w") as archive:
                for subject, answers in (("alpha", ("A",)), ("beta", ("B", "C"))):
                    stream = io.StringIO(newline="")
                    writer = csv.writer(stream)
                    writer.writerow(("id", "question", "A", "B", "C", "D", "answer") if name == "ceval"
                                    else ("", "Question", "A", "B", "C", "D", "Answer"))
                    for index, answer in enumerate(answers):
                        writer.writerow((index, "这是测试题，正确答案不可放入提示词。", "甲", "乙", "丙", "丁", answer))
                    archive.writestr(f"{spec['split']}/{subject}{spec['member_suffix']}", stream.getvalue().encode("utf-8"))
                # Irrelevant dev/test CSVs and malicious extraction paths must
                # never be extracted or included in the selected public split.
                archive.writestr("../../untrusted.py", "raise RuntimeError('not executed')")
                archive.writestr("dev/ignored.csv", "not the requested split")
            spec.update(bytes=source.stat().st_size, sha256=evaluator.sha256(source))
            self.sources[name] = spec
        with mock.patch.object(evaluator.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")):
            self.manifest = evaluator.prepare_data(self.data, self.sources)
        self.checkpoint = self.directory / "model.pth"
        self.checkpoint.write_bytes(b"fake checkpoint: never loaded")
        self.args = argparse.Namespace(checkpoint=self.checkpoint, lora=None, output_dir=self.directory / "eval",
                                       data_dir=self.data, resume=False, device="cpu", dtype="float32", max_context=32768)

    @staticmethod
    def scorer(record):
        # Always A: alpha=1/1, beta=0/2; macro=.5, micro=1/3.
        return {"scores": {"A": -1.0, "B": -2.0, "C": -3.0, "D": -4.0}, "context_tokens": 10,
                "candidate_tokens": {key: 1 for key in evaluator.CHOICES}}

    def run_eval(self, scorer=None):
        return evaluator.evaluate(self.args, self.manifest,
                                  scorer_factory=lambda args: self.scorer if scorer is None else scorer)

    def test_prepare_full_counts_hashes_and_no_remote_execution(self):
        verified = evaluator.validate_prepared(self.data, self.sources)
        self.assertEqual(verified, self.manifest)
        for name in self.sources:
            details = verified["datasets"][name]
            self.assertEqual(details["rows"], 3)
            self.assertEqual(len(details["subjects"]), 2)
            records = list(evaluator.iter_jsonl(self.data / details["records_file"]))
            self.assertEqual([row["source_id"] for row in records], ["0", "0", "1"])
            self.assertIn("测试题", records[0]["question"])
        self.assertFalse((self.directory / "untrusted.py").exists())

    def test_changed_normalized_records_fail_even_if_manifest_hash_is_changed(self):
        path = self.data / "ceval.records.jsonl"
        path.write_bytes(path.read_bytes().replace("测试题".encode(), "改试题".encode()))
        manifest = evaluator.read_json(self.data / "manifest.json")
        manifest["datasets"]["ceval"]["records_sha256"] = evaluator.sha256(path)
        evaluator.atomic_json(self.data / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "pinned source"):
            evaluator.validate_prepared(self.data, self.sources)

    def test_source_archive_corruption_and_incomplete_budget_fail(self):
        spec = self.sources["ceval"]
        source = evaluator.archive_path(self.data, "ceval", spec)
        with self.assertRaisesRegex(ValueError, "all 4 rows"):
            evaluator.normalize_archive(source, "ceval", dict(spec, rows=4))
        with source.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changed pinned"):
            evaluator.validate_prepared(self.data, self.sources)

    def test_duplicate_id_and_invalid_gold_rejected(self):
        for duplicate in (False, True):
            spec = dict(self.sources["ceval"], subjects=1, rows=2)
            path = self.directory / f"bad{duplicate}.zip"
            with zipfile.ZipFile(path, "w") as archive:
                answer = "A" if duplicate else "E"
                archive.writestr("val/alpha_val.csv", f"id,question,A,B,C,D,answer\n0,q,a,b,c,d,A\n0,q,a,b,c,d,{answer}\n")
            spec.update(bytes=path.stat().st_size, sha256=evaluator.sha256(path))
            with self.assertRaisesRegex(ValueError, "Duplicate|invalid answer"):
                evaluator.normalize_archive(path, "ceval", spec)

    def test_full_evaluation_macro_micro_receipts(self):
        summary = self.run_eval()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["checkpoint_sha256"], evaluator.sha256(self.checkpoint))
        for name, report in summary["datasets"].items():
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["rows"], 3)
            self.assertAlmostEqual(report["accuracy"], 1 / 3)
            self.assertEqual(report["macro_accuracy"], .5)
            self.assertEqual(report["predictions_sha256"], evaluator.sha256(self.args.output_dir / "predictions" / f"{name}.jsonl"))
        self.assertIn("not been ruled out", " ".join(summary["limitations"]))
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_eval()

    def test_completed_resume_does_not_load_model_and_rejects_changed_checkpoint(self):
        summary = self.run_eval()
        self.args.resume = True
        with mock.patch.object(evaluator, "load_scorer", side_effect=AssertionError("model forbidden")):
            again = evaluator.evaluate(self.args, self.manifest,
                                       scorer_factory=lambda args: self.fail("completed resume loaded a model"))
        self.assertEqual(summary, again)
        self.checkpoint.write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.run_eval()

    def test_failed_run_resumes_valid_prefix_without_repeating_rows(self):
        observed = []
        def failing(record):
            observed.append(record["sample_id"])
            if len(observed) == 3:
                raise RuntimeError("synthetic interruption")
            return self.scorer(record)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            self.run_eval(failing)
        failed = evaluator.read_json(self.args.output_dir / "summary.json")
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(failed["completion_claim"])
        old = (self.args.output_dir / "predictions/ceval.jsonl").read_bytes()
        resumed = []
        def succeeding(record):
            resumed.append(record["sample_id"])
            return self.scorer(record)
        self.args.resume = True
        summary = self.run_eval(succeeding)
        self.assertEqual(len(resumed), 4)
        self.assertEqual(resumed[0], observed[-1])
        self.assertTrue((self.args.output_dir / "predictions/ceval.jsonl").read_bytes().startswith(old))
        self.assertEqual(summary["status"], "completed")

    def test_corrupted_prediction_tail_fails_before_loading(self):
        self.run_eval()
        path = self.args.output_dir / "predictions/ceval.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"partial":')
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, "Incomplete JSONL tail"):
            evaluator.evaluate(self.args, self.manifest, scorer_factory=lambda args: self.fail("model loaded"))

    def test_tampered_prediction_or_summary_is_rejected(self):
        self.run_eval()
        path = self.args.output_dir / "predictions/ceval.jsonl"
        records = list(evaluator.iter_jsonl(path))
        records[0]["gold"] = "D"
        path.write_bytes(b"".join(evaluator.encoded(record) + b"\n" for record in records))
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.run_eval()

    def test_lora_hash_and_protocol_are_part_of_resume_contract(self):
        self.args.lora = self.directory / "lora_medical.pth"
        self.args.lora.write_bytes(b"medical adapter")
        summary = self.run_eval()
        self.assertEqual(summary["lora_sha256"], evaluator.sha256(self.args.lora))
        self.args.resume = True
        self.args.max_context = 1024
        with self.assertRaisesRegex(ValueError, "differs"):
            self.run_eval()

    def test_ties_invalid_scores_and_prompt_has_no_gold(self):
        self.assertEqual(evaluator.select_choice(dict.fromkeys(evaluator.CHOICES, -1.0)), "A")
        for value in (float("nan"), float("inf"), .1, True):
            scores = dict.fromkeys(evaluator.CHOICES, -1.0)
            scores["A"] = value
            with self.assertRaises(ValueError):
                evaluator.select_choice(scores)
        record = next(evaluator.iter_jsonl(self.data / "ceval.records.jsonl"))
        prompt = evaluator.question_prompt(record)
        self.assertIn("A. 甲", prompt)
        record["answer"] = "D"
        self.assertEqual(prompt, evaluator.question_prompt(record))

    def test_prepare_only_cli_does_not_import_torch_and_writes_no_scores(self):
        real_import = __import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "datasets", "transformers"):
                raise AssertionError(f"prepare-only imported {name}")
            return real_import(name, *args, **kwargs)
        output = self.directory / "prepared"
        with mock.patch.object(evaluator, "SOURCES", self.sources), mock.patch("builtins.__import__", side_effect=guarded):
            status = evaluator.main(["--prepare-only", "--data-dir", str(self.data), "--output-dir", str(output)])
        self.assertEqual(status, 0)
        receipt = evaluator.read_json(output / "preparation.json")
        self.assertEqual(receipt["status"], "prepared")
        self.assertFalse(receipt["evaluation_executed"])
        self.assertFalse((output / "summary.json").exists())

    def test_atomic_json_preserves_existing_on_write_failure(self):
        path = self.directory / "atomic.json"
        evaluator.atomic_json(path, {"old": True})
        with mock.patch.object(evaluator.os, "replace", side_effect=OSError("synthetic")):
            with self.assertRaises(OSError):
                evaluator.atomic_json(path, {"new": True})
        self.assertEqual(evaluator.read_json(path), {"old": True})
        self.assertEqual(list(self.directory.glob("atomic.json.*.tmp")), [])


class ConditionalLikelihoodTests(unittest.TestCase):
    """Real torch CPU tensors; fail immediately if any CUDA initialization occurs."""
    @classmethod
    def setUpClass(cls):
        import datasets  # noqa: F401 -- Windows DLL import order
        import torch
        cls.torch = torch

    def test_candidate_scoring_single_and_multiple_tokens(self):
        torch = self.torch
        class Tokenizer:
            def __init__(self, multiple):
                self.multiple = multiple
            def encode(self, text, **kwargs):
                if text in evaluator.CHOICES:
                    token = evaluator.CHOICES.index(text) + 1
                    return [token, 4] if self.multiple and text == "D" else [token]
                return [0, 0, 0]
            def apply_chat_template(self, messages, **kwargs):
                assert kwargs["open_thinking"] is False
                return "prompt"
        class Model:
            config = types.SimpleNamespace(max_position_embeddings=10)
            def __init__(self):
                self.inputs = []
            def eval(self):
                return self
            def __call__(self, input_ids, use_cache, logits_to_keep):
                self.inputs.append(input_ids.tolist()[0])
                # Same categorical distribution at each position. D wins if
                # one token, but sum of two D log-probs is worse than one C.
                logits = torch.tensor([0.0, 1.0, 2.0, 3.9, 4.0]).repeat(1, input_ids.shape[1], 1)
                return types.SimpleNamespace(logits=logits[:, -logits_to_keep:, :])
        record = {"sample_id": "tiny", "question": "题", "choices": dict.fromkeys(evaluator.CHOICES, "选项")}
        with mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            for multiple in (False, True):
                model = Model()
                scorer = evaluator.ChoiceScorer(model, Tokenizer(multiple), device="cpu", dtype="float32", max_context=10)
                result = scorer(record)
                self.assertEqual(evaluator.select_choice(result["scores"]), "C" if multiple else "D")
                expected = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0, 3.9, 4.0]), dim=-1)
                self.assertAlmostEqual(result["scores"]["D"], expected[4].item() * (2 if multiple else 1), places=6)
                self.assertEqual(len(model.inputs), 4 if multiple else 1)
                if multiple:
                    self.assertEqual(model.inputs[-1], [0, 0, 0, 4])
                self.assertEqual(result["context_tokens"], 3)

    def test_context_overflow_is_not_truncated(self):
        torch = self.torch
        tokenizer = mock.Mock()
        tokenizer.encode.side_effect = lambda text, **kwargs: [evaluator.CHOICES.index(text)] if text in evaluator.CHOICES else list(range(8))
        tokenizer.apply_chat_template.return_value = "prompt"
        model = mock.Mock(config=types.SimpleNamespace(max_position_embeddings=8))
        model.eval.return_value = model
        with mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            scorer = evaluator.ChoiceScorer(model, tokenizer, device="cpu", dtype="float32", max_context=8)
            with self.assertRaisesRegex(ValueError, "no truncation"):
                scorer({"sample_id": "too-long", "question": "题", "choices": dict.fromkeys(evaluator.CHOICES, "选项")})
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
