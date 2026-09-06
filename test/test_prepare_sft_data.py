import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_sft_data import (
    _validate_candidates,
    canonical_record,
    prepare_sft_subsets,
    stable_u64,
)
from model.tokenizer import CharTokenizer


class TestPrepareSFTData(unittest.TestCase):
    @staticmethod
    def records(count: int):
        return [
            {
                "conversations": [
                    {"role": "user", "content": f"问题{index}"},
                    {"role": "assistant", "content": f"答案{index}"},
                ]
            }
            for index in range(count)
        ]

    @staticmethod
    def tokenizer_for(*values) -> CharTokenizer:
        text = "system\nuser\nassistant\ntool\n<think></think>"
        text += "".join(json.dumps(value, ensure_ascii=False) for value in values)
        return CharTokenizer.build(text)

    @staticmethod
    def write_source(path: Path, lines: list[str]) -> None:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    @staticmethod
    def output_lines(path: Path) -> list[str]:
        return [line for line in path.read_text(encoding="utf-8").splitlines() if line]

    def run_prepare(
        self,
        directory: Path,
        source_lines: list[str],
        tokenizer: CharTokenizer,
        *,
        seed: int = 42,
        train_size: int = 20,
        val_size: int = 20,
        smoke_size: int = 5,
        overwrite: bool = False,
    ):
        directory.mkdir(parents=True, exist_ok=True)
        input_path = directory / "source.jsonl"
        train_path = directory / "train.jsonl"
        val_path = directory / "val.jsonl"
        smoke_path = directory / "smoke.jsonl"
        report_path = directory / "report.json"
        self.write_source(input_path, source_lines)
        report = prepare_sft_subsets(
            input_path=input_path,
            tokenizer=tokenizer,
            train_output=train_path,
            val_output=val_path,
            smoke_output=smoke_path,
            report_output=report_path,
            train_size=train_size,
            val_size=val_size,
            smoke_size=smoke_size,
            max_length=128,
            val_ratio=0.5,
            reserve_ratio=1.0,
            empty_think_ratio=0.0,
            seed=seed,
            progress_every=0,
            overwrite=overwrite,
        )
        return report, input_path, train_path, val_path, smoke_path, report_path

    def test_canonical_record_deduplicates_key_order_and_ignores_metadata(self):
        first = {
            "source": "first",
            "conversations": [
                {"role": "user", "content": "问"},
                {"role": "assistant", "content": "答"},
            ],
        }
        second = {
            "metadata": {"row": 99},
            "conversations": [
                {"content": "问", "role": "user"},
                {"content": "答", "role": "assistant"},
            ],
        }
        self.assertEqual(canonical_record(first), canonical_record(second))
        self.assertEqual(set(json.loads(canonical_record(first))), {"conversations"})

    def test_hash_split_sampling_dedup_leakage_smoke_determinism_and_report(self):
        records = self.records(400)
        canonical_lines = [json.dumps(record, ensure_ascii=False) for record in records]
        duplicate_with_reordered_keys = json.dumps(
            {
                "conversations": [
                    {"content": "问题0", "role": "user"},
                    {"content": "答案0", "role": "assistant"},
                ],
                "ignored_top_level_metadata": True,
            },
            ensure_ascii=False,
        )
        lines = canonical_lines + [duplicate_with_reordered_keys, "{bad json", '{"conversations":[]}', ""]
        tokenizer = self.tokenizer_for(records)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.run_prepare(root / "first", lines, tokenizer)
            second = self.run_prepare(root / "second", list(reversed(lines)), tokenizer)

            report_a, _, train_a, val_a, smoke_a, report_path_a = first
            report_b, _, train_b, val_b, smoke_b, _ = second
            train_lines = self.output_lines(train_a)
            val_lines = self.output_lines(val_a)
            smoke_lines = self.output_lines(smoke_a)

            # Stable priorities make both split assignment and capped sampling
            # independent of source order, not merely repeatable for one file.
            self.assertEqual(train_a.read_bytes(), train_b.read_bytes())
            self.assertEqual(val_a.read_bytes(), val_b.read_bytes())
            self.assertEqual(smoke_a.read_bytes(), smoke_b.read_bytes())
            self.assertEqual(report_a["train"]["sha256"], report_b["train"]["sha256"])
            self.assertEqual(report_a["validation"]["sha256"], report_b["validation"]["sha256"])

            self.assertEqual(len(train_lines), 20)
            self.assertEqual(len(val_lines), 20)
            self.assertEqual(len(set(train_lines)), 20)
            self.assertEqual(len(set(val_lines)), 20)
            self.assertTrue(set(train_lines).isdisjoint(val_lines))
            self.assertEqual(smoke_lines, train_lines[:5])
            self.assertTrue(set(smoke_lines).issubset(train_lines))

            threshold = int(0.5 * 2**64)
            for line in train_lines:
                fingerprint = hashlib.sha256(line.encode("utf-8")).digest()
                self.assertGreaterEqual(stable_u64(42, "split", fingerprint), threshold)
            for line in val_lines:
                fingerprint = hashlib.sha256(line.encode("utf-8")).digest()
                self.assertLess(stable_u64(42, "split", fingerprint), threshold)

            self.assertEqual(report_a["raw_lines"], len(lines))
            self.assertEqual(report_a["nonempty_lines"], len(lines) - 1)
            self.assertEqual(report_a["invalid_records"], 2)
            self.assertEqual(report_a["invalid_reasons"]["malformed_json"], 1)
            self.assertEqual(
                report_a["invalid_reasons"]["conversations must be a non-empty list"], 1
            )
            self.assertEqual(report_a["unique_records"], 400)
            self.assertEqual(report_a["exact_duplicates_removed"], 1)
            self.assertEqual(
                report_a["nonempty_lines"],
                report_a["invalid_records"]
                + report_a["unique_records"]
                + report_a["exact_duplicates_removed"],
            )
            self.assertEqual(
                report_a["unique_records"],
                report_a["eligible_train_records"] + report_a["eligible_val_records"],
            )
            self.assertEqual(report_a["train"]["records"], 20)
            self.assertEqual(report_a["validation"]["records"], 20)
            self.assertEqual(report_a["smoke"]["records"], 5)
            self.assertTrue(report_a["smoke"]["is_subset_of_train"])
            self.assertEqual(
                hashlib.sha256(train_a.read_bytes()).hexdigest(), report_a["train"]["sha256"]
            )
            self.assertEqual(
                hashlib.sha256(val_a.read_bytes()).hexdigest(),
                report_a["validation"]["sha256"],
            )
            self.assertEqual(json.loads(report_path_a.read_text(encoding="utf-8")), report_a)

    def test_validation_skips_candidate_with_no_target_after_truncation(self):
        too_long = {
            "conversations": [
                {"role": "user", "content": "很长的问题" * 20},
                {"role": "assistant", "content": "答案"},
            ]
        }
        valid = {
            "conversations": [
                {"role": "user", "content": "问"},
                {"role": "assistant", "content": "答"},
            ]
        }
        tokenizer = self.tokenizer_for(too_long, valid)
        too_long_canonical = canonical_record(too_long)
        valid_canonical = canonical_record(valid)
        candidates = [
            (
                0,
                hashlib.sha256(too_long_canonical.encode("utf-8")).digest(),
                too_long_canonical,
            ),
            (
                1,
                hashlib.sha256(valid_canonical.encode("utf-8")).digest(),
                valid_canonical,
            ),
        ]

        selected, stats = _validate_candidates(
            candidates,
            requested=1,
            tokenizer=tokenizer,
            max_length=20,
            seed=42,
            empty_think_ratio=0.0,
        )
        self.assertEqual(selected, [valid_canonical])
        self.assertEqual(stats["candidate_records_checked"], 2)
        self.assertEqual(
            stats["template_failures"]["no_supervised_target_after_truncation"], 1
        )
        self.assertEqual(stats["supervised_tokens_after_truncation"]["mean"], 1.0)

    def test_any_existing_output_aborts_without_partial_writes(self):
        records = self.records(100)
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        tokenizer = self.tokenizer_for(records)

        for existing_name in ("train.jsonl", "val.jsonl", "smoke.jsonl", "report.json"):
            with self.subTest(existing_name=existing_name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                input_path = directory / "source.jsonl"
                self.write_source(input_path, lines)
                outputs = {
                    name: directory / name
                    for name in ("train.jsonl", "val.jsonl", "smoke.jsonl", "report.json")
                }
                outputs[existing_name].write_text("sentinel", encoding="utf-8")

                with self.assertRaises(FileExistsError):
                    prepare_sft_subsets(
                        input_path=input_path,
                        tokenizer=tokenizer,
                        train_output=outputs["train.jsonl"],
                        val_output=outputs["val.jsonl"],
                        smoke_output=outputs["smoke.jsonl"],
                        report_output=outputs["report.json"],
                        train_size=5,
                        val_size=5,
                        smoke_size=2,
                        max_length=128,
                        val_ratio=0.5,
                        reserve_ratio=1.0,
                        empty_think_ratio=0.0,
                        seed=42,
                        progress_every=0,
                        overwrite=False,
                    )

                self.assertEqual(outputs[existing_name].read_text(encoding="utf-8"), "sentinel")
                for name, path in outputs.items():
                    if name != existing_name:
                        self.assertFalse(path.exists(), f"partial output was created: {name}")


if __name__ == "__main__":
    unittest.main()
