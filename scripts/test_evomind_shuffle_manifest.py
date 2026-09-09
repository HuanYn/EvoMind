"""Small standard-library-only fixtures; no GPU, ML imports, or full-data reads."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evomind_shuffle_manifest as shuffle

ROOT = Path(__file__).resolve().parents[1]


class ShuffleTests(unittest.TestCase):
    def setUp(self):
        # Keep all temporary outputs on this project's E-drive workspace.
        scratch = ROOT / "artifacts" / "test_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="shuffle-fixture-", dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source_dir = self.root / "source"
        (self.source_dir / "images").mkdir(parents=True)
        self.source = self.source_dir / "manifest.jsonl"
        self.records = []
        for index in range(48):
            # The fixture deliberately puts English before Chinese, with one
            # image represented by two conversations in the same split.
            image_index = index // 2
            contents = f"fixture-image-{image_index}".encode()
            image_path = self.source_dir / "images" / f"{image_index}.jpg"
            image_path.write_bytes(contents)
            split = ("train", "val", "test")[image_index % 3]
            self.records.append({"format_version": 1, "sample_id": f"sample-{index:03d}",
                                 "image_hash": hashlib.sha256(contents).hexdigest(),
                                 "image_path": image_path.relative_to(self.source_dir).as_posix(), "split": split,
                                 "conversations": [{"role": "user", "content": "<image> What is shown?" if index < 24 else "<image> 图中是什么？"},
                                                   {"role": "assistant", "content": f"A sample {index}" if index < 24 else f"示例 {index}"}],
                                 "reference_answers": [str(index)], "task_type": "open", "extra": {"keep": index}})
        self.write_source(self.records)

    def write_source(self, records, path=None):
        path = path or self.source
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")

    def run_shuffle(self, output="controlled", **kwargs):
        with redirect_stdout(io.StringIO()):
            return shuffle.shuffle_manifest(self.source, self.root / output, sqlite_cache_mib=1,
                                            commit_every=5, progress_every=0, **kwargs)

    def output_rows(self, output="controlled"):
        return [json.loads(line) for line in (self.root / output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_preserves_rows_splits_image_paths_and_records_hashes(self):
        source_hash = shuffle.sha256_file(self.source)
        summary = self.run_shuffle(seed=42)
        output = self.output_rows()
        by_id = {row["sample_id"]: row for row in self.records}
        self.assertEqual(len(output), len(self.records))
        self.assertEqual(summary["records"], 48)
        self.assertEqual(summary["unique_images"], 24)
        self.assertEqual(summary["split_counts"], {"test": 16, "train": 16, "val": 16})
        for row in output:
            original = by_id[row["sample_id"]]
            self.assertEqual({k: v for k, v in row.items() if k != "image_path"},
                             {k: v for k, v in original.items() if k != "image_path"})
            expected = (self.source_dir / original["image_path"]).resolve()
            self.assertEqual((self.root / "controlled" / row["image_path"]).resolve(), expected)
            self.assertFalse(Path(row["image_path"]).is_absolute())
        self.assertEqual(summary["source_manifest_sha256"], source_hash)
        self.assertEqual(summary["input_manifest_sha256"], source_hash)
        self.assertEqual(shuffle.sha256_file(self.source), source_hash)
        self.assertEqual(summary["manifest_sha256"], shuffle.sha256_file(self.root / "controlled" / "manifest.jsonl"))
        self.assertEqual(summary["output_manifest_sha256"], summary["manifest_sha256"])
        self.assertEqual(summary["counts"]["records"], 48)
        self.assertEqual(summary["status"], "complete")
        self.assertFalse(any("TEMP B-TREE" in line for line in summary["export_query_plan"]))
        self.assertEqual({p.name for p in (self.root / "controlled").iterdir()}, {"manifest.jsonl", "summary.json"})

    def test_order_matches_seeded_hash_and_not_source_prefix(self):
        self.run_shuffle(seed=42)
        ids = [row["sample_id"] for row in self.output_rows()]
        expected = sorted((row["sample_id"] for row in self.records), key=lambda value: (shuffle.order_key(42, value), value))
        self.assertEqual(ids, expected)
        self.assertNotEqual(ids[:12], [row["sample_id"] for row in self.records[:12]])
        self.assertTrue(any(int(value.split("-")[1]) < 24 for value in ids[:12]))
        self.assertTrue(any(int(value.split("-")[1]) >= 24 for value in ids[:12]))

    def test_source_order_independence_seed_difference_and_verified_reuse(self):
        first = self.run_shuffle("one", seed=42)
        target = self.root / "one" / "manifest.jsonl"
        before = target.stat().st_mtime_ns
        again = self.run_shuffle("one", seed=42)
        self.assertEqual(first, again)
        self.assertEqual(target.stat().st_mtime_ns, before)
        self.write_source(list(reversed(self.records)))
        reversed_source = self.run_shuffle("two", seed=42)
        self.assertEqual(first["manifest_sha256"], reversed_source["manifest_sha256"])
        different_seed = self.run_shuffle("three", seed=123)
        self.assertNotEqual(first["manifest_sha256"], different_seed["manifest_sha256"])
        with self.assertRaisesRegex(ValueError, "Source manifest changed"):
            self.run_shuffle("one", seed=42)

    def test_duplicate_ids_and_cross_split_images_rejected_without_publication(self):
        self.write_source(self.records + [self.records[0]])
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            self.run_shuffle("duplicates")
        self.assertFalse((self.root / "duplicates").exists())
        bad = dict(self.records[1], split="test")
        self.write_source([self.records[0], bad])
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            self.run_shuffle("leakage")
        self.assertFalse((self.root / "leakage").exists())

    def test_hash_mismatch_changed_output_and_foreign_outputs_are_preserved(self):
        with self.assertRaisesRegex(ValueError, "expected-source"):
            self.run_shuffle("wrong_hash", expected_source_sha256="0" * 64)
        self.assertFalse((self.root / "wrong_hash").exists())
        self.run_shuffle()
        manifest = self.root / "controlled" / "manifest.jsonl"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "shuffled manifest hash"):
            self.run_shuffle()
        foreign = self.root / "foreign"
        foreign.mkdir()
        note = foreign / "user.txt"
        note.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_shuffle("foreign")
        self.assertEqual(note.read_text(), "preserve")

    def test_source_change_during_ingest_does_not_publish(self):
        original = shuffle._normalize_record
        touched = False

        def modifying(*args, **kwargs):
            nonlocal touched
            result = original(*args, **kwargs)
            if not touched:
                with self.source.open("ab") as stream:
                    stream.write(b"\n")
                touched = True
            return result

        with patch.object(shuffle, "_normalize_record", side_effect=modifying):
            with self.assertRaisesRegex(RuntimeError, "changed while reading"):
                self.run_shuffle("changing")
        self.assertFalse((self.root / "changing").exists())

    def test_line_size_limit_empty_input_and_argument_alias(self):
        with self.assertRaisesRegex(ValueError, "inside the evomind project"):
            shuffle.shuffle_manifest(self.source, ROOT.parent / "outside-shuffle-fixture")
        self.source.write_bytes(b" " * (1024 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, "max-line-mib"):
            self.run_shuffle("oversize", max_line_mib=1)
        self.source.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no records"):
            self.run_shuffle("empty")
        self.write_source(self.records)
        arguments = ["shuffle", "--input", str(self.source), "--output-dir", str(self.root / "cli"),
                     "--seed", "42", "--sqlite-cache-mib", "1", "--progress-every", "0"]
        with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
            shuffle.main()
        self.assertTrue((self.root / "cli" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
