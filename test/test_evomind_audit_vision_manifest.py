"""Tiny stdlib fixtures only: no real datasets, image decoding, or GPU imports."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evomind_audit_vision_manifest.py"
SPEC = importlib.util.spec_from_file_location("evomind_audit_vision_manifest", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class ManifestAuditTests(unittest.TestCase):
    def setUp(self):
        temporary_parent = SCRIPT.parents[1] / "artifacts" / "test_tmp"
        temporary_parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="vision_manifest_audit_", dir=temporary_parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "images").mkdir()
        source = self.root / "source.parquet"
        source.write_bytes(b"tiny provenance fixture, deliberately not decoded")
        self.rows = []
        for index, contents in enumerate((b"image-A", b"image-B", b"image-A")):
            image_hash = hashlib.sha256(contents).hexdigest()
            path = self.root / "images" / f"{image_hash}.png"
            path.write_bytes(contents)
            self.rows.append({"format_version": 1, "sample_id": f"0:{index}:{image_hash[:16]}",
                "image_hash": image_hash, "image_path": f"images/{image_hash}.png",
                "split": audit.stable_split(image_hash, 42, 0.2, 0.2), "source_index": 0,
                "source_row": index, "task_type": "open", "reference_answers": ["红色 red"],
                "conversations": [{"role": "user", "content": "<image> What?"},
                                  {"role": "assistant", "content": "红色 red"}]})
        self.summary = {"status": "complete", "format_version": 1, "seed": 42,
            "val_fraction": 0.2, "test_fraction": 0.2, "grouping": "sha256_original_image_bytes",
            "max_rows": None, "sources": [{"path": str(source), "total_rows": 3,
                "size_bytes": source.stat().st_size, "sha256": audit.file_hash(source)}],
            "counts": {"scanned": 3, "accepted": 3, "task_open": 3},
            "unique_images": 2, "unique_images_by_split": {}}
        unique = {}
        for row in self.rows:
            key = f"{row['split']}_records"
            self.summary["counts"][key] = self.summary["counts"].get(key, 0) + 1
            unique[row["image_hash"]] = row["split"]
        for split in unique.values():
            self.summary["unique_images_by_split"][split] = self.summary["unique_images_by_split"].get(split, 0) + 1

    def write_fixture(self):
        manifest = self.root / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows), encoding="utf-8")
        self.summary["manifest_sha256"] = audit.file_hash(manifest)
        (self.root / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")

    def run_audit(self, **kwargs):
        self.write_fixture()
        return audit.audit_manifest(self.root / "summary.json", self.root / "audit.json", **kwargs)

    def test_pass_bounded_sqlite_and_heuristics(self):
        report = self.run_audit(verify_image_samples=32, verify_source_hashes=True)
        self.assertEqual(report["status"], "pass", report)
        self.assertEqual(report["unique_sample_ids"], 3)
        self.assertEqual(report["image_hash_verification"]["verified_count"], 2)
        self.assertEqual(report["assistant_turn_count_distribution"], {"1": 3})
        self.assertEqual(report["character_script_heuristics"]["character_counts"]["han"], 6)
        self.assertFalse(report["scope"]["source_rows_redecoded_from_parquet"])
        self.assertEqual(list(self.root.glob("vision_manifest_audit_*")), [])

    def test_duplicate_id_detected(self):
        self.rows[2] = copy.deepcopy(self.rows[0])
        report = self.run_audit()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_counts"]["duplicate_sample_id"], 1)

    def test_cross_split_and_deterministic_assignment(self):
        self.rows[2]["split"] = "val" if self.rows[0]["split"] != "val" else "train"
        report = self.run_audit()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_counts"]["cross_split_image_leakage"], 1)
        self.assertEqual(report["error_counts"]["deterministic_split_mismatch"], 1)

    def test_source_row_bounds(self):
        self.rows[1]["source_row"] = 3
        report = self.run_audit()
        self.assertEqual(report["status"], "failed")
        self.assertIn("source_index/source_row", report["error_examples"][0]["detail"])

    def test_path_escape_and_hash_filename(self):
        for index, bad in enumerate(("../outside.png", "images/wrong.png", "C:/outside.png")):
            with self.subTest(path=bad):
                self.rows[0]["image_path"] = bad
                self.write_fixture()
                report = audit.audit_manifest(self.root / "summary.json", self.root / f"audit_{index}.json")
                self.assertEqual(report["status"], "failed")
                self.assertGreater(report["error_counts"]["invalid_record"], 0)

    def test_count_mismatch(self):
        self.summary["counts"]["accepted"] += 1
        report = self.run_audit()
        self.assertEqual(report["status"], "failed")
        self.assertIn("summary_count_mismatch", report["error_counts"])

    def test_manifest_hash_mismatch(self):
        self.write_fixture()
        with (self.root / "manifest.jsonl").open("ab") as stream:
            stream.write(b"\n")
        report = audit.audit_manifest(self.root / "summary.json", self.root / "audit.json")
        self.assertEqual(report["status"], "failed")
        self.assertIn("manifest_hash_mismatch", report["error_counts"])

    def test_default_does_not_read_image_bytes_and_opt_in_detects_corruption(self):
        image = self.root / self.rows[0]["image_path"]
        image.write_bytes(b"corrupt")
        self.assertEqual(self.run_audit()["status"], "pass")
        report = audit.audit_manifest(self.root / "summary.json", self.root / "sample_audit.json", verify_image_samples=1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_counts"]["sample_image_hash_mismatch"], 1)

    def test_source_hash_mode_and_no_output_overwrite(self):
        report = self.run_audit()
        self.assertFalse(report["source_provenance"][0]["content_hash_recomputed"])
        output = self.root / "audit.json"
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            audit.audit_manifest(self.root / "summary.json", output)
        self.assertEqual(output.read_bytes(), original)

    def test_cli_missing_summary_fails_with_report(self):
        code = audit.main(["--input-summary", str(self.root / "missing.json"), "--output", str(self.root / "audit.json")])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads((self.root / "audit.json").read_text())["status"], "failed")

    def test_optional_source_rehash_detects_same_size_change(self):
        source = Path(self.summary["sources"][0]["path"])
        source.write_bytes(b"x" * source.stat().st_size)
        report = self.run_audit(verify_source_hashes=True)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_counts"]["source_hash_mismatch"], 1)

    def test_incomplete_summary_refused_before_scanning_manifest(self):
        self.write_fixture()
        (self.root / "manifest.jsonl").unlink()
        self.summary["status"] = "running"
        (self.root / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
        report = audit.audit_manifest(self.root / "summary.json", self.root / "audit.json")
        self.assertEqual(report["status"], "failed")
        self.assertIn("status=complete", report["fatal_error"])

    def test_cli_success_status_protocol(self):
        self.write_fixture()
        code = audit.main(["--input-summary", str(self.root / "summary.json"),
                           "--output", str(self.root / "audit.json"), "--verify-image-samples", "32"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads((self.root / "audit.json").read_text(encoding="utf-8"))["status"], "pass")


if __name__ == "__main__":
    unittest.main()
