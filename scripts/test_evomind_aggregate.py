"""Synthetic CPU-only aggregate tests; fixtures are not reported research results."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import evomind_aggregate as report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def fixture(root, seeds=(42, 123, 2026)):
    artifacts = root / "vision" / "artifacts"
    rows = []
    for index in range(3):
        content = f"fixture-image-{index}".encode()
        image = root / f"image{index}.png"
        image.write_bytes(content)
        context = [{"role": "user", "content": f"<image> Question {index}"}]
        rows.append({"sample_id": str(index), "image_hash": hashlib.sha256(content).hexdigest(), "image_path": str(image),
                     "question": context[0]["content"], "conversation_context": context, "prediction": f"answer {index}",
                     "generated_token_ids": [index + 10, 2], "ended_with_eos": True, "repeated_4gram_fraction": 0.0})
    for seed_index, seed in enumerate(seeds):
        for variant in report.VARIANTS:
            name = f"{variant}_seed{seed}"
            arguments = {"variant": variant, "seed": seed, "epochs": 2, "batch_size": 1, "grad_accum": 4,
                         "learning_rate": 5e-6, "weight_decay": 0.01, "grad_clip": 1.0, "dtype": "bfloat16",
                         "max_seq_len": 768, "hidden_size": 768, "num_hidden_layers": 8, "allow_text_init": True,
                         "cache_dtype": "float32"}
            config = {key: key + "-same" for key in report.CONTROL_HASHES}
            config.update(arguments=arguments, initial_projector_sha256=f"projector-{seed}", planned_optimizer_steps=20,
                          train_records=40, val_records=64)
            directory = artifacts / "training" / name
            write_json(directory / "run_config.json", config)
            runtime = (90 if variant == "C" else 100) + seed_index
            events = [{"event": "train", "global_step": 20, "answer_nll": 1.5, "learning_rate": 5e-7,
                       "peak_allocated_bytes": 1000000},
                      {"event": "validation", "global_step": 20, "answer_nll": 1.6},
                      {"event": "complete", "global_step": 20, "wall_seconds_this_process": runtime,
                       "totals": {"samples": 80, "optimizer_steps": 20}}]
            write_jsonl(directory / "metrics.jsonl", events)
            summary = {"variant": variant, "training_seed": seed, "split": "test", "samples": 3,
                       "heldout_sample_ids_sha256": report.sample_ids_sha(rows), "max_new_tokens": 128,
                       "decoding": "greedy", "global_step": 20, "eos_rate": 1.0,
                       "mean_repeated_4gram_fraction": 0.0, "mean_generation_seconds": 0.1,
                       "wall_seconds": 3.0, "peak_allocated_bytes": 1000000,
                       "inference_dtype": "bfloat16", "max_seq_len": 768, "prompt_format": "evomind_labelled_views",
                       "manifest_sha256": config["manifest_sha256"], "vision_fingerprint": config["vision_fingerprint"]}
            evaluation = artifacts / "evaluation" / name
            write_json(evaluation / "summary.json", summary)
            write_jsonl(evaluation / "predictions.jsonl", rows)
    official = dict(summary, variant="official_reference", training_seed=None, global_step=None,
                    prompt_format="official_native_single")
    write_json(artifacts / "evaluation" / "official" / "summary.json", official)
    write_jsonl(artifacts / "evaluation" / "official" / "predictions.jsonl", rows)
    write_json(artifacts / "cache_prepare" / "cache_summary.json",
               {"event": "cache_complete", "created": 3, "existing": 0, "unique_images": 3, "elapsed_seconds": 15.0,
                "metadata": {"fingerprint": config["vision_fingerprint"], "dtype": "float32", "mode": "multi"}})
    return artifacts


class AggregateTests(unittest.TestCase):
    def test_sample_std_and_missing_replications(self):
        statistics = report.mean_std([1, 2, 3])
        self.assertEqual(statistics["mean"], 2)
        self.assertEqual(statistics["sample_std"], 1)
        self.assertEqual(statistics["status"], "complete")
        self.assertIsNone(report.mean_std([1])["sample_std"])
        self.assertEqual(report.mean_std([1])["status"], "pending")

    def test_empty_state_is_pending_without_fabricated_curves(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, path = report.aggregate(temporary, expected_seeds=[42, 123, 2026])
            self.assertEqual(result["status"], "pending")
            self.assertFalse(result["controlled_comparison_ready"])
            self.assertEqual(result["plots"]["files"], [])
            self.assertIsNone(result["variant_statistics"]["A"]["evaluation"]["eos_rate"]["mean"])
            self.assertTrue(path.exists())

    def test_complete_fixture_metrics_cache_blind_and_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            result, _ = report.aggregate(root, expected_seeds=[42, 123, 2026], expected_test_samples=3, image_count=2)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["variant_statistics"]["B"]["train_runtime_seconds"]["mean"], 101)
            self.assertEqual(result["variant_statistics"]["B"]["train_runtime_seconds"]["sample_std"], 1)
            self.assertEqual(result["B_C_single_seed"]["text_differences"], 0)
            self.assertFalse("proof" in result["B_C_single_seed"])
            costs = result["cache_amortization"]["comparisons"]
            self.assertEqual(costs[0]["C_end_to_end_total_seconds"], 105)
            self.assertEqual(costs[1]["C_end_to_end_total_seconds"], 288)
            self.assertEqual(len(result["plots"]["files"]), 3)
            package = Path(result["blind_evaluation"]["package_dir"])
            key = Path(result["blind_evaluation"]["key_path"])
            self.assertNotEqual(key.parent, package)
            with (package / "annotation.csv").open(encoding="utf-8-sig") as stream:
                responses = list(csv.DictReader(stream))
            self.assertEqual(len(responses), 20)  # Two images times nine controlled runs plus reference.
            self.assertEqual(len({row["response_id"] for row in responses}), 20)
            self.assertTrue(all(not row["correctness_1_to_5"] and not row["hallucination_yes_no"] for row in responses))
            self.assertFalse(any(key in responses[0] for key in ("variant", "seed", "run_id", "sample_id")))
            original_csv = (package / "annotation.csv").read_bytes()
            report.aggregate(root, expected_seeds=[42, 123, 2026], expected_test_samples=3, image_count=2)
            self.assertEqual((package / "annotation.csv").read_bytes(), original_csv)

    def test_initialization_or_budget_mismatch_blocks_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = fixture(Path(temporary))
            config_path = artifacts / "training" / "C_seed42" / "run_config.json"
            config = json.loads(config_path.read_text())
            config["initial_projector_sha256"] = "wrong-projector"
            write_json(config_path, config)
            warnings = []
            training = report.discover_training(artifacts / "training", warnings)
            evaluations = report.discover_evaluations(artifacts / "evaluation", warnings, expected_samples=3)
            check = report.compare_controls(training, evaluations, 42)
            self.assertFalse(check["comparable"])
            self.assertEqual(check["status"], "incompatible")
            self.assertFalse(report.compare_bc_predictions(training, evaluations, 42)["comparison_performed"])

    def test_incomplete_predictions_are_not_hidden_by_inner_join(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = fixture(Path(temporary))
            path = artifacts / "evaluation" / "C_seed42" / "predictions.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()][:-1]
            write_jsonl(path, rows)
            warnings = []
            training = report.discover_training(artifacts / "training", warnings)
            evaluations = report.discover_evaluations(artifacts / "evaluation", warnings, expected_samples=3)
            result = report.compare_bc_predictions(training, evaluations)
            self.assertFalse(result["comparison_performed"])
            self.assertEqual(result["only_in_B"], ["2"])


if __name__ == "__main__":
    unittest.main()
