import json
import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_training_curves import (  # noqa: E402
    _smooth_finite_segments,
    plot_training_curves,
)
from model.training_history import (  # noqa: E402
    TrainingHistoryError,
    TrainingHistoryWriter,
    append_training_record,
    load_training_history,
    make_record,
    prepare_history_for_resume,
)


class TestTrainingHistory(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.history_path = self.root / "history.jsonl"

    @staticmethod
    def record(step, **metrics):
        return make_record(
            run_id="dense-run",
            stage="pretrain",
            architecture="dense",
            step=step,
            **metrics,
        )

    def test_non_finite_metrics_become_null_and_are_named(self):
        append_training_record(
            self.history_path,
            self.record(
                1,
                train_ce=float("nan"),
                val_ce=float("inf"),
                grad_norm=-float("inf"),
                router_max_load_by_layer=[0.25, float("nan")],
                intentionally_missing=None,
            ),
        )

        raw = self.history_path.read_text(encoding="utf-8")
        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)
        payload = json.loads(raw)
        self.assertIsNone(payload["train_ce"])
        self.assertIsNone(payload["val_ce"])
        self.assertIsNone(payload["grad_norm"])
        self.assertIsNone(payload["router_max_load_by_layer"][1])
        self.assertIsNone(payload["intentionally_missing"])
        self.assertEqual(
            payload["invalid_metrics"],
            [
                "grad_norm",
                "router_max_load_by_layer[1]",
                "train_ce",
                "val_ce",
            ],
        )
        self.assertNotIn("intentionally_missing", payload["invalid_metrics"])

    def test_unterminated_final_half_line_is_recovered(self):
        append_training_record(self.history_path, self.record(1, train_ce=4.0))
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write('{"schema_version":1,"step":2')

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            records = load_training_history(self.history_path)
        self.assertEqual([record["step"] for record in records], [1])
        self.assertTrue(any("truncated final line" in str(item.message) for item in caught))

    def test_malformed_middle_line_raises(self):
        valid_one = json.dumps(self.record(1), allow_nan=False)
        valid_two = json.dumps(self.record(2), allow_nan=False)
        self.history_path.write_text(
            f"{valid_one}\n{{not valid json}}\n{valid_two}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(TrainingHistoryError, r":2: malformed JSONL"):
            load_training_history(self.history_path)

    def test_complete_malformed_final_line_raises(self):
        valid = json.dumps(self.record(1), allow_nan=False)
        self.history_path.write_text(f"{valid}\n{{bad}}\n", encoding="utf-8")
        with self.assertRaises(TrainingHistoryError):
            load_training_history(self.history_path)

    def test_resume_trims_future_steps_and_last_duplicate_wins(self):
        append_training_record(self.history_path, self.record(1, train_ce=5.0))
        append_training_record(self.history_path, self.record(2, train_ce=4.5))
        append_training_record(self.history_path, self.record(2, train_ce=4.25))
        append_training_record(self.history_path, self.record(4, train_ce=3.5))

        repaired = prepare_history_for_resume(self.history_path, checkpoint_step=2)
        self.assertEqual([record["step"] for record in repaired], [1, 2])
        self.assertEqual(repaired[-1]["train_ce"], 4.25)
        reloaded = load_training_history(self.history_path)
        self.assertEqual(reloaded, repaired)
        self.assertTrue(self.history_path.read_bytes().endswith(b"\n"))

    def test_writer_enforces_identity_and_increasing_optimizer_steps(self):
        writer = TrainingHistoryWriter(
            self.history_path,
            run_id="dense-run",
            stage="pretrain",
            architecture="dense",
        )
        writer.append(1, train_ce=5.0)
        with self.assertRaisesRegex(TrainingHistoryError, "not newer"):
            writer.append(1, train_ce=4.0)

        with self.assertRaisesRegex(TrainingHistoryError, "non-empty history"):
            TrainingHistoryWriter(
                self.history_path,
                run_id="dense-run",
                stage="pretrain",
                architecture="dense",
            )

        with self.assertRaisesRegex(TrainingHistoryError, "identity"):
            TrainingHistoryWriter(
                self.history_path,
                run_id="different-run",
                stage="pretrain",
                architecture="dense",
                checkpoint_step=1,
            )

        replacement = TrainingHistoryWriter(
            self.history_path,
            run_id="replacement-run",
            stage="sft",
            architecture="moe",
            overwrite=True,
        )
        self.assertEqual(self.history_path.read_text(encoding="utf-8"), "")
        replacement.append(1, train_ce=3.0)
        self.assertEqual(load_training_history(self.history_path)[0]["run_id"], "replacement-run")

    def test_inconsistent_identity_inside_file_raises(self):
        append_training_record(self.history_path, self.record(1))
        append_training_record(
            self.history_path,
            make_record(
                run_id="other-run",
                stage="pretrain",
                architecture="dense",
                step=2,
            ),
        )
        with self.assertRaisesRegex(TrainingHistoryError, "changed within one file"):
            load_training_history(self.history_path)


class TestTrainingPlot(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_smoothing_resets_at_gaps_without_filling_them(self):
        values = [5.0, 3.0, math.nan, 9.0, 5.0]
        smoothed = _smooth_finite_segments(values, window=2)
        self.assertEqual(smoothed[:2], [5.0, 4.0])
        self.assertTrue(math.isnan(smoothed[2]))
        self.assertEqual(smoothed[3:], [9.0, 7.0])

    def test_dense_and_moe_comparison_creates_valid_atomic_png(self):
        dense_path = self.root / "dense.jsonl"
        moe_path = self.root / "moe.jsonl"
        output_path = self.root / "plots" / "comparison.png"

        for step, train_ce, val_ce in ((1, 5.0, 5.2), (2, 4.5, 4.7), (3, 4.2, 4.4)):
            append_training_record(
                dense_path,
                make_record(
                    run_id="dense",
                    stage="pretrain",
                    architecture="dense",
                    step=step,
                    train_ce=train_ce,
                    val_ce=val_ce,
                    val_ppl=math.exp(val_ce),
                    lr=3e-4 / step,
                    grad_norm=1.0 + step / 10,
                ),
            )
            append_training_record(
                moe_path,
                make_record(
                    run_id="moe",
                    stage="pretrain",
                    architecture="moe",
                    step=step,
                    train_ce=train_ce - 0.1,
                    train_total_loss=train_ce - 0.08,
                    val_ce=val_ce - 0.1,
                    val_ppl=math.exp(val_ce - 0.1),
                    lr=3e-4 / step,
                    grad_norm=None if step == 2 else 1.2,
                    num_experts=4,
                    router_aux_raw=8.0,
                    router_aux_weighted=0.08,
                    router_max_load_by_layer=[0.25, 0.27 + step / 100],
                ),
            )

        result = plot_training_curves(
            [dense_path, moe_path],
            output_path,
            title="Dense vs MoE",
            dpi=80,
            smoothing_window=2,
        )
        self.assertEqual(result, output_path)
        self.assertTrue(output_path.is_file())
        self.assertGreater(output_path.stat().st_size, 1_000)

        from PIL import Image

        with Image.open(output_path) as image:
            self.assertEqual(image.format, "PNG")
            self.assertGreater(image.width, image.height)

        leftovers = list(output_path.parent.glob("*.tmp.png"))
        self.assertEqual(leftovers, [])

    def test_sparse_metrics_and_dense_only_history_do_not_crash(self):
        dense_path = self.root / "sparse.jsonl"
        output_path = self.root / "sparse.png"
        append_training_record(
            dense_path,
            make_record(
                run_id="sparse",
                stage="sft",
                architecture="dense",
                step=1,
                train_ce=None,
                val_ppl=float("nan"),
                lr=None,
            ),
        )
        plot_training_curves([dense_path], output_path, dpi=60)
        self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
