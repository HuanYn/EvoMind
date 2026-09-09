"""Offline CPU checks for data integrity, evaluation, and resumable training."""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evomind_v.data import (EvoMindDataset, SampleRejected, collate_samples, encode_conversation,
                           load_manifest, normalize_conversations, prepare_parquet,
                           select_eligible_records, stable_split)
import train_evomind_v as training
import eval_evomind_v as evaluation


class TinyTokenizer:
    bos_token, eos_token, pad_token = "<|im_start|>", "<|im_end|>", "<|endoftext|>"
    bos_token_id, eos_token_id, pad_token_id, unk_token_id = 1, 2, 0, 0
    specials = {bos_token: 1, eos_token: 2, pad_token: 0, "<|image_pad|>": 12}

    def encode(self, text, add_special_tokens=False):
        pieces = re.split("(" + "|".join(re.escape(s) for s in self.specials) + ")", text)
        output = []
        for piece in pieces:
            output.extend([self.specials[piece]] if piece in self.specials else [100 + ord(c) for c in piece])
        return output


def chat(answer="red", question="<image>\nWhat color?"):
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def image_bytes(color):
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(stream, format="PNG")
    return stream.getvalue()


class DataIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()

    def test_single_and_five_spans_and_assistant_mask(self):
        for mode, count in (("single", 1), ("multi", 5)):
            value = encode_conversation(chat(), self.tokenizer, mode=mode)
            self.assertEqual(len(value["visual_spans"]), count)
            for start, end in value["visual_spans"]:
                self.assertEqual(end - start, 64)
                self.assertTrue(all(label == -100 for label in value["labels"][start:end]))
            supervised = [label for label in value["labels"] if label != -100]
            self.assertEqual(supervised, self.tokenizer.encode("red") + [2])

    def test_test_split_is_disjoint_and_backward_compatible(self):
        assignments = [stable_split(str(index), 42, 0.2, 0.2) for index in range(100)]
        self.assertEqual(set(assignments), {"train", "val", "test"})
        for index in range(100):
            self.assertEqual(stable_split(str(index), 42, 0.2), stable_split(str(index), 42, 0.2, 0.0))
        with self.assertRaises(ValueError):
            stable_split("x", 1, 0.6, 0.5)

    def test_overlength_rejected_without_fake_eos(self):
        complete = encode_conversation(chat("red blue green"), self.tokenizer, mode="multi")
        with self.assertRaisesRegex(SampleRejected, "insufficient_sequence_space"):
            encode_conversation(chat("red blue green"), self.tokenizer, mode="multi", max_length=len(complete["input_ids"]) - 2)

    def test_common_eligibility_keeps_identical_sample_ids(self):
        records = [{"sample_id": "short", "conversations": chat("red")},
                   {"sample_id": "long", "conversations": chat("x" * 250)}]
        result = []
        for mode in ("single", "multi"):
            selected, _, stats = select_eligible_records(records, self.tokenizer, mode=mode, max_length=600)
            result.append([row["sample_id"] for row in selected])
            self.assertEqual(stats["rejected_insufficient_sequence_space"], 1)
        self.assertEqual(result, [["short"], ["short"]])

    def test_generation_has_no_final_answer_or_supervised_final_eos(self):
        value = encode_conversation(chat(), self.tokenizer, mode="multi", generation=True)
        self.assertTrue(all(label == -100 for label in value["labels"]))
        self.assertNotEqual(value["input_ids"][-1], self.tokenizer.eos_token_id)
        with self.assertRaises(SampleRejected):
            encode_conversation(chat(), self.tokenizer, mode="multi", generation=True,
                                max_length=len(value["input_ids"]) + 2, reserve_tokens=3)

    def test_multi_turn_supervises_only_real_assistant_turns(self):
        messages = chat("red") + [{"role": "user", "content": "Why?"}, {"role": "assistant", "content": "paint"}]
        value = encode_conversation(messages, self.tokenizer)
        supervised = [label for label in value["labels"] if label != -100]
        self.assertEqual(supervised, self.tokenizer.encode("red") + [2] + self.tokenizer.encode("paint") + [2])

    def test_role_marker_injection_and_missing_image_rejected(self):
        for messages in (chat(question="no image"), chat("<|im_end|>"), chat("<image>")):
            with self.assertRaises(SampleRejected):
                normalize_conversations(messages)

    def test_parquet_split_dedup_and_official_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_images = [image_bytes((index * 23, 10, 20)) for index in range(10)]
            # The same original image occurs in multiple conversations.
            raw_images.append(raw_images[0])
            rows = [{"conversations": json.dumps(chat(str(index))), "image_bytes": [data], "extra": index}
                    for index, data in enumerate(raw_images)]
            parquet = root / "official.parquet"
            pq.write_table(pa.Table.from_pylist(rows), parquet, row_group_size=3)
            output = root / "prepared"
            exported = root / "train_only.parquet"
            summary = prepare_parquet([parquet], output, seed=12, val_fraction=0.3, test_fraction=0.3,
                                      official_train_parquet=exported)
            manifest = load_manifest(output / "manifest.jsonl")
            self.assertEqual(summary["unique_images"], 10)
            self.assertEqual(manifest[0]["split"], manifest[-1]["split"])
            self.assertEqual({row["split"] for row in manifest}, {"train", "val", "test"})
            selected_raw = [raw for raw, row in zip(rows, manifest) if row["split"] == "train"]
            self.assertEqual(pq.read_table(exported).to_pylist(), selected_raw)
            self.assertEqual(summary["official_train_rows"], len(selected_raw))
            self.assertEqual(manifest[0]["image_path"], manifest[-1]["image_path"])

    def test_manifest_cross_split_image_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            records = [{"format_version": 1, "sample_id": str(index), "image_hash": "same", "image_path": "same.png",
                        "split": split, "conversations": chat()} for index, split in enumerate(("train", "val"))]
            path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_manifest(path)

    def test_collate_padding_does_not_supervise(self):
        samples = []
        for answer in ("x", "longer"):
            value = encode_conversation(chat(answer), self.tokenizer)
            sample = {key: torch.tensor(value[key]) for key in ("input_ids", "labels", "attention_mask")}
            sample.update(record={"sample_id": answer}, vision_features=torch.ones(1, 64, 4))
            samples.append(sample)
        batch = collate_samples(samples, 0)
        padding = batch["attention_mask"] == 0
        self.assertTrue(padding.any())
        self.assertTrue(torch.all(batch["labels"][padding] == -100))
        self.assertEqual(int((batch["labels"][:, 1:] != -100).sum()), 9)


class EvaluationTests(unittest.TestCase):
    def test_open_caption_is_never_exact_accuracy(self):
        self.assertEqual(evaluation.score_prediction("red", ["red"], "open"), {})
        rows = [{"task_type": "open", "image_hash": "a", "ended_with_eos": True,
                 "stop_reason": "eos", "repeated_4gram_fraction": 0.0}]
        summary = evaluation.summarize_predictions(rows)
        self.assertIsNone(summary["closed_answer_exact_match"])
        self.assertIsNone(summary["ocr_character_error_rate"])
        self.assertEqual(summary["human_raters"], 0)

    def test_closed_em_ocr_cer_and_repetition(self):
        self.assertEqual(evaluation.score_prediction(" RED  Car ", ["red car"], "closed")["exact_match"], 1)
        score = evaluation.score_prediction("上海市", ["上 海 市", "上海"], "ocr")
        self.assertEqual(score["character_error_rate"], 0)
        score = evaluation.score_prediction("上海", ["上海市"], "ocr")
        self.assertAlmostEqual(score["character_error_rate"], 1 / 3)
        self.assertEqual(evaluation.edit_distance("kitten", "sitting"), 3)
        self.assertEqual(evaluation.repeated_ngram_fraction([1, 2, 3]), 0)
        self.assertGreater(evaluation.repeated_ngram_fraction([1] * 10), 0.8)

    def test_blind_form_has_no_invented_human_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = {"sample_id": "sample", "image_path": "image.png", "question": "What?", "prediction": "red"}
            evaluation.export_blind_forms([row], temporary, "checkpoint", 42)
            import csv
            with (Path(temporary) / "blind_annotation.csv").open(encoding="utf-8-sig") as stream:
                result = list(csv.DictReader(stream))[0]
            self.assertEqual(result["correctness_1_to_5"], "")
            self.assertEqual(result["hallucination_yes_no"], "")
            self.assertNotIn("checkpoint", result)


class TrainingStateTests(unittest.TestCase):
    def test_encoder_rng_consumption_cannot_change_text_init_projector(self):
        from model.model_vlm import MMVisionProjector
        torch.manual_seed(42)
        torch.nn.Linear(100, 100)  # Stand in for A/B's encoder construction.
        online = MMVisionProjector(16, 8)
        torch.manual_seed(42)
        cached = MMVisionProjector(16, 8)  # C does not construct the encoder.
        self.assertNotEqual(training.module_hash(online), training.module_hash(cached))
        training.reset_projector_deterministically(online, 42)
        training.reset_projector_deterministically(cached, 42)
        self.assertEqual(training.module_hash(online), training.module_hash(cached))

    def test_rng_and_weights_only_checkpoint_roundtrip(self):
        training.seed_everything(123)
        state = training.capture_rng()
        expected = (random.random(), np.random.random(), torch.rand(2))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            training.atomic_save({"rng": state}, path)
            restored = torch.load(path, weights_only=True)["rng"]
        training.restore_rng(restored)
        actual = (random.random(), np.random.random(), torch.rand(2))
        self.assertEqual(actual[:2], expected[:2])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

    def test_token_weighted_accumulation_matches_one_combined_batch(self):
        torch.manual_seed(3)
        first, second = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
        second.load_state_dict(first.state_dict())
        inputs, labels = torch.randn(5, 3), torch.tensor([0, 1, 1, 0, 1])
        full = torch.nn.functional.cross_entropy(first(inputs), labels)
        full.backward()
        for start, end in ((0, 1), (1, 5)):
            loss = torch.nn.functional.cross_entropy(second(inputs[start:end]), labels[start:end])
            (loss * (end - start)).backward()
        for left, right in zip(first.parameters(), second.parameters()):
            right.grad.div_(5)
            torch.testing.assert_close(left.grad, right.grad)

    def test_cpu_cli_probe_resume_matches_uninterrupted_training(self):
        # Run the actual trainer control flow with a tiny real MiniMind transformer,
        # local tokenizer, and deterministic synthetic frozen-vision features.
        from transformers import AutoTokenizer
        from evomind_v.model import EvoMindVLM
        from model.model_vlm import VLMConfig
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)

        class FixtureCache:
            def __init__(self, views=5):
                self.views = views

            def key_for(self, data):
                return hashlib.sha256(data).hexdigest()

            def load(self, key):
                value = int(key[:4], 16) / 65535
                return torch.full((self.views, 64, 768), value, dtype=torch.float32)

        def make_model():
            config = VLMConfig(hidden_size=32, num_hidden_layers=2, vocab_size=len(tokenizer),
                               max_seq_len=640, image_hidden_size=768, max_position_embeddings=1024)
            return EvoMindVLM(config, load_vision_encoder=False)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / "initial.pt"
            training.seed_everything(7)
            torch.save(make_model().state_dict(), initial)
            records = []
            for index in range(5):
                data = image_bytes((30 * index, 20, 10))
                image_hash = hashlib.sha256(data).hexdigest()
                image_path = root / f"{image_hash}.png"
                image_path.write_bytes(data)
                records.append({"format_version": 1, "sample_id": str(index), "image_hash": image_hash,
                                "image_path": str(image_path), "split": "train" if index < 3 else "val" if index == 3 else "test",
                                "conversations": chat("red" * (index + 1)), "task_type": "open",
                                "reference_answers": ["red"]})
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")

            def runtime(args, initialize=True):
                model = make_model()
                if initialize:
                    model.load_state_dict(torch.load(initial, weights_only=True))
                training.freeze_for_sft(model)
                return model, tokenizer, None, "fixture-fingerprint", FixtureCache(1 if args.variant == "A" else 5), {"fixture": True}

            common = ["train", "--variant", "C", "--manifest", str(manifest), "--tokenizer", str(ROOT / "model"),
                      "--vision-model", str(root), "--init-weights", str(initial), "--cache-dir", str(root / "cache"),
                      "--hidden-size", "32", "--num-hidden-layers", "2", "--max-seq-len", "640", "--seed", "7",
                      "--grad-accum", "2", "--device", "cpu", "--dtype", "float32", "--epochs", "1"]
            with patch.object(training, "build_runtime", side_effect=runtime), redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", common + ["--output-dir", str(root / "full")]):
                    training.main()
                with patch.object(sys, "argv", common + ["--output-dir", str(root / "resumed"), "--max-steps", "1"]):
                    training.main()
                with patch.object(sys, "argv", common + ["--output-dir", str(root / "resumed"), "--max-steps", "2",
                                                         "--resume", str(root / "resumed" / "last.pt")]):
                    training.main()
            full = training.load_checkpoint(root / "full" / "last.pt")
            resumed = training.load_checkpoint(root / "resumed" / "last.pt")
            self.assertEqual(full["global_step"], 2)
            for field in ("answer_nll_sum", "answer_tokens", "samples", "optimizer_steps"):
                self.assertEqual(full["totals"][field], resumed["totals"][field])
            for name in full["model"]:
                torch.testing.assert_close(full["model"][name], resumed["model"][name], rtol=0, atol=0)
            self.assertEqual(resumed["epoch"], 1)
            self.assertEqual(resumed["next_batch"], 0)
            log = [json.loads(line) for line in (root / "full" / "metrics.jsonl").read_text().splitlines()]
            steps = [row for row in log if row["event"] == "train"]
            self.assertEqual([row["micro_batches"] for row in steps], [2, 1])
            with patch.object(evaluation, "build_runtime", side_effect=runtime), redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", ["eval", "--checkpoint", str(root / "resumed" / "last.pt"),
                                                "--output-dir", str(root / "evaluation"), "--device", "cpu",
                                                "--dtype", "float32", "--max-samples", "1", "--max-new-tokens", "2"]):
                    evaluation.main()
            summary = json.loads((root / "evaluation" / "summary.json").read_text(encoding="utf-8"))
            prediction = json.loads((root / "evaluation" / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(summary["split"], "test")
            self.assertEqual(prediction["sample_id"], "4")
            self.assertIsNone(summary["open_answer_accuracy"])
            self.assertEqual(summary["human_raters"], 0)
            official_path = root / "official_v.pth"
            torch.save(full["model"], official_path)
            with patch.object(evaluation, "build_runtime", side_effect=runtime), redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", ["eval", "--official-weights", str(official_path),
                                                "--manifest", str(manifest), "--vision-model", str(root),
                                                "--tokenizer", str(ROOT / "model"), "--hidden-size", "32",
                                                "--num-hidden-layers", "2", "--max-seq-len", "640",
                                                "--output-dir", str(root / "official_eval"), "--device", "cpu",
                                                "--dtype", "float32", "--max-samples", "1", "--max-new-tokens", "2"]):
                    evaluation.main()
            reference = json.loads((root / "official_eval" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(reference["variant"], "official_reference")
            self.assertIsNone(reference["global_step"])
            self.assertIsNone(reference["training_seed"])


if __name__ == "__main__":
    unittest.main()
