import json
import tempfile
import unittest
from pathlib import Path

import torch

from model.chat_template import IGNORE_INDEX
from model.dpo_data import DPODataset
from model.tokenizer import CharTokenizer


class TestDPOData(unittest.TestCase):
    @staticmethod
    def write_jsonl(directory: str, records) -> Path:
        path = Path(directory) / "dpo.jsonl"
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def tokenizer_for(*values) -> CharTokenizer:
        template_text = "system\nuser\nassistant\n<think></think>"
        return CharTokenizer.build(template_text + json.dumps(values, ensure_ascii=False))

    def test_pair_is_right_padded_and_only_answers_are_labeled(self):
        chosen = [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "好答案"}]
        rejected = [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "差答案"}]
        tokenizer = self.tokenizer_for(chosen, rejected)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [{"chosen": chosen, "rejected": rejected}])
            item = DPODataset(path, tokenizer, max_length=96)[0]

        self.assertEqual(set(item), {
            "chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"
        })
        for value in item.values():
            self.assertEqual(value.shape, (96,))
            self.assertEqual(value.dtype, torch.long)
        self.assertTrue(torch.any(item["chosen_labels"][1:] != IGNORE_INDEX))
        self.assertTrue(torch.any(item["rejected_labels"][1:] != IGNORE_INDEX))
        chosen_pad = next(i for i, token in enumerate(item["chosen_input_ids"].tolist()) if token == tokenizer.pad_id)
        self.assertTrue(torch.all(item["chosen_labels"][chosen_pad:] == IGNORE_INDEX))

    def test_rejects_mismatched_prefix_and_identical_answer(self):
        chosen = [{"role": "user", "content": "问题 A"}, {"role": "assistant", "content": "答案"}]
        mismatch = [{"role": "user", "content": "问题 B"}, {"role": "assistant", "content": "答案2"}]
        same = [{"role": "user", "content": "问题 A"}, {"role": "assistant", "content": "答案"}]
        tokenizer = self.tokenizer_for(chosen, mismatch)
        with tempfile.TemporaryDirectory() as directory:
            bad_prefix = self.write_jsonl(directory, [{"chosen": chosen, "rejected": mismatch}])
            with self.assertRaisesRegex(ValueError, "mismatched prompt prefixes"):
                DPODataset(bad_prefix, tokenizer)
            identical = self.write_jsonl(directory, [{"chosen": chosen, "rejected": same}])
            with self.assertRaisesRegex(ValueError, "identical answers"):
                DPODataset(identical, tokenizer)

    def test_rejects_branch_that_would_be_truncated(self):
        chosen = [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "这是较好的长答案"}]
        rejected = [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "这是较差的长答案"}]
        tokenizer = self.tokenizer_for(chosen, rejected)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [{"chosen": chosen, "rejected": rejected}])
            dataset = DPODataset(path, tokenizer, max_length=8)
            with self.assertRaisesRegex(ValueError, "filtering overlength pairs"):
                _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
