import json
import tempfile
import unittest
from pathlib import Path

import torch

from model.chat_template import (
    IGNORE_INDEX,
    encode_generation_prompt,
    encode_sft_conversation,
)
from model.sft_data import SYSTEM_PROMPTS, SFTDataset
from model.tokenizer import CharTokenizer


class TestSFTData(unittest.TestCase):
    @staticmethod
    def make_tokenizer(*values) -> CharTokenizer:
        """Build a transparent tokenizer containing every character in fixtures."""

        def collect(value):
            if isinstance(value, str):
                return value
            return json.dumps(value, ensure_ascii=False, sort_keys=True)

        wire_alphabet = (
            "system\nuser\nassistant\ntool\n"
            "<think></think><tools></tools><tool_call></tool_call>"
            "<tool_response></tool_response># Tools:-_.,:(){}[]\" "
        )
        return CharTokenizer.build(wire_alphabet + "".join(collect(value) for value in values))

    @staticmethod
    def find_subsequence(sequence, subsequence, start=0):
        for index in range(start, len(sequence) - len(subsequence) + 1):
            if sequence[index : index + len(subsequence)] == subsequence:
                return index
        raise AssertionError(f"subsequence not found: {subsequence}")

    @staticmethod
    def write_jsonl(directory: str, records) -> Path:
        path = Path(directory) / "sft.jsonl"
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def test_single_turn_masks_header_and_supervises_answer_eos_and_first_shifted_token(self):
        conversation = [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "回答"},
        ]
        tokenizer = self.make_tokenizer(conversation)
        input_ids, labels = encode_sft_conversation(conversation, tokenizer, max_length=128)

        assistant_header = [tokenizer.bos_id] + tokenizer.encode("assistant\n")
        header_start = self.find_subsequence(input_ids, assistant_header)
        payload_start = header_start + len(assistant_header)
        answer_ids = tokenizer.encode("回答")

        self.assertTrue(all(label == IGNORE_INDEX for label in labels[:payload_start]))
        self.assertEqual(input_ids[payload_start : payload_start + len(answer_ids)], answer_ids)
        self.assertEqual(labels[payload_start : payload_start + len(answer_ids)], answer_ids)
        self.assertEqual(input_ids[payload_start + len(answer_ids)], tokenizer.eos_id)
        self.assertEqual(labels[payload_start + len(answer_ids)], tokenizer.eos_id)

        # Causal training uses logits[:, :-1] against labels[:, 1:].  The first
        # answer token must therefore survive as the first non-ignored shifted target.
        shifted_labels = labels[1:]
        first_shifted_target = next(
            index for index, token_id in enumerate(shifted_labels) if token_id != IGNORE_INDEX
        )
        self.assertEqual(first_shifted_target, payload_start - 1)
        self.assertEqual(shifted_labels[first_shifted_target], answer_ids[0])

    def test_multiturn_supervises_every_assistant_turn_only(self):
        conversation = [
            {"role": "system", "content": "遵循要求"},
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"},
            {"role": "assistant", "content": "第二答"},
        ]
        tokenizer = self.make_tokenizer(conversation)
        input_ids, labels = encode_sft_conversation(conversation, tokenizer, max_length=256)

        supervised = [token_id for token_id in labels if token_id != IGNORE_INDEX]
        expected = (
            tokenizer.encode("第一答")
            + [tokenizer.eos_id]
            + tokenizer.encode("第二答")
            + [tokenizer.eos_id]
        )
        self.assertEqual(supervised, expected)

        for text in ("遵循要求", "第一问", "第二问"):
            start = self.find_subsequence(input_ids, tokenizer.encode(text))
            self.assertTrue(
                all(label == IGNORE_INDEX for label in labels[start : start + len(text)])
            )

    def test_reasoning_and_final_answer_are_both_supervised(self):
        conversation = [
            {"role": "user", "content": "计算"},
            {
                "role": "assistant",
                "reasoning_content": "先分析再计算",
                "content": "答案是四",
            },
        ]
        tokenizer = self.make_tokenizer(conversation)
        _, labels = encode_sft_conversation(conversation, tokenizer, max_length=256)

        supervised = [token_id for token_id in labels if token_id != IGNORE_INDEX]
        self.assertEqual(supervised[-1], tokenizer.eos_id)
        self.assertEqual(
            tokenizer.decode(supervised[:-1]),
            "<think>\n先分析再计算\n</think>\n\n答案是四",
        )

    def test_tools_tool_calls_and_tool_response_follow_masking_contract(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "查询时间",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        conversation = [
            {"role": "system", "content": "你可以调用工具", "tools": tools},
            {"role": "user", "content": "现在几点"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": '{"hour":10}'},
            {"role": "assistant", "content": "现在是十点"},
        ]
        tokenizer = self.make_tokenizer(conversation)
        input_ids, labels = encode_sft_conversation(conversation, tokenizer, max_length=2048)

        wire_text = tokenizer.decode(input_ids)
        supervised_text = tokenizer.decode(
            [token_id for token_id in labels if token_id != IGNORE_INDEX]
        )
        self.assertIn("<tools>", wire_text)
        self.assertIn("<tool_response>", wire_text)
        self.assertIn('<tool_call>\n{"arguments":{},"name":"get_time"}', wire_text)
        self.assertIn("<tool_call>", supervised_text)
        self.assertIn("现在是十点", supervised_text)
        self.assertNotIn("<tools>", supervised_text)
        self.assertNotIn("<tool_response>", supervised_text)
        self.assertNotIn("现在几点", supervised_text)

    def test_dataset_right_padding_shape_and_dtype(self):
        conversation = [
            {"role": "user", "content": "问"},
            {"role": "assistant", "content": "答"},
        ]
        tokenizer = self.make_tokenizer(conversation)
        unpadded_ids, unpadded_labels = encode_sft_conversation(
            conversation, tokenizer, max_length=256
        )
        max_length = len(unpadded_ids) + 7

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [{"conversations": conversation}])
            input_ids, labels = SFTDataset(
                path,
                tokenizer,
                max_length=max_length,
                empty_think_ratio=0.0,
            )[0]

        self.assertEqual(input_ids.shape, (max_length,))
        self.assertEqual(labels.shape, (max_length,))
        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(input_ids[: len(unpadded_ids)].tolist(), unpadded_ids)
        self.assertEqual(labels[: len(unpadded_labels)].tolist(), unpadded_labels)
        self.assertTrue(torch.all(input_ids[len(unpadded_ids) :] == tokenizer.pad_id))
        self.assertTrue(torch.all(labels[len(unpadded_labels) :] == IGNORE_INDEX))

    def test_truncation_never_fabricates_eos_and_dataset_rejects_no_target(self):
        conversation = [
            {"role": "user", "content": "问"},
            {"role": "assistant", "content": "abcdefghijklmnop"},
        ]
        tokenizer = self.make_tokenizer(conversation)
        full_ids, full_labels = encode_sft_conversation(conversation, tokenizer, max_length=256)
        first_target = next(
            index for index, token_id in enumerate(full_labels) if token_id != IGNORE_INDEX
        )
        max_length = first_target + 3
        truncated_ids, truncated_labels = encode_sft_conversation(
            conversation, tokenizer, max_length=max_length
        )
        self.assertEqual(truncated_ids, full_ids[:max_length])
        self.assertEqual(truncated_labels, full_labels[:max_length])
        self.assertNotEqual(truncated_ids[-1], tokenizer.eos_id)
        self.assertEqual(truncated_labels[-1], truncated_ids[-1])

        user_heavy = [
            {"role": "user", "content": "这是一个非常非常长的问题"},
            {"role": "assistant", "content": "答"},
        ]
        tokenizer = self.make_tokenizer(user_heavy)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [{"conversations": user_heavy}])
            dataset = SFTDataset(
                path,
                tokenizer,
                max_length=10,
                empty_think_ratio=0.0,
                system_prompt_ratio=1.0,
            )
            with self.assertRaisesRegex(ValueError, "no supervised assistant target"):
                _ = dataset[0]

    def test_generation_prompt_matches_training_prefix_and_survives_left_truncation(self):
        history = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "旧问题很长很长很长"},
            {"role": "assistant", "content": "旧回答也很长很长很长"},
            {"role": "user", "content": "新问题"},
        ]
        final_answer = {"role": "assistant", "content": "新答案"}
        tokenizer = self.make_tokenizer(history, final_answer)

        generation_ids = encode_generation_prompt(
            history, tokenizer, max_length=1024, open_thinking=False
        )
        history_ids, _ = encode_sft_conversation(
            history,
            tokenizer,
            max_length=1024,
            keep_empty_think=False,
        )
        training_ids, _ = encode_sft_conversation(
            history + [final_answer],
            tokenizer,
            max_length=1024,
            keep_empty_think=False,
        )
        assistant_header = [tokenizer.bos_id] + tokenizer.encode("assistant\n")
        self.assertEqual(generation_ids[: len(history_ids)], history_ids)
        self.assertEqual(training_ids[: len(history_ids)], history_ids)
        self.assertEqual(
            generation_ids[len(history_ids) : len(history_ids) + len(assistant_header)],
            assistant_header,
        )
        self.assertEqual(
            training_ids[len(history_ids) : len(history_ids) + len(assistant_header)],
            assistant_header,
        )

        prompt_suffix = (
            [tokenizer.bos_id]
            + tokenizer.encode("assistant\n")
            + tokenizer.encode("<think>\n\n</think>\n\n")
        )
        short_max_length = len(prompt_suffix) + 8
        shortened = encode_generation_prompt(
            history,
            tokenizer,
            max_length=short_max_length,
            open_thinking=False,
        )
        self.assertEqual(len(shortened), short_max_length)
        self.assertEqual(shortened[-len(prompt_suffix) :], prompt_suffix)

    def test_encoding_and_empty_think_sampling_are_deterministic(self):
        records = [
            {
                "conversations": [
                    {"role": "user", "content": f"问题{index}"},
                    {"role": "assistant", "content": f"答案{index}"},
                ]
            }
            for index in range(6)
        ]
        tokenizer = self.make_tokenizer(records)
        conversation = records[0]["conversations"]
        self.assertEqual(
            encode_sft_conversation(conversation, tokenizer, max_length=128),
            encode_sft_conversation(conversation, tokenizer, max_length=128),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, records)
            first = SFTDataset(
                path,
                tokenizer,
                max_length=128,
                seed=42,
                empty_think_ratio=0.5,
            )
            second = SFTDataset(
                path,
                tokenizer,
                max_length=128,
                seed=42,
                empty_think_ratio=0.5,
            )
            first_pass = [(first[index][0].clone(), first[index][1].clone()) for index in range(6)]
            torch.manual_seed(999)
            second_pass = [second[index] for index in reversed(range(6))]

        for index, (input_ids, labels) in enumerate(first_pass):
            reversed_input, reversed_labels = second_pass[5 - index]
            self.assertTrue(torch.equal(input_ids, reversed_input))
            self.assertTrue(torch.equal(labels, reversed_labels))

    def test_system_prompt_ratio_zero_and_one_have_exact_boundary_behaviour(self):
        records = [
            {
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ]
        tokenizer = self.make_tokenizer(records, SYSTEM_PROMPTS)

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, records)
            disabled = SFTDataset(
                path,
                tokenizer,
                max_length=256,
                empty_think_ratio=0.0,
                system_prompt_ratio=0.0,
            )
            enabled = SFTDataset(
                path,
                tokenizer,
                max_length=256,
                empty_think_ratio=0.0,
                system_prompt_ratio=1.0,
            )

            disabled_ids, _ = disabled[0]
            enabled_ids, _ = enabled[0]

        disabled_wire = tokenizer.decode(
            disabled_ids[disabled_ids != tokenizer.pad_id].tolist()
        )
        enabled_wire = tokenizer.decode(enabled_ids[enabled_ids != tokenizer.pad_id].tolist())
        self.assertNotIn("system\n", disabled_wire)
        # CharTokenizer.decode omits special tokens by design, so the visible
        # wire representation starts at the system role header.
        self.assertTrue(enabled_wire.startswith("system\n"))
        self.assertIn(enabled._system_prompt(0), SYSTEM_PROMPTS)
        self.assertIn(enabled._system_prompt(0), enabled_wire)

        # Reading an item must not mutate the source conversations held by
        # either the caller or the dataset.
        self.assertEqual(records[0]["conversations"][0]["role"], "user")
        self.assertEqual(enabled.conversations[0][0]["role"], "user")
        self.assertEqual(len(enabled.conversations[0]), 2)

    def test_system_prompt_augmentation_falls_back_when_it_truncates_all_targets(self):
        conversation = [
            {"role": "user", "content": "a moderately long user question"},
            {"role": "assistant", "content": "answer"},
        ]
        tokenizer = self.make_tokenizer(conversation, SYSTEM_PROMPTS)
        _, full_labels = encode_sft_conversation(conversation, tokenizer, max_length=4096)
        first_target = next(
            index for index, token_id in enumerate(full_labels) if token_id != IGNORE_INDEX
        )
        # Keep exactly the first target in the original sample.  Prepending any
        # official system prompt then pushes all assistant targets out of view.
        max_length = first_target + 1

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [{"conversations": conversation}])
            dataset = SFTDataset(
                path,
                tokenizer,
                max_length=max_length,
                empty_think_ratio=0.0,
                system_prompt_ratio=1.0,
            )

            augmented = dataset._conversation_for_index(0)
            _, augmented_labels = encode_sft_conversation(
                augmented,
                tokenizer,
                max_length=max_length,
                keep_empty_think=False,
            )
            self.assertFalse(
                any(label != IGNORE_INDEX for label in augmented_labels[1:])
            )

            actual_ids, actual_labels = dataset[0]

        expected_ids, expected_labels = encode_sft_conversation(
            conversation,
            tokenizer,
            max_length=max_length,
            keep_empty_think=False,
        )
        self.assertEqual(actual_ids.tolist(), expected_ids)
        self.assertEqual(actual_labels.tolist(), expected_labels)
        self.assertTrue(torch.any(actual_labels[1:] != IGNORE_INDEX))
        self.assertEqual(dataset.conversations[0], conversation)

    def test_system_prompt_augmentation_skips_existing_system_and_tool_samples(self):
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        records = [
            {
                "conversations": [
                    {"role": "system", "content": "existing system"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "question"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "lookup", "arguments": {}}],
                    },
                    {"role": "tool", "content": "result"},
                    {"role": "assistant", "content": "answer"},
                ]
            },
            {
                "conversations": [
                    {"role": "system", "content": "", "tools": tools},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            },
        ]
        tokenizer = self.make_tokenizer(records, SYSTEM_PROMPTS)

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, records)
            dataset = SFTDataset(
                path,
                tokenizer,
                max_length=1024,
                empty_think_ratio=0.0,
                system_prompt_ratio=1.0,
            )

            self.assertIs(dataset._conversation_for_index(0), dataset.conversations[0])
            self.assertIs(dataset._conversation_for_index(1), dataset.conversations[1])
            self.assertIs(dataset._conversation_for_index(2), dataset.conversations[2])

    def test_system_prompt_sampling_is_deterministic_and_order_independent(self):
        records = [
            {
                "conversations": [
                    {"role": "user", "content": f"question {index}"},
                    {"role": "assistant", "content": f"answer {index}"},
                ]
            }
            for index in range(20)
        ]
        tokenizer = self.make_tokenizer(records, SYSTEM_PROMPTS)

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, records)
            first = SFTDataset(
                path,
                tokenizer,
                max_length=256,
                seed=73,
                empty_think_ratio=0.0,
                system_prompt_ratio=0.5,
            )
            second = SFTDataset(
                path,
                tokenizer,
                max_length=256,
                seed=73,
                empty_think_ratio=0.0,
                system_prompt_ratio=0.5,
            )

            forward = [first._conversation_for_index(index) for index in range(len(first))]
            torch.manual_seed(987654)
            reverse = {
                index: second._conversation_for_index(index)
                for index in reversed(range(len(second)))
            }

        self.assertEqual(forward, [reverse[index] for index in range(len(first))])
        augmented = [conversation for conversation in forward if conversation[0]["role"] == "system"]
        untouched = [conversation for conversation in forward if conversation[0]["role"] != "system"]
        self.assertTrue(augmented)
        self.assertTrue(untouched)
        self.assertTrue(all(conversation[0]["content"] in SYSTEM_PROMPTS for conversation in augmented))

    def test_system_prompt_ratio_validation_and_official_prompt_inventory(self):
        self.assertEqual(len(SYSTEM_PROMPTS), 10)
        self.assertEqual(len(set(SYSTEM_PROMPTS)), 10)
        records = [
            {
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ]
        tokenizer = self.make_tokenizer(records, SYSTEM_PROMPTS)

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, records)
            for invalid_ratio in (-0.1, 1.1, float("inf"), float("nan"), True, "0.2"):
                with self.subTest(invalid_ratio=invalid_ratio):
                    with self.assertRaisesRegex(ValueError, "system_prompt_ratio"):
                        SFTDataset(path, tokenizer, system_prompt_ratio=invalid_ratio)


if __name__ == "__main__":
    unittest.main()
