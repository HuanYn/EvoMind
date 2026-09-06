"""Stream-inspect MiniMind SFT JSONL before preparing training tensors."""

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


def summarize(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def preview(text: str, limit: int = 100) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def iter_prefix_records(path: Path, max_records: int):
    selected = 0
    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            if max_records and selected >= max_records:
                break
            selected += 1
            yield line_number, raw_line


def reservoir_sample_records(
    path: Path, sample_size: int, seed: int
) -> tuple[list[tuple[int, str]], int]:
    """Uniformly sample JSONL records while keeping only sample_size lines in memory."""
    rng = random.Random(seed)
    reservoir: list[tuple[int, str]] = []
    nonempty_records = 0
    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            nonempty_records += 1
            item = (line_number, raw_line)
            if len(reservoir) < sample_size:
                reservoir.append(item)
                continue
            replacement_index = rng.randrange(nonempty_records)
            if replacement_index < sample_size:
                reservoir[replacement_index] = item
    return reservoir, nonempty_records


def follows_basic_chat_order(conversations: list[dict]) -> bool:
    """Check ordinary system?/user/assistant alternation; tool chats are excluded."""
    roles = [message.get("role") for message in conversations]
    if any(role not in {"system", "user", "assistant"} for role in roles):
        return True
    if any(message.get("tools") or message.get("tool_calls") for message in conversations):
        return True
    if roles and roles[0] == "system":
        roles = roles[1:]
    return bool(roles) and all(
        role == ("user" if index % 2 == 0 else "assistant")
        for index, role in enumerate(roles)
    )


def approximate_project_template_tokens(
    conversations: list[dict], tokenizer: Tokenizer
) -> int:
    """Estimate length under the BOS/role/content/EOS template planned for this project."""
    total = 0
    for message in conversations:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        content = content if isinstance(content, str) else str(content)
        reasoning = message.get("reasoning_content", "")
        reasoning = reasoning if isinstance(reasoning, str) else str(reasoning)

        text = f"{role}\n"
        if role == "assistant" and reasoning.strip():
            text += f"<think>\n{reasoning}\n</think>\n"
        text += content

        for field in ("tools", "tool_calls"):
            value = message.get(field)
            if value:
                serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                text += f"\n<{field}>\n{serialized}"

        # One existing <bos> and <eos> token surround every message.
        total += 1 + len(tokenizer.encode(text).ids) + 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MiniMind SFT JSONL")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/minimind_sft/sft_t2t_mini.jsonl"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizers/minimind_bpe_16k_110k.json"),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=10_000,
        help="Maximum non-empty lines to inspect; use 0 for the complete file",
    )
    parser.add_argument(
        "--sampling",
        choices=["prefix", "reservoir"],
        default="reservoir",
        help="Use a fast prefix or a uniform sample drawn while streaming the complete file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reservoir sampling seed")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[256, 512, 768, 1024],
    )
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/sft_data_inspection_10k.json"),
    )
    args = parser.parse_args()

    if args.max_records < 0:
        raise ValueError("--max-records must be non-negative")
    if any(length < 1 for length in args.context_lengths):
        raise ValueError("--context-lengths values must be positive")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    roles: Counter[str] = Counter()
    message_fields: Counter[str] = Counter()
    empty_content_by_role: Counter[str] = Counter()
    invalid_examples: list[dict] = []
    safe_samples: list[dict] = []
    turn_counts: list[int] = []
    character_counts: list[int] = []
    token_counts: list[int] = []
    seen_hashes: set[bytes] = set()

    attempted = valid = malformed_json = invalid_schema = duplicate_records = 0
    empty_messages = empty_without_payload = unknown_role_messages = non_basic_order = 0
    reasoning_messages = nonempty_reasoning_messages = 0
    tools_messages = tool_call_messages = 0

    source_nonempty_records = None
    if args.max_records and args.sampling == "reservoir":
        records, source_nonempty_records = reservoir_sample_records(
            args.input, args.max_records, args.seed
        )
    else:
        records = iter_prefix_records(args.input, args.max_records)

    for line_number, raw_line in records:
            attempted += 1

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                malformed_json += 1
                if len(invalid_examples) < 5:
                    invalid_examples.append({"line": line_number, "reason": str(error)})
                continue

            conversations = record.get("conversations") if isinstance(record, dict) else None
            if not isinstance(conversations, list) or not conversations:
                invalid_schema += 1
                if len(invalid_examples) < 5:
                    invalid_examples.append(
                        {"line": line_number, "reason": "missing/non-list/empty conversations"}
                    )
                continue
            if any(not isinstance(message, dict) for message in conversations):
                invalid_schema += 1
                if len(invalid_examples) < 5:
                    invalid_examples.append(
                        {"line": line_number, "reason": "conversation contains a non-object message"}
                    )
                continue

            valid += 1
            canonical = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
            digest = hashlib.blake2b(canonical, digest_size=16).digest()
            duplicate_records += digest in seen_hashes
            seen_hashes.add(digest)

            turn_counts.append(len(conversations))
            if not follows_basic_chat_order(conversations):
                non_basic_order += 1

            conversation_characters = 0
            for message in conversations:
                message_fields.update(message.keys())
                role = message.get("role")
                role_name = role if isinstance(role, str) else "<invalid>"
                roles[role_name] += 1
                unknown_role_messages += role_name not in {"system", "user", "assistant", "tool"}

                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    empty_messages += 1
                    empty_content_by_role[role_name] += 1
                    reasoning = message.get("reasoning_content")
                    has_reasoning = isinstance(reasoning, str) and bool(reasoning.strip())
                    if not has_reasoning and not message.get("tools") and not message.get("tool_calls"):
                        empty_without_payload += 1
                if isinstance(content, str):
                    conversation_characters += len(content)

                if "reasoning_content" in message:
                    reasoning_messages += 1
                    reasoning = message.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning.strip():
                        nonempty_reasoning_messages += 1
                        conversation_characters += len(reasoning)
                tools_messages += bool(message.get("tools"))
                tool_call_messages += bool(message.get("tool_calls"))

            character_counts.append(conversation_characters)
            token_counts.append(approximate_project_template_tokens(conversations, tokenizer))

            if len(safe_samples) < args.sample_count:
                safe_samples.append(
                    {
                        "line": line_number,
                        "roles": [message.get("role") for message in conversations],
                        "first_user_preview": next(
                            (
                                preview(message["content"])
                                for message in conversations
                                if message.get("role") == "user"
                                and isinstance(message.get("content"), str)
                            ),
                            "",
                        ),
                        "first_assistant_preview": next(
                            (
                                preview(message["content"])
                                for message in conversations
                                if message.get("role") == "assistant"
                                and isinstance(message.get("content"), str)
                            ),
                            "",
                        ),
                        "approximate_tokens": token_counts[-1],
                    }
                )

    if attempted == 0:
        raise ValueError("No non-empty JSONL records found")

    over_context = {
        str(length): {
            "records": sum(count > length for count in token_counts),
            "rate": sum(count > length for count in token_counts) / max(valid, 1),
        }
        for length in sorted(set(args.context_lengths))
    }
    if source_nonempty_records is None and args.max_records == 0:
        source_nonempty_records = attempted
    inspected_complete_source = (
        args.max_records == 0
        or source_nonempty_records is not None
        and attempted == source_nonempty_records
    )
    inspection_scope = (
        "complete"
        if inspected_complete_source
        else "uniform_reservoir_sample"
        if args.sampling == "reservoir"
        else "prefix_sample"
    )
    sample_fraction = (
        attempted / source_nonempty_records
        if source_nonempty_records is not None and source_nonempty_records
        else None
    )
    report = {
        "input": str(args.input),
        "tokenizer": str(args.tokenizer),
        "inspection_scope": inspection_scope,
        "sampling_method": "algorithm_r" if inspection_scope == "uniform_reservoir_sample" else args.sampling,
        "sampling_seed": args.seed if inspection_scope == "uniform_reservoir_sample" else None,
        "source_nonempty_records": source_nonempty_records,
        "sample_fraction": sample_fraction,
        "attempted_records": attempted,
        "valid_records": valid,
        "malformed_json_records": malformed_json,
        "invalid_schema_records": invalid_schema,
        "exact_duplicates_within_sample": duplicate_records,
        "role_counts": dict(roles),
        "message_field_counts": dict(message_fields),
        "empty_or_non_string_content_messages": empty_messages,
        "empty_content_by_role": dict(empty_content_by_role),
        "empty_content_without_reasoning_or_tool_payload": empty_without_payload,
        "unknown_role_messages": unknown_role_messages,
        "basic_chat_order_violations": non_basic_order,
        "messages_with_reasoning_field": reasoning_messages,
        "messages_with_nonempty_reasoning": nonempty_reasoning_messages,
        "messages_with_tools": tools_messages,
        "messages_with_tool_calls": tool_call_messages,
        "turns_per_conversation": summarize(turn_counts),
        "characters_per_conversation": summarize(character_counts),
        "approximate_project_template_tokens": summarize(token_counts),
        "over_context_length": over_context,
        "samples": safe_samples,
        "invalid_examples": invalid_examples,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"scope: {report['inspection_scope']} | source records: "
        f"{source_nonempty_records if source_nonempty_records is not None else 'not counted'} | "
        f"attempted: {attempted:,} | valid: {valid:,}"
    )
    print(f"malformed JSON: {malformed_json:,} | invalid schema: {invalid_schema:,}")
    print(f"roles: {dict(roles)}")
    print(
        f"reasoning messages: {reasoning_messages:,} "
        f"(non-empty: {nonempty_reasoning_messages:,}) | "
        f"tools: {tools_messages:,} | tool calls: {tool_call_messages:,}"
    )
    print(f"turns/conversation: {report['turns_per_conversation']}")
    print(f"approximate template tokens: {report['approximate_project_template_tokens']}")
    for length, stats in over_context.items():
        print(f"> {length} tokens: {stats['records']:,} ({stats['rate']:.2%})")
    print(f"saved report: {args.report}")


if __name__ == "__main__":
    main()
