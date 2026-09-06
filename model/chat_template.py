"""Deterministic MiniMind-style chat serialization for the project tokenizer.

The project owns a 16K BPE vocabulary with only ``<bos>`` and ``<eos>`` as
conversation boundary tokens.  Role names and XML-like tool/thinking markers
remain ordinary text so the pretrained embedding table never needs resizing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


IGNORE_INDEX = -100
TEMPLATE_VERSION = "minimind-bpe16k-bos-eos-v1"

_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
_ALLOWED_MESSAGE_FIELDS = {
    "role",
    "content",
    "reasoning_content",
    "tools",
    "tool_calls",
}

_TOOLS_INTRO = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
)
_TOOLS_OUTRO = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


def _validate_max_length(max_length: int) -> None:
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")


def _special_id(tokenizer: Any, name: str) -> int:
    token_id = getattr(tokenizer, name, None)
    if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
        raise TypeError(f"tokenizer.{name} must be a non-negative integer")
    return token_id


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    """Encode ordinary text without asking the tokenizer to add boundaries."""

    try:
        encoded = tokenizer.encode(text, add_bos=False, add_eos=False)
    except TypeError as exc:
        raise TypeError(
            "tokenizer.encode must accept text, add_bos=False and add_eos=False"
        ) from exc

    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    if not isinstance(encoded, (list, tuple)):
        raise TypeError("tokenizer.encode must return a list of token IDs")

    ids = list(encoded)
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in ids):
        raise TypeError("tokenizer.encode returned a non-integer token ID")
    return ids


def _canonical_json(value: Any, field_name: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain JSON-serializable data") from exc


def _parse_json_field(value: Any, field_name: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be valid JSON when provided as text") from exc
    return value


def _optional_field(message: Mapping[str, Any], name: str) -> Any:
    value = message.get(name)
    return None if value is None or value == "" else value


def _validate_conversations(conversations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(conversations, (str, bytes)) or not isinstance(conversations, Sequence):
        raise TypeError("conversations must be a sequence of message mappings")
    if not conversations:
        raise ValueError("conversations must not be empty")

    validated: list[dict[str, Any]] = []
    tools_message_count = 0
    for index, raw_message in enumerate(conversations):
        if not isinstance(raw_message, Mapping):
            raise TypeError(f"conversation message {index} must be a mapping")

        unknown_fields = set(raw_message) - _ALLOWED_MESSAGE_FIELDS
        if unknown_fields:
            names = ", ".join(sorted(map(str, unknown_fields)))
            raise ValueError(f"conversation message {index} has unsupported fields: {names}")

        role = raw_message.get("role")
        if role not in _ALLOWED_ROLES:
            raise ValueError(
                f"conversation message {index} has invalid role {role!r}; "
                f"expected one of {sorted(_ALLOWED_ROLES)}"
            )

        content = raw_message.get("content")
        if not isinstance(content, str):
            raise TypeError(f"conversation message {index}.content must be a string")

        reasoning = raw_message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise TypeError(
                f"conversation message {index}.reasoning_content must be a string or null"
            )
        if role != "assistant" and reasoning not in (None, ""):
            raise ValueError("reasoning_content is only valid on assistant messages")

        tools = _optional_field(raw_message, "tools")
        if tools is not None:
            if role != "system":
                raise ValueError("tools is only valid on system messages")
            tools_message_count += 1
            if tools_message_count > 1:
                raise ValueError("only one system message may define tools")

        tool_calls = _optional_field(raw_message, "tool_calls")
        if tool_calls is not None and role != "assistant":
            raise ValueError("tool_calls is only valid on assistant messages")

        validated.append(dict(raw_message))

    return validated


def _normalise_tools(value: Any) -> list[Mapping[str, Any]]:
    parsed = _parse_json_field(value, "tools")
    if not isinstance(parsed, list):
        raise TypeError("tools must be a JSON array")
    for index, tool in enumerate(parsed):
        if not isinstance(tool, Mapping):
            raise TypeError(f"tools[{index}] must be a JSON object")
    return parsed


def _normalise_tool_calls(value: Any) -> list[dict[str, Any]]:
    parsed = _parse_json_field(value, "tool_calls")
    if not isinstance(parsed, list):
        raise TypeError("tool_calls must be a JSON array")

    normalised: list[dict[str, Any]] = []
    for index, raw_call in enumerate(parsed):
        if not isinstance(raw_call, Mapping):
            raise TypeError(f"tool_calls[{index}] must be a JSON object")

        if "function" in raw_call:
            unsupported = set(raw_call) - {"id", "type", "function"}
            if unsupported:
                raise ValueError(f"tool_calls[{index}] has unsupported wrapper fields")
            call = raw_call["function"]
        else:
            call = raw_call

        if not isinstance(call, Mapping):
            raise TypeError(f"tool_calls[{index}].function must be a JSON object")
        unsupported = set(call) - {"name", "arguments"}
        if unsupported:
            raise ValueError(f"tool_calls[{index}] has unsupported function fields")

        name = call.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"tool_calls[{index}].name must be a non-empty string")

        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            arguments = _parse_json_field(arguments, f"tool_calls[{index}].arguments")
        if not isinstance(arguments, Mapping):
            raise TypeError(f"tool_calls[{index}].arguments must be a JSON object")
        normalised.append({"name": name, "arguments": dict(arguments)})

    return normalised


def _render_system_payload(message: Mapping[str, Any]) -> str:
    content = message["content"]
    tools_value = _optional_field(message, "tools")
    if tools_value is None:
        return content

    tools = _normalise_tools(tools_value)
    rendered_tools = "\n".join(_canonical_json(tool, "tools") for tool in tools)
    prefix = f"{content}\n\n" if content else ""
    return prefix + _TOOLS_INTRO + rendered_tools + _TOOLS_OUTRO


def _split_reasoning(message: Mapping[str, Any]) -> tuple[str, str]:
    content = message["content"]
    if message.get("reasoning_content") is not None:
        return message.get("reasoning_content", ""), content

    if "</think>" not in content:
        return "", content

    before, after = content.split("</think>", 1)
    reasoning = before.rsplit("<think>", 1)[-1].lstrip("\n")
    return reasoning.rstrip("\n"), after.lstrip("\n")


def _render_assistant_payload(
    message: Mapping[str, Any],
    *,
    keep_empty_think: bool,
) -> str:
    reasoning, content = _split_reasoning(message)
    reasoning = reasoning.strip("\n")
    content = content.lstrip("\n")

    if reasoning:
        payload = f"<think>\n{reasoning}\n</think>\n\n{content}"
    elif keep_empty_think:
        payload = f"<think>\n\n</think>\n\n{content}"
    else:
        payload = content

    tool_calls_value = _optional_field(message, "tool_calls")
    if tool_calls_value is None:
        return payload

    calls = _normalise_tool_calls(tool_calls_value)
    rendered_calls = [
        f"<tool_call>\n{_canonical_json(call, 'tool_calls')}\n</tool_call>"
        for call in calls
    ]
    if not rendered_calls:
        return payload
    if content:
        return payload + "\n" + "\n".join(rendered_calls)
    return payload + "\n".join(rendered_calls)


def _canonicalise_tool_content(content: str) -> str:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return content
    return _canonical_json(value, "tool content")


def _render_tool_group(messages: Sequence[Mapping[str, Any]]) -> str:
    responses = []
    for message in messages:
        content = _canonicalise_tool_content(message["content"])
        responses.append(f"<tool_response>\n{content}\n</tool_response>")
    return "\n".join(responses)


def _conversation_turns(
    conversations: Sequence[Mapping[str, Any]],
    *,
    keep_empty_think: bool,
) -> list[tuple[str, str, bool]]:
    """Return ``(wire_role, payload, supervise_payload)`` turns."""

    messages = _validate_conversations(conversations)
    turns: list[tuple[str, str, bool]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message["role"]

        if role == "tool":
            end = index + 1
            while end < len(messages) and messages[end]["role"] == "tool":
                end += 1
            # MiniMind/Qwen-style templates expose tool results to the model as
            # one user turn, grouping adjacent tool responses under one header.
            turns.append(("user", _render_tool_group(messages[index:end]), False))
            index = end
            continue

        if role == "system":
            payload = _render_system_payload(message)
            supervise = False
        elif role == "user":
            payload = message["content"]
            supervise = False
        else:
            payload = _render_assistant_payload(
                message,
                keep_empty_think=keep_empty_think,
            )
            supervise = True

        turns.append((role, payload, supervise))
        index += 1

    return turns


def _encode_turn(
    role: str,
    payload: str,
    supervise_payload: bool,
    tokenizer: Any,
) -> tuple[list[int], list[int]]:
    bos_id = _special_id(tokenizer, "bos_id")
    eos_id = _special_id(tokenizer, "eos_id")

    # Keep header and payload in separate tokenizer calls.  A BPE merge can
    # therefore never cross the exact point where the loss mask changes.
    header_ids = [bos_id] + _encode_text(tokenizer, f"{role}\n")
    payload_ids = _encode_text(tokenizer, payload)
    input_ids = header_ids + payload_ids + [eos_id]

    labels = [IGNORE_INDEX] * len(header_ids)
    if supervise_payload:
        labels.extend(payload_ids)
        labels.append(eos_id)
    else:
        labels.extend([IGNORE_INDEX] * (len(payload_ids) + 1))
    return input_ids, labels


def _encode_completed_conversation(
    conversations: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    keep_empty_think: bool,
) -> tuple[list[int], list[int]]:
    input_ids: list[int] = []
    labels: list[int] = []
    for role, payload, supervise in _conversation_turns(
        conversations,
        keep_empty_think=keep_empty_think,
    ):
        turn_ids, turn_labels = _encode_turn(role, payload, supervise, tokenizer)
        input_ids.extend(turn_ids)
        labels.extend(turn_labels)
    return input_ids, labels


def encode_sft_conversation(
    conversations: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
    keep_empty_think: bool = False,
) -> tuple[list[int], list[int]]:
    """Encode one SFT conversation with an assistant-only causal-LM target.

    The result is not padded.  Both arrays are right-truncated to
    ``max_length``; truncation never appends or substitutes an artificial EOS.
    """

    _validate_max_length(max_length)
    if not isinstance(keep_empty_think, bool):
        raise TypeError("keep_empty_think must be a bool")

    input_ids, labels = _encode_completed_conversation(
        conversations,
        tokenizer,
        keep_empty_think=keep_empty_think,
    )
    return input_ids[:max_length], labels[:max_length]


def encode_generation_prompt(
    conversations: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
    open_thinking: bool = False,
) -> list[int]:
    """Encode completed history and append the shared assistant prompt."""

    _validate_max_length(max_length)
    if not isinstance(open_thinking, bool):
        raise TypeError("open_thinking must be a bool")

    history_ids, _ = _encode_completed_conversation(
        conversations,
        tokenizer,
        keep_empty_think=False,
    )
    prompt_ids = [_special_id(tokenizer, "bos_id")]
    prompt_ids.extend(_encode_text(tokenizer, "assistant\n"))
    if open_thinking:
        prompt_ids.extend(_encode_text(tokenizer, "<think>\n"))
    else:
        prompt_ids.extend(_encode_text(tokenizer, "<think>\n\n</think>\n\n"))

    if len(prompt_ids) > max_length:
        raise ValueError("max_length is too short to hold the assistant generation prompt")
    history_budget = max_length - len(prompt_ids)
    if history_budget:
        history_ids = history_ids[-history_budget:]
    else:
        history_ids = []
    return history_ids + prompt_ids


__all__ = [
    "IGNORE_INDEX",
    "TEMPLATE_VERSION",
    "encode_generation_prompt",
    "encode_sft_conversation",
]
