"""Pure-Python token/mask invariants for multi-turn Agent rollouts."""
from __future__ import annotations

import math


def trim_generated_turn(ids, logps, padding_mask, *, eos_token_id, pad_token_id):
    """Retain sampled tokens through the first EOS, inclusive, for this turn."""
    if not (len(ids) == len(logps) == len(padding_mask)):
        raise ValueError("Rollout token/logprob/mask lengths differ")
    kept_ids, kept_logps = [], []
    for token, logp, valid in zip(ids, logps, padding_mask):
        if not valid:
            break
        # If pad==eos, the first valid occurrence is EOS, not padding.
        if token == pad_token_id and token != eos_token_id:
            break
        if not math.isfinite(logp):
            raise ValueError("Non-finite rollout logprob")
        kept_ids.append(token)
        kept_logps.append(logp)
        if token == eos_token_id:
            break
    return kept_ids, kept_logps


def observation_suffix(observed_ids, prompt_ids, response_ids):
    """Do not silently reassign old logprobs after template re-tokenization."""
    prefix = prompt_ids + response_ids
    if observed_ids[:len(prefix)] != prefix:
        raise ValueError("Chat template changed sampled token prefix; cannot align Agent old logprobs")
    return observed_ids[len(prefix):]


def tool_observation_tokens(tokenizer, tool_messages, *, tools, add_generation_prompt,
                            open_thinking, already_ended):
    """Render only the template-owned suffix, never re-encode sampled history.

    The official template normalizes assistant reasoning whitespace. Rendering
    that history again can change sampled token IDs while keeping stale old
    logprobs. A synthetic anchor extracts the observation serialization, keeping
    the actual policy-generated prefix byte/token-for-token intact.
    """
    marker = "EVOMIND_ASSISTANT_SUFFIX_ANCHOR"
    while any(marker in message['content'] for message in tool_messages):
        marker += "_"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}, {"role": "assistant", "content": marker}, *tool_messages],
        tokenize=False, add_generation_prompt=add_generation_prompt, tools=tools, open_thinking=open_thinking)
    if text.count(marker) != 1:
        raise ValueError("Cannot isolate the tokenizer's tool-observation suffix")
    suffix = text.split(marker, 1)[1]
    ids = tokenizer(suffix, add_special_tokens=False)['input_ids']
    if not ids or ids[0] != tokenizer.eos_token_id:
        raise ValueError("Expected the official template's assistant EOS before tool observation")
    return ids[1:] if already_ended else ids


def pack_agent_sample(prompt, response, mask, old_logps, max_total_len):
    if not prompt or max_total_len < 2:
        raise ValueError("Agent sample needs a prompt and at least two total tokens")
    if not (len(response) == len(mask) == len(old_logps)) or any(m not in (0, 1) for m in mask):
        raise ValueError("Agent response/mask/logprob lengths differ")
    ids = prompt + response
    full_mask = [0] * len(prompt) + mask
    prediction_logps = [0.0] * (len(prompt) - 1) + old_logps
    if len(ids) > max_total_len:
        ids, full_mask = ids[-max_total_len:], full_mask[-max_total_len:]
        prediction_logps = prediction_logps[-(len(ids) - 1):]
    # Token zero has no preceding retained context and cannot be a policy target.
    full_mask[0] = 0
    if len(prediction_logps) != len(ids) - 1:
        raise ValueError("Agent next-token logprob alignment differs")
    first_response = next((i for i, valid in enumerate(full_mask) if valid), len(ids))
    return ids, full_mask, first_response, prediction_logps
