"""Recorded automatic prompts from pinned upstream eval_llm.py; no human scores."""
import ast
import argparse
import collections
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evomind_run import atomic_json, sha256
from evomind_load_text_model import load_model

def upstream_prompts():
    tree = ast.parse((ROOT / "eval_llm.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "prompts" for t in node.targets):
            return [(text, None) for text in ast.literal_eval(node.value)]
    raise ValueError("Upstream automatic prompt list missing")

PROMPTS = upstream_prompts()
MODES = ("sampled",)

def repetition(ids, n):
    grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return 1-len(set(grams))/len(grams) if grams else None

def validate_saved_rows(rows, seeds, thinking, *, complete=False):
    expected = {(mode, seed, index) for mode in MODES
                for seed in ([seeds[0]] if mode == "greedy" else seeds) for index in range(len(PROMPTS))}
    keys = set()
    for row in rows:
        key = (row["mode"], row["seed"], row["prompt_id"])
        if key not in expected or key in keys:
            raise ValueError("Unexpected or duplicate saved diagnostic identity")
        prompt, answer = PROMPTS[row["prompt_id"]]
        if row["prompt"] != prompt or row["expected_short_answer"] != answer or row["thinking"] != thinking:
            raise ValueError("Saved diagnostic prompt/target/thinking differs")
        keys.add(key)
    if complete and keys != expected:
        raise ValueError("Incomplete diagnostic identities")
    return keys

def valid_mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return sum(values)/len(values) if values else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lora", help="Adapter file; --checkpoint must remain the exact SFT base")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Reuse only identically contracted outputs by default, including supervisor retries; --no-resume refuses existing outputs")
    a = p.parse_args()
    dest = Path(a.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if a.max_new_tokens < 1 or not a.seeds or len(set(a.seeds)) != len(a.seeds):
        raise ValueError("Positive generation budget and unique seeds required")
    contract = {"checkpoint":str(Path(a.checkpoint).resolve()), "checkpoint_sha256":sha256(a.checkpoint),
        "lora":str(Path(a.lora).resolve()) if a.lora else None, "lora_sha256":sha256(a.lora) if a.lora else None,
        "tokenizer_sha256":{f.name:sha256(f) for f in sorted((ROOT / "model").glob("*.json"))},
        "evaluator_sha256":sha256(__file__), "loader_sha256":sha256(ROOT / "scripts/evomind_load_text_model.py"),
        "prompts":PROMPTS, "seeds":a.seeds, "max_new_tokens":a.max_new_tokens, "thinking":a.thinking,
        "upstream_eval_sha256":sha256(ROOT / "eval_llm.py"),
        "sampling":{"temperature":0.85,"top_k":50,"top_p":0.95}}
    contract = json.loads(json.dumps(contract,ensure_ascii=False))
    if (dest / "contract.json").exists():
        if not a.resume or json.loads((dest / "contract.json").read_text(encoding="utf-8")) != contract:
            raise ValueError("Existing diagnostic requires resume enabled and unchanged inputs/decoding/code")
    else:
        if (dest / "records.jsonl").exists() or (dest / "summary.json").exists():
            raise FileExistsError("Unbound previous outputs; choose a fresh evaluation directory")
        atomic_json(dest / "contract.json",contract)
    rows = []
    if (dest / "records.jsonl").exists():
        rows = [json.loads(line) for line in (dest / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = validate_saved_rows(rows, a.seeds, a.thinking)
    if (dest / "summary.json").exists():
        previous = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
        validate_saved_rows(rows, a.seeds, a.thinking, complete=True)
        if (previous.get("status") != "completed" or any(previous.get(k) != v for k,v in contract.items())
                or previous.get("records_sha256") != sha256(dest / "records.jsonl")):
            raise ValueError("Completed diagnostic receipt or raw records changed")
        print("Validated completed diagnostic; no model loaded.", flush=True)
        return
    model, tokenizer = load_model(a.checkpoint, a.lora)
    import torch
    torch.cuda.reset_peak_memory_stats()
    with (dest / "records.jsonl").open("a", encoding="utf-8") as out:
        for mode in MODES:
            for seed in ([a.seeds[0]] if mode == "greedy" else a.seeds):
                for index, (prompt, expected) in enumerate(PROMPTS):
                    if (mode,seed,index) in keys:
                        continue
                    torch.manual_seed(seed + index)
                    text = tokenizer.apply_chat_template([{"role":"user","content":prompt}],
                            tokenize=False, add_generation_prompt=True, open_thinking=a.thinking)
                    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        answer = model.generate(ids, max_new_tokens=a.max_new_tokens,
                              temperature=0.85, top_k=50, top_p=0.95, do_sample=mode=="sampled", eos_token_id=tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-started
                    completion = answer[0, ids.shape[1]:].tolist()
                    eos = tokenizer.eos_token_id in completion
                    if eos:
                        completion = completion[:completion.index(tokenizer.eos_token_id)]
                    decoded = tokenizer.decode(completion, skip_special_tokens=True)
                    final_answer = decoded.split("</think>",1)[-1].strip()
                    row = {"prompt_id":index,"prompt":prompt,"expected_short_answer":expected,
                           "mode":mode,"seed":seed,"answer":decoded,"eos":eos,
                           "thinking":a.thinking,"token_ids":completion,"stop_reason":"eos" if eos else "token_budget",
                           "completion_tokens":len(completion),"elapsed_seconds":elapsed,
                           "distinct_2": None if len(completion)<2 else 1-repetition(completion,2),
                           "repeat_4":repetition(completion,4),
                           "strict_exact_match": final_answer==expected if expected is not None else None}
                    rows.append(row)
                    out.write(json.dumps(row,ensure_ascii=False)+"\n"); out.flush()
                    print(f"Text diagnostic {mode} seed={seed} prompt={index+1}/{len(PROMPTS)} eos={eos}",flush=True)
    groups = {}
    for mode in MODES:
        group = [r for r in rows if r["mode"]==mode]
        values = {"records":len(group)}
        for key in ("eos","distinct_2","repeat_4","strict_exact_match","elapsed_seconds","completion_tokens"):
            valid = [r[key] for r in group if r[key] is not None]
            values[key] = sum(valid)/len(valid) if valid else None
            values[key+"_n"] = len(valid)
        groups[mode] = values
    expected_records = len(PROMPTS) * len(a.seeds)
    if len(rows) != expected_records:
        raise ValueError("Incomplete or unexpected diagnostic record count")
    validate_saved_rows(rows, a.seeds, a.thinking, complete=True)
    result = {**contract,"status":"completed","records_sha256":sha256(dest / "records.jsonl"),"diagnostic_only":True,
              "not_standard_benchmark":True,"not_human_scored":True,
              "prompt_set":PROMPTS,"seeds":a.seeds,"groups":groups,
              "per_seed": {str(seed): {k: valid_mean([r for r in rows if r["mode"]=="sampled" and r["seed"]==seed],k)
                           for k in ("eos","distinct_2","repeat_4")} for seed in a.seeds},
              "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
              "definitions":{"distinct_2":"macro unique token bigrams / all token bigrams",
                 "repeat_4":"macro 1 - unique token 4grams / all token 4grams; EOS excluded",
                 "strict_exact_match":"No authored short-answer targets; upstream open-ended prompt examples only",
                 "elapsed_seconds":"synchronized end-to-end generate including prefill, not decode-only latency"}}
    atomic_json(dest / "summary.json", result)

if __name__ == "__main__":
    main()
