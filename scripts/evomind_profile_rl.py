"""Isolated RL timing and optimization experiments, never imported by training.

Runs only on an explicitly bound, free GPU. Reads a trusted optimizer-boundary
snapshot, but never writes to the formal run. No baseline quality re-evaluation.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evomind_run import atomic_json, sha256, now


def reward_messages(messages, answer):
    history = '\n'.join(f"{m['role']}: {m['content']}" for m in messages[:-1])
    query = messages[-1]['content'] if messages else ''
    context = f'{history}\n以上是对话历史。我的新问题是：\n{query}' if history else query
    return [{'role': 'user', 'content': context}, {'role': 'assistant', 'content': answer}]


def normalize_scores(value, count):
    values = value if isinstance(value, list) else [value]
    if len(values) != count:
        raise ValueError('Reward batch size mismatch')
    import math
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite reward score')
    return values


def topk_nucleus(logits, k=50, p=0.85):
    """Exact dense-vocabulary support, sort only top-k when boundary has no ties.

    Keep dense multinomial draw/order to avoid changing RNG mapping. Fall back to
    native full sort when top-k threshold ties could retain more than k entries.
    This experimental path intentionally synchronizes once to detect ties.
    """
    import torch
    original = logits.clone()
    if k <= 0 or k >= logits.size(-1):
        chosen = None
    else:
        values, indices = torch.topk(logits, k)
        threshold = values[:, -1:]
        # Internal ties can also change nucleus boundary ordering, especially BF16.
        tied = ((logits >= threshold).sum(-1) > k) | (values[:, 1:] == values[:, :-1]).any(-1)
        chosen = None if bool(tied.any()) else (values, indices)
        original[original < threshold] = -float('inf')
    if p >= 1:
        return original
    if chosen is None:
        values, indices = torch.sort(original, descending=True)
    else:
        values, indices = chosen
    mask = torch.cumsum(torch.softmax(values, dim=-1), dim=-1) > p
    mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), False
    if chosen is None:
        original[mask.scatter(1, indices, mask)] = -float('inf')
        return original
    values = values.masked_fill(mask, -float('inf'))
    return torch.full_like(logits, -float('inf')).scatter(1, indices, values)


def fast_generate(model, input_ids, attention_mask, max_new_tokens, eos_token_id,
                  temperature=0.8, optimized_sampling=False, do_sample=True):
    import torch
    ids, mask = input_ids.clone(), attention_mask.clone()
    cache = None
    finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
    for _ in range(max_new_tokens):
        past = cache[0][0].shape[1] if cache else 0
        outputs = model(ids[:, past:], attention_mask=mask, past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
        logits = outputs.logits[:, -1, :] / temperature
        if optimized_sampling:
            logits = topk_nucleus(logits)
        else:
            logits[logits < torch_topk_threshold(logits)] = -float('inf')
            values, indices = torch.sort(logits, descending=True)
            removed = torch.cumsum(torch.softmax(values, dim=-1), dim=-1) > 0.85
            removed[..., 1:], removed[..., 0] = removed[..., :-1].clone(), 0
            logits[removed.scatter(1, indices, removed)] = -float('inf')
        token = torch.multinomial(torch.softmax(logits, dim=-1), 1) if do_sample else logits.argmax(-1, keepdim=True)
        token = torch.where(finished[:, None], torch.full_like(token, eos_token_id), token)
        ids = torch.cat([ids, token], dim=-1)
        mask = torch.cat([mask, mask.new_ones(mask.size(0), 1)], dim=-1)
        cache = outputs.past_key_values
        finished |= token.squeeze(-1).eq(eos_token_id)
        if bool(finished.all()):
            break
    return ids


def torch_topk_threshold(logits):
    import torch
    return torch.topk(logits, 50)[0][..., -1, None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--resume-checkpoint', required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--prompts', type=int, default=4)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    if a.prompts < 2 or a.prompts % 2 or a.repeats < 2:
        raise ValueError('Use an even prompt count >=2 and >=2 timing repeats')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu_uuid:
        raise ValueError('Explicit single GPU binding required')
    raw = subprocess.check_output(['nvidia-smi', '-i', a.gpu_uuid,
        '--query-gpu=memory.used,gpu_recovery_action', '--format=csv,noheader,nounits'], text=True)
    memory, recovery = map(str.strip, raw.strip().split(','))
    if int(memory) >= 500 or recovery != 'None':
        raise RuntimeError('Profiling requires an idle healthy GPU; never co-run with training')
    dest = Path(a.run_dir).resolve()
    if not dest.is_relative_to(ROOT / 'profile_output') or dest.exists():
        raise ValueError('Use a fresh dedicated project profile_output subdirectory')
    dest.mkdir(parents=True)
    atomic_json(dest / 'state.json', {'status': 'loading', 'at': now(), 'pid': os.getpid()})
    try:
        run(a, dest)
    except BaseException as exc:
        atomic_json(dest / 'state.json', {'status': 'failed', 'error': str(exc), 'at': now()})
        raise


def run(a, dest):
    import datasets  # Windows import order
    import importlib.util
    import torch
    from transformers import AutoTokenizer
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer import evomind_rl_runtime as rt
    from trainer import rollout_engine as rollout
    from trainer import train_grpo as upstream
    from dataset.lm_dataset import RLAIFDataset

    with Path(a.resume_checkpoint).open('rb') as stream:
        import hashlib
        source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu', weights_only=False)
    contract = saved['contract']
    expected = hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if saved['contract_sha256'] != expected or saved['format'] != rt.FORMAT:
        raise ValueError('Checkpoint format/contract integrity mismatch')
    if saved['probe_only'] or contract['settings']['loss_type'] not in ('grpo', 'cispo'):
        raise ValueError('Expected trusted formal RL checkpoint')
    for item in (contract['initial'], contract['data'], contract['runtime_source'], contract['upstream_source'],
                 *contract['tokenizer'].values(), *contract['reward_model'].values()):
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Formal input/source hash mismatch')
    args = SimpleNamespace(**contract['settings'])
    upstream.args = args
    rt.seed_rng(42, 'cuda')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    cfg = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    actor = MiniMindForCausalLM(cfg)
    actor.load_state_dict(saved['models']['actor'], strict=True)
    actor = actor.cuda().train()
    reference = MiniMindForCausalLM(cfg)
    reference.load_state_dict(torch.load(contract['initial']['path'], map_location='cpu', weights_only=True), strict=True)
    reference = reference.cuda().eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=3e-7)
    optimizer.load_state_dict(saved['optimizers']['actor'])
    atomic_json(dest / 'contract.json', {'created_at': now(), 'argv': sys.argv,
        'source_checkpoint_sha256': source_hash, 'source_update': saved['optimizer_updates'],
        'formal_contract': contract, 'profiler_sha256': sha256(__file__), 'gpu_uuid': a.gpu_uuid,
        'torch': torch.__version__, 'purpose': 'isolated performance replay, NOT quality evaluation or formal continuation',
        'available_engine_packages': {x: bool(importlib.util.find_spec(x)) for x in ('sglang', 'vllm', 'triton')},
        'gpu_name': torch.cuda.get_device_name(), 'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
        'host_threads': torch.get_num_threads(), 'pid': os.getpid(),
        'scope': '4 deterministic real training prompts by default; not representative of every length/task',
        'numerical_gate': 'report raw/clipped score error, within-group rank/advantage sign changes; no automatic deployment'})
    del saved
    reward = rt.LocalRewardModel(args.reward_model_path, 'cuda', torch.float32)
    dataset = RLAIFDataset(args.data_path, tokenizer, max_length=1792, thinking_ratio=args.thinking_ratio)
    rt.seed_rng(42, 'cuda')
    order = torch.randperm(len(dataset)).tolist()[:a.prompts]
    prompts = [dataset[i]['prompt'] for i in order]
    atomic_json(dest / 'prompts.json', [{'source_index': i, 'prompt': q} for i, q in zip(order, prompts)])
    measurements, cases = [], []

    def timer(label, fn, case=None):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        row = {'stage': label, 'case': case, 'seconds': time.perf_counter()-start,
            'allocated_bytes': torch.cuda.memory_allocated(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        measurements.append(row)
        with (dest / 'timings.jsonl').open('a') as out:
            out.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        return value

    def get_inputs(prompt):
        batch = tokenizer([prompt], return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False)
        return batch['input_ids'][:, -768:].cuda(), batch['attention_mask'][:, -768:].cuda()

    # Warm up the real generate path without changing the retained measurements.
    ids, mask = get_inputs(prompts[0])
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        warm_ids = actor.generate(input_ids=ids.repeat_interleave(6, 0), attention_mask=mask.repeat_interleave(6, 0),
                       max_new_tokens=16, temperature=0.8, eos_token_id=tokenizer.eos_token_id).clone()
        warm_mask = (warm_ids != tokenizer.pad_token_id).long()
        rollout.compute_per_token_logps(actor, warm_ids, warm_ids.size(1)-ids.size(1), warm_mask)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        warm_loss = actor(warm_ids, attention_mask=warm_mask, logits_to_keep=1).logits.float().mean()
    warm_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        reference(warm_ids, attention_mask=warm_mask, logits_to_keep=1)
        upstream.calculate_rewards([prompts[0]], tokenizer.batch_decode(warm_ids[:,ids.size(1):],skip_special_tokens=True), reward)
    del warm_loss, warm_ids, warm_mask
    torch.cuda.reset_peak_memory_stats()
    for i, prompt in enumerate(prompts):
        ids, mask = get_inputs(prompt)
        rt.seed_rng(4200+i, 'cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = timer('generation', lambda: actor.generate(input_ids=ids.repeat_interleave(6, 0),
                attention_mask=mask.repeat_interleave(6, 0), max_new_tokens=1024, temperature=0.8,
                eos_token_id=tokenizer.eos_token_id).clone(), i)
            keep = outputs.size(1)-ids.size(1)
            full_mask = (outputs != tokenizer.pad_token_id).long()
            old = timer('old_logprobs', lambda: rollout.compute_per_token_logps(actor, outputs, keep, full_mask), i)
        completion = outputs[:, ids.size(1):]
        texts = tokenizer.batch_decode(completion, skip_special_tokens=True)
        result = rollout.RolloutResult(outputs, completion, old, texts,
            ids.new_full((6,), ids.size(1)), completion.new_ones(completion.shape))
        state = rt.completion_state(result, tokenizer, 'cuda')
        rewards = timer('reward_serial_fp32', lambda: upstream.calculate_rewards([prompt], texts, reward), i)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logps, _ = timer('policy_forward', lambda: rt.masked_completion_logps(actor, state['outputs'], state['full_mask'], state['positions']), i)
        with torch.no_grad():
            ref, _ = timer('reference_forward', lambda: rt.masked_completion_logps(reference, state['outputs'], state['full_mask'], state['positions']), i)
        loss, _ = timer('objective', lambda: rt.grpo_objective(logps, state['old_logps'], ref,
            rewards, state['mask'], generations=6, beta=args.beta, loss_type=args.loss_type,
            epsilon=args.epsilon, epsilon_high=args.epsilon_high), i)
        timer('backward', lambda: (loss/2).backward(), i)
        if (i+1) % 2 == 0:
            def update():
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            timer('optimizer_update', update, i)
        cases.append({'prompt': prompt, 'texts': texts, 'completion_tokens': state['mask'].sum(1).tolist()})
        atomic_json(dest / 'cases.json', cases)
        del result, state, old, outputs, completion, logps, ref, rewards, loss
    del optimizer, reference
    gc.collect(); torch.cuda.empty_cache()

    # Hold answers fixed when comparing RM precision/batching. Use the upstream
    # rule code to collect EXACT messages/answer extraction instead of rewriting it.
    pairs, rules = [], []
    class Recorder:
        def get_score(self, messages, answer):
            pairs.append(reward_messages(messages, answer))
            return 0.0
    for case in cases:
        rules.extend(upstream.calculate_rewards([case['prompt']], case['texts'], Recorder()).tolist())
    atomic_json(dest / 'reward_inputs.json', {'pairs': pairs, 'rule_rewards': rules})
    reward_results = {}
    for precision in ('float32', 'bfloat16'):
        if precision == 'bfloat16':
            reward.model = reward.model.to(dtype=torch.bfloat16)
        for batch in (1, 2, 3, 6):
            label = f'rm_{precision}_batch{batch}'
            def score_all():
                values = []
                for start in range(0, len(pairs), batch):
                    part = pairs[start:start+batch]
                    raw = (reward.model.get_score(reward.tokenizer, part[0]) if batch == 1
                        else reward.model.get_scores(reward.tokenizer, part))
                    values.extend(normalize_scores(raw, len(part)))
                return values
            try:
                score_all()  # untimed warm-up per path
                torch.cuda.reset_peak_memory_stats()
                runs = [timer(label, score_all, j) for j in range(a.repeats)]
                raw = runs[-1]
                clipped = [max(-3.0, min(3.0, s)) for s in raw]
                total = [r+s for r, s in zip(rules, clipped)]
                reward_results[label] = {'raw': raw, 'clipped': clipped, 'total': total,
                    'repeat_max_abs_error': max(abs(x-y) for run in runs for x,y in zip(run,raw)),
                    'median_seconds': statistics.median(r['seconds'] for r in measurements if r['stage']==label)}
            except torch.cuda.OutOfMemoryError as exc:
                reward_results[label] = {'status': 'oom', 'error': str(exc)}
                gc.collect(); torch.cuda.empty_cache()
            atomic_json(dest / 'reward_results.json', reward_results)
    baseline = reward_results['rm_float32_batch1']
    for value in reward_results.values():
        if 'raw' not in value:
            continue
        value['raw_max_abs_error'] = max(abs(x-y) for x,y in zip(value['raw'], baseline['raw']))
        value['clipped_max_abs_error'] = max(abs(x-y) for x,y in zip(value['clipped'], baseline['clipped']))
        flips = sign_flips = 0
        for start in range(0, len(pairs), 6):
            b, c = baseline['total'][start:start+6], value['total'][start:start+6]
            bm, cm = statistics.mean(b), statistics.mean(c)
            sign_flips += sum((x-bm)*(y-cm)<0 for x,y in zip(b,c))
            for j in range(6):
                for k in range(j):
                    flips += (b[j]-b[k])*(c[j]-c[k]) < 0
        value['group_pair_rank_flips'] = flips
        value['advantage_sign_flips'] = sign_flips
        value['speedup_vs_serial_fp32'] = baseline['median_seconds']/value['median_seconds']
    atomic_json(dest / 'reward_results.json', reward_results)
    del reward
    gc.collect(); torch.cuda.empty_cache()

    # In-process native engine acceleration, no new inference service/dependency.
    # Same weights/input/RNG/temperature/top-k/top-p, record mismatches explicitly.
    engine_results = []
    actor.eval()
    for i, prompt in enumerate(prompts):
        ids, mask = get_inputs(prompt)
        ids, mask = ids.repeat_interleave(6, 0), mask.repeat_interleave(6, 0)
        native = None
        for mode in ('native', 'last_logit_only', 'topk_nucleus'):
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                def generate(limit=1024):
                    if mode in ('native', 'last_logit_only'):
                        extra = {'logits_to_keep': 1} if mode == 'last_logit_only' else {}
                        return actor.generate(input_ids=ids, attention_mask=mask, max_new_tokens=limit,
                            temperature=0.8, eos_token_id=tokenizer.eos_token_id, **extra)
                    return fast_generate(actor, ids, mask, limit, tokenizer.eos_token_id,
                        optimized_sampling=mode=='topk_nucleus')
                generate(16)
                repeat_ids = []
                for repeat in range(a.repeats):
                    rt.seed_rng(8200+i, 'cuda')
                    output = timer('engine_'+mode, generate, {'prompt': i, 'repeat': repeat})
                    repeat_ids.append(output.cpu())
            if native is None:
                native = output.cpu()
            identical = native.shape == output.shape and torch.equal(native, output.cpu())
            engine_results.append({'case': i, 'mode': mode, 'identical_token_ids': identical,
                'repeat_identical': all(torch.equal(repeat_ids[0], x) for x in repeat_ids),
                'prompt_length': ids.size(1), 'decode_iterations': output.size(1)-ids.size(1),
                'output_ids': output.cpu().tolist()})
            atomic_json(dest / 'engine_results.json', engine_results)
    atomic_json(dest / 'summary.json', {'status': 'completed', 'timings': measurements,
        'reward_results': reward_results,
        'engine_results': [{k:v for k,v in r.items() if k!='output_ids'} for r in engine_results],
        'limits': ['Isolated replay, not formal training throughput or quality evidence',
                   'Small real-prompt sample; no automatic precision/backend adoption',
                   'Phase timing includes synchronization; initialization/checkpoint I/O excluded',
                   'Generation path tested native last-logit projection and sampler only; SGLang/vLLM not integrated'],
        'created_files': [p.name for p in dest.iterdir()], 'modified_training_files': []})
    atomic_json(dest / 'state.json', {'status': 'completed', 'at': now()})


if __name__ == '__main__':
    main()
