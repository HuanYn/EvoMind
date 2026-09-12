"""Admit existing native text evidence without retraining or re-evaluating.

An explicit import spec pins checkpoint hashes and published evaluation contracts.
Remote paths are never executed or trusted as local paths. Original evidence is
kept byte-for-byte; hardware/harness differences remain visible in the receipt.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import evomind_product as product
from evomind_harness_eval import verify_result
from evomind_run import atomic_json, exclusive_run_lock, file_record, now, sha256

ROOT = Path(__file__).resolve().parents[1]


def local(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Import path outside project: {value}")
    return path


def checked(record):
    path = local(record['path'])
    if not path.is_file() or sha256(path) != record['sha256']:
        raise ValueError(f"Missing or changed evidence: {path}")
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def original_contract(path, published):
    """Only a documented dataset archive location may differ after relocation."""
    actual = read(path)
    if sha256(path) != published['original_summary_sha256']:
        raise ValueError('Original benchmark summary bytes changed')
    expected = published['contract']
    normalized = copy.deepcopy(actual['contract'])
    source = normalized.get('dataset_loading', {})
    if 'archive' in source:
        expected_path = expected['dataset_loading']['archive']
        if not source['archive'].replace('\\', '/').endswith('/' + expected_path) and source['archive'] != expected_path:
            raise ValueError('Unexpected archive relocation')
        if sha256(local(expected_path)) != source['source']['sha256']:
            raise ValueError('Original benchmark dataset archive changed')
        source['archive'] = expected_path
    if normalized != expected or actual['results_sha256'] != published['original_results_sha256']:
        raise ValueError('Original benchmark contract/result hash changed')
    return actual['contract']


def verify_bundle(spec):
    snapshots = {name: read(checked(item)) for name, item in spec['snapshots'].items()}
    source_paths = {name: checked(item) for name, item in spec['archived_sources'].items()}
    policy = product.settings()['selection']
    if set(spec['models']) != {'full_sft', 'dpo', 'cispo', 'grpo'}:
        raise ValueError('Require SFT, DPO and both full RL branches')
    evidence, scores, candidates = [], {}, {}

    def retain(path):
        item = file_record(path)
        item['path'] = str(Path(path).relative_to(ROOT))
        evidence.append(item)

    for name, entry in spec['models'].items():
        weight = checked(entry['checkpoint'])
        digest = sha256(weight)
        candidates[name] = {'path': str(weight), 'sha256': digest}
        published = snapshots['benchmarks']['models'][name]
        if published['checkpoint_sha256'] != digest:
            raise ValueError('Checkpoint differs from published evidence')
        receipt_path = checked(entry['training_receipt'])
        receipt = read(receipt_path)
        if name == 'full_sft':
            row = receipt['stages']['full_sft']
            if row['status'] != 'completed' or row['exit_code'] != 0 or row['output']['sha256'] != digest:
                raise ValueError('SFT training is incomplete')
        else:
            if receipt.get('status') not in ('complete', 'completed') or receipt.get('stop_reason') != 'epochs_complete' or receipt.get('probe_only') is not False:
                raise ValueError('A probe or incomplete run cannot be selected')
            expected_steps = 4292 if name == 'dpo' else 9751
            if receipt['optimizer_updates'] != expected_steps:
                raise ValueError('Full epoch update budget not satisfied')
            if name == 'dpo' and (receipt.get('export_sha256') != digest or
                    any(receipt['contract']['inputs'][key]['sha256'] != spec['models']['full_sft']['checkpoint']['sha256']
                        for key in ('policy_base', 'reference'))):
                raise ValueError('DPO export or SFT parent changed')
        retain(receipt_path)
        if name in ('cispo', 'grpo'):
            contract_path = checked(entry['training_contract'])
            contract = read(contract_path)
            contract_hash = hashlib.sha256(json.dumps(contract['contract'], sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
            if contract_hash != receipt['contract_sha256']:
                raise ValueError('RL training contract digest changed')
            if contract['contract']['initial']['sha256'] != spec['models']['dpo']['checkpoint']['sha256']:
                raise ValueError('RL parent is not the selected DPO')
            if contract['contract']['settings']['loss_type'] != name:
                raise ValueError('RL algorithm differs')
            if receipt['configured_epochs'] != 1 or contract['contract']['settings']['epochs'] != 1:
                raise ValueError('RL epoch budget differs')
            retain(contract_path)
        root = local(entry['evaluation_dir'])
        benchmarks = {}
        if set(published['tasks']) != set(product.TASKS):
            raise ValueError('Require all seven benchmark contracts')
        for task in product.TASKS:
            path = local(root / entry['harness_pattern'].format(task=task) / 'summary.json')
            expected = original_contract(path, published['tasks'][task])
            if expected['checkpoint_sha256'] != digest:
                raise ValueError('Benchmark checkpoint mismatch')
            # Pin recorded original contract, not this host's installed harness.
            report = verify_result(path, expected)
            benchmarks[task] = product.benchmark_score(report, task)
            retain(path)
            retain(path.parent / 'results.json')
        modes = {}
        for mode in ('thinking_off', 'thinking_on'):
            summary_path = local(root / mode / 'summary.json')
            report = read(summary_path)
            published_diag = snapshots['diagnostics']['models'][name]['text'][mode]
            from evomind_text_eval import PROMPTS
            required = {'status': 'completed', 'checkpoint_sha256': digest, 'lora_sha256': None,
                        'thinking': mode == 'thinking_on', 'seeds': [42], 'max_new_tokens': 8192,
                        'sampling': {'temperature': .85, 'top_k': 50, 'top_p': .95},
                        'prompts': json.loads(json.dumps(PROMPTS, ensure_ascii=False)),
                        'evaluator_sha256': sha256(ROOT / 'scripts/evomind_text_eval.py'),
                        'upstream_eval_sha256': sha256(ROOT / 'eval_llm.py'),
                        'loader_sha256': sha256(source_paths['scripts/evomind_load_text_model.py']),
                        'tokenizer_sha256': {p.name: sha256(p) for p in sorted((ROOT / 'model').glob('*.json'))},
                        'records_sha256': published_diag['records_sha256']}
            if any(report.get(k) != v for k, v in required.items()):
                differing = [k for k, v in required.items() if report.get(k) != v]
                raise ValueError(f'Diagnostic protocol changed: {name}/{mode}: {differing}')
            if sha256(summary_path.parent / 'records.jsonl') != report['records_sha256']:
                raise ValueError('Raw diagnostic output changed')
            rows = product.read_lines(summary_path.parent / 'records.jsonl')
            identities = {(r['seed'], r['prompt_id']) for r in rows}
            if len(rows) != 8 or identities != {(42, i) for i in range(8)}:
                raise ValueError('Missing or duplicated diagnostic examples')
            modes[mode] = {'eos': sum(bool(r['eos']) for r in rows) / 8,
                           'repeat4': sum(product.metric(r['repeat_4']) for r in rows) / 8}
            retain(summary_path)
            retain(summary_path.parent / 'records.jsonl')
        tool_path = local(entry['tool_result'])
        if sha256(tool_path) != snapshots['diagnostics']['models'][name]['tool']['raw_sha256']:
            raise ValueError('Original ToolCall output changed')
        tool = read(tool_path)
        expected_protocol = {'seed': 42, 'max_turns': 3, 'max_new_tokens': 512,
                             'backend': 'local', 'temperature': .9, 'top_p': .9}
        if tool.get('status') != 'complete' or len(tool.get('cases', [])) != 8 or any(tool.get(k) != v for k, v in expected_protocol.items()):
            raise ValueError('Incomplete/incompatible tool diagnostic')
        if set(tool.get('provenance', {})) != {'scripts/eval_toolcall.py', 'trainer/evomind_tools.py',
                'scripts/evomind_load_text_model.py', 'model/model_minimind.py', 'model/model_lora.py',
                'checkpoint', 'tokenizer', 'tokenizer_config'}:
            raise ValueError('Missing/extra ToolCall provenance')
        for key, record in tool['provenance'].items():
            path = weight if key == 'checkpoint' else source_paths.get(key, ROOT / {'tokenizer': 'model/tokenizer.json', 'tokenizer_config': 'model/tokenizer_config.json'}.get(key, key))
            if not path.is_file() or sha256(path) != record['sha256']:
                raise ValueError(f'Tool provenance mismatch: {key}')
        retain(tool_path)
        scores[name] = {'benchmarks': benchmarks, **modes['thinking_off'], 'thinking_on': modes['thinking_on']}
    if not product.baseline_ready(scores['full_sft'], policy):
        raise ValueError('SFT baseline fails screening')
    selected, decisions = 'full_sft', {}
    for name in ('dpo', 'cispo'):
        if name == 'cispo' and selected != 'dpo':
            raise ValueError('Recorded CISPO parent is DPO, but selection retained SFT')
        decision = product.compare(scores[selected], scores[name], policy)
        decision.update(parent=selected, candidate=name)
        if decision['promote']:
            selected = name
        decision['selected_name'] = selected
        decisions[name] = decision
    decisions['grpo_comparison'] = product.compare(scores['dpo'], scores['grpo'], policy)
    decisions['grpo_comparison']['scope'] = 'same-DPO comparison only; no automatic mainline replacement'
    return candidates, scores, decisions, selected, evidence


def make_spec(destination):
    """Materialize an explicit mapping for the preserved September text run."""
    def record(value):
        path = local(value)
        return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
    spec = {'snapshots': {'benchmarks': record('docs/results/text_benchmarks_20260911.json'),
                          'diagnostics': record('docs/results/text_diagnostics_20260911.json')},
            'archived_sources': {'scripts/evomind_load_text_model.py': record('artifacts/handoff_20260912/eval_source/recorded_remote_loader.py')},
            'models': {}, 'limitations': [
                'Single training and sampling seed; screening is not statistical significance.',
                'Local and remote harness source hashes differ; original contracts preserved.',
                'Eight thinking-off prompts decide degeneration screen; thinking-on kept separate.',
                'ToolCall completion is not accuracy; failures preserved; no visual transfer evidence.']}
    spec['models']['full_sft'] = {
        'checkpoint': record('artifacts/runs/text_official_mini_20260908/snapshots/full_sft/full_sft_768.pth'),
        'training_receipt': record('artifacts/runs/text_official_mini_20260908/state.json'),
        'evaluation_dir': 'artifacts/evaluation/text_product_20260909/full_sft',
        'harness_pattern': 'harness_{task}_v2',
        'tool_result': 'artifacts/evaluation/text_product_20260909/full_sft/tool_seed42/results.json'}
    spec['models']['dpo'] = {
        'checkpoint': record('artifacts/runs/text_product_20260909/dpo/full/weights/dpo_768.pth'),
        'training_receipt': record('artifacts/runs/text_product_20260909/dpo/full/weights/dpo_768.runtime.json'),
        'evaluation_dir': 'artifacts/evaluation/text_product_20260909/dpo',
        'harness_pattern': 'harness_{task}_v2',
        'tool_result': 'artifacts/evaluation/text_product_20260909/dpo/tool_seed42/results.json'}
    for name in ('cispo', 'grpo'):
        folder = f'artifacts/handoff_20260912/final_{name}_20260911'
        spec['models'][name] = {
            'checkpoint': record(f'artifacts/handoff_20260912/{name}_768.pth'),
            'training_receipt': record(f'artifacts/remote_evidence_20260911/{name}/training_summary.json'),
            'training_contract': record(f'artifacts/remote_evidence_20260911/{name}/training_contract.json'),
            'evaluation_dir': folder, 'harness_pattern': 'harness/{task}',
            'tool_result': f'{folder}/toolcall_resume.json'}
    destination = local(destination)
    if destination.exists() and read(destination) != spec:
        raise FileExistsError('Do not replace a different import mapping')
    atomic_json(destination, spec)


def verify_imported_receipt(receipt):
    spec_path = checked(receipt['import_spec'])
    candidates, scores, decisions, selected, evidence = verify_bundle(read(spec_path))
    if (receipt['selected_name'] != selected or receipt['selected'] != candidates[selected]
            or receipt['decisions'] != decisions or receipt['scores'] != scores
            or receipt['evaluation_evidence'] != evidence):
        raise ValueError('Imported selection no longer matches raw evidence')
    return Path(candidates[selected]['path'])


def accept(spec_path):
    spec_path = local(spec_path)
    candidates, scores, decisions, selected, evidence = verify_bundle(read(spec_path))
    run = ROOT / product.settings()['run_dir']
    with exclusive_run_lock(run):
        path = run / 'state.json'
        state = read(path)
        if state['config_sha256'] != sha256(product.CONFIG):
            raise ValueError('Historical product contract changed')
        backup = run / 'state.before_final_acceptance_20260912.json'
        if not backup.exists():
            atomic_json(backup, copy.deepcopy(state))
        state['branches']['cispo'].update(status='completed', output=candidates['cispo'],
            completion_import=str(spec_path), completed_at=now())
        state['selections'].update(decisions)
        state.update(status='text_selected', selected_name=selected, completed_at=now(),
                     scores=scores, evaluation_import=str(spec_path))
        atomic_json(path, state)
        record = {'status': 'accepted_text_screening', 'verification_mode': 'imported_native_v1',
                  'config_sha256': sha256(product.CONFIG), 'state_sha256': sha256(path),
                  'selected_name': selected, 'selected': candidates[selected],
                  'import_spec': file_record(spec_path), 'evaluation_evidence': evidence,
                  'scores': scores, 'decisions': decisions, 'accepted_at': now(),
                  'human_review': 'cancelled_by_user', 'video_quality': 'not_evaluated',
                  'claim': 'Engineering initialization selection, not statistical superiority or visual quality.',
                  'limitations': read(spec_path)['limitations']}
        atomic_json(run / 'product_acceptance.json', record)
    product.enable_vision()
    print(json.dumps({'selected': selected, 'sha256': candidates[selected]['sha256'], 'vision_enabled': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--make-spec', action='store_true', help='Write the explicit preserved-run path/hash mapping, no acceptance')
    args = parser.parse_args()
    if args.make_spec:
        make_spec(args.spec)
    elif args.check_only:
        _, _, decisions, selected, _ = verify_bundle(read(local(args.spec)))
        print(json.dumps({'selected': selected, 'decisions': decisions}, indent=2))
    else:
        accept(args.spec)
