"""CPU-only admission checks using tiny synthetic evidence in temporary folders.

These tests never load model weights, evaluate a model, or touch real acceptance
state. The native harness verifier remains real so raw coverage is exercised.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import evomind_accept_text as admission
import evomind_product as product
from evomind_text_eval import PROMPTS
from eval_toolcall import TEST_CASES


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


class TextAcceptanceImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / 'project'
        self.root.mkdir()
        self.plan = copy.deepcopy(product.settings())
        self.plan['run_dir'] = 'run'
        self.plan['eval_dir'] = 'evaluation'
        self.config = self.write_json('product.json', self.plan)
        for target, value in ((admission, self.root), (product, self.root)):
            patcher = patch.object(target, 'ROOT', value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(product, 'CONFIG', self.config)
        patcher.start()
        self.addCleanup(patcher.stop)

        # Source bytes are independent of installed libraries and frontend code.
        self.sources = ['scripts/evomind_text_eval.py', 'eval_llm.py',
                        'scripts/evomind_load_text_model.py', 'scripts/eval_toolcall.py',
                        'trainer/evomind_tools.py', 'model/model_minimind.py',
                        'model/model_lora.py', 'model/tokenizer.json',
                        'model/tokenizer_config.json', 'archive/recorded_loader.py']
        for source in self.sources:
            self.write_bytes(source, source.encode())
        self.write_bytes('dataset/archive.zip', b'fixed synthetic dataset archive')
        self.spec = {'snapshots': {}, 'archived_sources': {
            'scripts/evomind_load_text_model.py': self.record('archive/recorded_loader.py')},
            'models': {}, 'limitations': ['Synthetic test evidence only.']}
        self.benchmarks = {'models': {}}
        self.diagnostics = {'models': {}}
        for name in ('full_sft', 'dpo', 'cispo', 'grpo'):
            self.write_bytes(f'weights/{name}.pth', name.encode())
        for name in ('full_sft', 'dpo', 'cispo', 'grpo'):
            self.build_model(name)
        self.pin_snapshots()

    def write_bytes(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def write_json(self, relative, value):
        return self.write_bytes(relative, json.dumps(value, ensure_ascii=False).encode())

    def record(self, relative):
        path = self.root / relative
        return {'path': str(path.relative_to(self.root)), 'sha256': admission.sha256(path)}

    def pin_snapshots(self):
        for name, value in (('benchmarks', self.benchmarks), ('diagnostics', self.diagnostics)):
            self.write_json(f'{name}.json', value)
            self.spec['snapshots'][name] = self.record(f'{name}.json')

    def build_model(self, name):
        checkpoint = self.record(f'weights/{name}.pth')
        digest = checkpoint['sha256']
        base = self.record('weights/full_sft.pth')
        dpo = self.record('weights/dpo.pth')
        receipt = {'status': 'completed', 'stop_reason': 'epochs_complete',
                   'probe_only': False, 'optimizer_updates': 4292 if name == 'dpo' else 9751,
                   'export_sha256': digest, 'optimizer_boundary': True,
                   'checkpoint_optimizer_update': 9751, 'algorithm': 'grpo', 'loss_type': name,
                   'configured_epochs': 1}
        if name == 'full_sft':
            receipt = {'stages': {'full_sft': {'status': 'completed', 'exit_code': 0,
                                              'output': checkpoint}}}
        elif name == 'dpo':
            receipt['contract'] = {'branch': 'dpo', 'parameters': {'from_weight': 'full_sft'},
                                   'inputs': {'policy_base': base, 'reference': base}}
        entry = {'checkpoint': checkpoint, 'evaluation_dir': f'evaluation/{name}',
                 'harness_pattern': 'harness/{task}',
                 'tool_result': f'evaluation/{name}/tool.json'}
        if name in ('cispo', 'grpo'):
            contract = {'contract': {'initial': dpo, 'settings': {'loss_type': name,
                                         'from_weight': 'dpo', 'epochs': 1}}}
            self.write_json(f'training/{name}.contract.json', contract)
            entry['training_contract'] = self.record(f'training/{name}.contract.json')
            receipt['contract_sha256'] = canonical_hash(contract['contract'])
        self.write_json(f'training/{name}.json', receipt)
        entry['training_receipt'] = self.record(f'training/{name}.json')
        self.spec['models'][name] = entry
        published = {'checkpoint_sha256': digest, 'tasks': {}}
        self.benchmarks['models'][name] = published
        accuracy = .5 if name == 'full_sft' else .52
        for task in product.TASKS:
            folder = f'evaluation/{name}/harness/{task}'
            raw = {'results': {task: {'acc,none': accuracy}},
                   'samples': {task: [{'doc_id': 0}, {'doc_id': 1}]},
                   'n-samples': {task: {'original': 2, 'effective': 2}},
                   'config': {'limit': None}}
            result = self.write_json(folder + '/results.json', raw)
            contract = {'checkpoint_sha256': digest, 'task': task, 'limit': None,
                        'dataset_loading': {'archive': 'dataset/archive.zip',
                            'source': {'sha256': admission.sha256(self.root / 'dataset/archive.zip')}}}
            actual_contract = copy.deepcopy(contract)
            actual_contract['dataset_loading']['archive'] = '/remote/project/dataset/archive.zip'
            summary = {'status': 'completed', 'contract': actual_contract,
                       'results_sha256': admission.sha256(result),
                       'scores': raw['results'], 'n_samples': raw['n-samples']}
            path = self.write_json(folder + '/summary.json', summary)
            published['tasks'][task] = {'contract': contract,
                'original_summary_sha256': admission.sha256(path),
                'original_results_sha256': admission.sha256(result)}
        self.diagnostics['models'][name] = {'text': {}}
        for mode in ('thinking_off', 'thinking_on'):
            folder = f'evaluation/{name}/{mode}'
            rows = [{'seed': 42, 'prompt_id': i, 'prompt': text,
                     'expected_short_answer': answer, 'mode': 'sampled',
                     'thinking': mode == 'thinking_on', 'eos': True, 'repeat_4': .1}
                    for i, (text, answer) in enumerate(PROMPTS)]
            path = self.write_bytes(folder + '/records.jsonl',
                                    ('\n'.join(json.dumps(r) for r in rows) + '\n').encode())
            summary = {'status': 'completed', 'checkpoint_sha256': digest, 'lora_sha256': None,
                       'thinking': mode == 'thinking_on', 'seeds': [42], 'max_new_tokens': 8192,
                       'sampling': {'temperature': .85, 'top_k': 50, 'top_p': .95},
                       'prompts': json.loads(json.dumps(PROMPTS)),
                       'evaluator_sha256': admission.sha256(self.root / 'scripts/evomind_text_eval.py'),
                       'upstream_eval_sha256': admission.sha256(self.root / 'eval_llm.py'),
                       'loader_sha256': admission.sha256(self.root / 'archive/recorded_loader.py'),
                       'tokenizer_sha256': {p.name: admission.sha256(p) for p in sorted((self.root / 'model').glob('*.json'))},
                       'records_sha256': admission.sha256(path)}
            self.write_json(folder + '/summary.json', summary)
            self.diagnostics['models'][name]['text'][mode] = copy.deepcopy(summary)
        provenance = {key: self.record(key) for key in self.sources
                      if key not in ('scripts/evomind_text_eval.py', 'eval_llm.py',
                                     'archive/recorded_loader.py', 'model/tokenizer.json',
                                     'model/tokenizer_config.json')}
        provenance.update(checkpoint=checkpoint, tokenizer=self.record('model/tokenizer.json'),
                          tokenizer_config=self.record('model/tokenizer_config.json'))
        provenance['scripts/evomind_load_text_model.py'] = self.record('archive/recorded_loader.py')
        tool = {'status': 'complete', 'seed': 42, 'max_turns': 3, 'max_new_tokens': 512,
                'backend': 'local', 'temperature': .9, 'top_p': .9, 'provenance': provenance,
                'cases': [{'prompt': case['prompt'], 'status': 'complete', 'unfinished': False,
                           'turns': []} for case in TEST_CASES]}
        path = self.write_json(entry['tool_result'], tool)
        self.diagnostics['models'][name]['tool'] = {'raw_sha256': admission.sha256(path)}

    def rewrite_receipt(self, name, change):
        relative = self.spec['models'][name]['training_receipt']['path']
        value = admission.read(self.root / relative)
        change(value)
        self.write_json(relative, value)
        self.spec['models'][name]['training_receipt'] = self.record(relative)

    def test_complete_bundle_is_read_only_and_uses_archived_loader(self):
        before = {str(p.relative_to(self.root)): admission.sha256(p)
                  for p in self.root.rglob('*') if p.is_file()}
        with patch.object(admission, 'atomic_json') as write:
            candidates, scores, decisions, selected, evidence = admission.verify_bundle(self.spec)
        self.assertEqual(selected, 'dpo')
        self.assertEqual(set(scores), {'full_sft', 'dpo', 'cispo', 'grpo'})
        self.assertFalse(decisions['cispo']['promote'])
        self.assertEqual(candidates[selected]['sha256'], self.record('weights/dpo.pth')['sha256'])
        self.assertGreater(len(evidence), 70)
        write.assert_not_called()
        self.assertEqual(before, {str(p.relative_to(self.root)): admission.sha256(p)
                                 for p in self.root.rglob('*') if p.is_file()})

    def test_paths_cannot_escape_project(self):
        for value in ('../outside.json', self.workspace / 'outside.json'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                admission.local(value)

    def test_checked_record_rejects_mutated_bytes(self):
        record = self.record('weights/dpo.pth')
        self.write_bytes(record['path'], b'mutated')
        with self.assertRaisesRegex(ValueError, 'changed evidence'):
            admission.checked(record)

    def test_harness_pattern_escape_is_rejected_before_read(self):
        self.spec['models']['full_sft']['harness_pattern'] = '../../../outside/{task}'
        with patch.object(admission, 'original_contract') as read_contract:
            with self.assertRaises(ValueError):
                admission.verify_bundle(self.spec)
            read_contract.assert_not_called()

    def test_original_summary_byte_pin_is_enforced(self):
        path = self.root / 'evaluation/full_sft/harness/piqa/summary.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'summary bytes changed'):
            admission.verify_bundle(self.spec)

    def test_changed_native_raw_hash_is_rejected_by_native_verifier(self):
        self.write_json('evaluation/full_sft/harness/piqa/results.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'Harness raw result changed'):
            admission.verify_bundle(self.spec)

    def test_rehashed_incomplete_native_coverage_is_still_rejected(self):
        folder = 'evaluation/full_sft/harness/piqa'
        raw = admission.read(self.root / folder / 'results.json')
        raw['samples']['piqa'].pop()
        result = self.write_json(folder + '/results.json', raw)
        summary = admission.read(self.root / folder / 'summary.json')
        summary['results_sha256'] = admission.sha256(result)
        path = self.write_json(folder + '/summary.json', summary)
        self.benchmarks['models']['full_sft']['tasks']['piqa'].update(
            original_summary_sha256=admission.sha256(path), original_results_sha256=admission.sha256(result))
        self.pin_snapshots()
        with self.assertRaisesRegex(ValueError, 'Missing/duplicate per-document'):
            admission.verify_bundle(self.spec)

    def test_archive_relocation_requires_pinned_bytes_and_identity(self):
        path = self.root / 'evaluation/full_sft/harness/piqa/summary.json'
        published = self.benchmarks['models']['full_sft']['tasks']['piqa']
        original = admission.original_contract(path, published)
        self.assertEqual(original['dataset_loading']['archive'], '/remote/project/dataset/archive.zip')
        self.write_bytes('dataset/archive.zip', b'changed dataset')
        with self.assertRaisesRegex(ValueError, 'dataset archive changed'):
            admission.original_contract(path, published)

    def test_unexpected_archive_basename_rejected_even_when_summary_repinned(self):
        path = self.root / 'evaluation/full_sft/harness/piqa/summary.json'
        published = copy.deepcopy(self.benchmarks['models']['full_sft']['tasks']['piqa'])
        report = admission.read(path)
        report['contract']['dataset_loading']['archive'] = '/remote/other.zip'
        self.write_json(path, report)
        published['original_summary_sha256'] = admission.sha256(path)
        with self.assertRaisesRegex(ValueError, 'archive relocation'):
            admission.original_contract(path, published)

    def test_probes_rejected_even_with_a_full_update_count(self):
        for name in ('dpo', 'cispo', 'grpo'):
            with self.subTest(name=name):
                self.rewrite_receipt(name, lambda value: value.update(probe_only=True))
                with self.assertRaisesRegex(ValueError, 'probe|incomplete'):
                    admission.verify_bundle(self.spec)
                self.rewrite_receipt(name, lambda value: value.update(probe_only=False))

    def test_wrong_dpo_parent_rejected(self):
        def wrong_parent(value):
            value['contract']['inputs']['policy_base']['sha256'] = '0' * 64
        self.rewrite_receipt('dpo', wrong_parent)
        with self.assertRaises(ValueError):
            admission.verify_bundle(self.spec)

    def test_wrong_dpo_export_hash_rejected(self):
        self.rewrite_receipt('dpo', lambda value: value.update(export_sha256='0' * 64))
        with self.assertRaises(ValueError):
            admission.verify_bundle(self.spec)

    def test_wrong_rl_parent_rejected_even_with_valid_contract_digest(self):
        entry = self.spec['models']['cispo']
        relative = entry['training_contract']['path']
        contract = admission.read(self.root / relative)
        contract['contract']['initial']['sha256'] = self.record('weights/full_sft.pth')['sha256']
        self.write_json(relative, contract)
        entry['training_contract'] = self.record(relative)
        self.rewrite_receipt('cispo', lambda value: value.update(contract_sha256=canonical_hash(contract['contract'])))
        with self.assertRaisesRegex(ValueError, 'parent|DPO'):
            admission.verify_bundle(self.spec)

    def test_tool_raw_bytes_are_pinned_to_published_snapshot(self):
        relative = self.spec['models']['dpo']['tool_result']
        tool = admission.read(self.root / relative)
        tool['cases'][0]['turns'] = [{'assistant': 'replacement response'}]
        self.write_json(relative, tool)
        with self.assertRaises(ValueError):
            admission.verify_bundle(self.spec)

    def test_empty_tool_provenance_rejected_even_when_raw_hash_repinned(self):
        relative = self.spec['models']['dpo']['tool_result']
        tool = admission.read(self.root / relative)
        tool['provenance'] = {}
        path = self.write_json(relative, tool)
        self.diagnostics['models']['dpo']['tool']['raw_sha256'] = admission.sha256(path)
        self.pin_snapshots()
        with self.assertRaises(ValueError):
            admission.verify_bundle(self.spec)

    def write_selection(self):
        candidates, scores, decisions, selected, evidence = admission.verify_bundle(self.spec)
        self.write_json('spec.json', self.spec)
        state = {'status': 'text_selected', 'config_sha256': admission.sha256(self.config),
                 'selected_name': selected, 'base': candidates['full_sft'], 'selections': decisions,
                 'branches': {'dpo': {'status': 'completed', 'initial': candidates['full_sft'],
                                     'output': candidates['dpo']},
                              'cispo': {'status': 'completed', 'initial': candidates['dpo'],
                                        'output': candidates['cispo']}}}
        path = self.write_json('run/state.json', state)
        receipt = {'status': 'accepted_text_screening', 'verification_mode': 'imported_native_v1',
                   'config_sha256': admission.sha256(self.config), 'state_sha256': admission.sha256(path),
                   'selected_name': selected, 'selected': candidates[selected],
                   'import_spec': self.record('spec.json'), 'scores': scores,
                   'decisions': decisions, 'evaluation_evidence': evidence}
        self.write_json('run/product_acceptance.json', receipt)
        return state, receipt

    def test_selected_checkpoint_rechecks_imported_raw_evidence(self):
        self.write_selection()
        self.assertEqual(product.selected_checkpoint(), self.root / 'weights/dpo.pth')
        self.write_bytes('evaluation/dpo/thinking_off/records.jsonl', b'altered raw output')
        with self.assertRaises(ValueError):
            product.selected_checkpoint()

    def test_receipt_cannot_change_selection_or_decisions(self):
        _, receipt = self.write_selection()
        receipt['decisions']['cispo']['promote'] = True
        with self.assertRaisesRegex(ValueError, 'no longer matches'):
            admission.verify_imported_receipt(receipt)

    def test_changed_state_hash_rejected_before_import_reverification(self):
        state, _ = self.write_selection()
        state['status'] = 'running'
        self.write_json('run/state.json', state)
        with patch.object(admission, 'verify_imported_receipt') as verify:
            with self.assertRaisesRegex(ValueError, 'state changed'):
                product.selected_checkpoint()
            verify.assert_not_called()

    def test_incomplete_branch_cannot_bypass_gate_with_a_repinned_state(self):
        state, receipt = self.write_selection()
        state['branches']['cispo']['status'] = 'training'
        path = self.write_json('run/state.json', state)
        receipt['state_sha256'] = admission.sha256(path)
        self.write_json('run/product_acceptance.json', receipt)
        with self.assertRaises((ValueError, RuntimeError)):
            product.selected_checkpoint()

    def test_acceptance_failure_never_writes_state_or_enables_vision(self):
        self.write_json('spec.json', self.spec)
        with patch.object(admission, 'verify_bundle', side_effect=ValueError('bad evidence')), \
                patch.object(admission, 'atomic_json') as write, \
                patch.object(product, 'enable_vision') as enable:
            with self.assertRaisesRegex(ValueError, 'bad evidence'):
                admission.accept('spec.json')
        write.assert_not_called()
        enable.assert_not_called()


if __name__ == '__main__':
    unittest.main()
