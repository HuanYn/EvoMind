"""Independent stdlib-only evaluation-gate fixtures; no real human/model scores.

All weights, answers and reviewer IDs below are explicitly synthetic test data.
Temporary artifacts stay inside evomind and are deleted after each fixture.
No evaluator implementation is patched, except its output root and salt source.
"""
from __future__ import annotations

import builtins
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
_original_import = builtins.__import__
_forbidden = {'torch', 'transformers', 'datasets', 'openai'}


def stdlib_guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in _forbidden:
        raise AssertionError(f'Stdlib-only fixture attempted to import {name}')
    return _original_import(name, *args, **kwargs)


with patch('builtins.__import__', side_effect=stdlib_guarded_import):
    import evomind_posttrain_evaluate as evaluation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class EvaluationGateFixture(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / 'artifacts/test_tmp'
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='evaluation_gate_', dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evaluation_root = self.root / 'evaluation'
        for fixture_patch in (
                patch.object(evaluation, 'EVAL', self.evaluation_root),
                patch.object(evaluation.secrets, 'token_hex', return_value='ab' * 32),
                patch('builtins.__import__', side_effect=stdlib_guarded_import)):
            fixture_patch.start()
            self.addCleanup(fixture_patch.stop)
        self.models = {'full_sft': {}, 'dpo': {}}
        self.state = {'models': {name: {'thinking_off': {'status': 'completed'}}
                                 for name in self.models}}
        self.base_rows = self.rows('reference response')
        self.branch_rows = self.rows('candidate response')
        self.write_diagnostic('full_sft', self.base_rows)
        self.write_diagnostic('dpo', self.branch_rows)

    @staticmethod
    def rows(answer_prefix):
        return [{'seed': seed, 'prompt_id': prompt, 'mode': 'sampled',
                 'prompt': f'Synthetic diagnostic question {prompt}',
                 'answer': f'{answer_prefix}; question={prompt}; seed={seed}'}
                for seed in (42, 123, 2026) for prompt in (0, 1, 2)]

    def write_diagnostic(self, name, rows, *, update_hash=True):
        directory = self.evaluation_root / name / 'thinking_off'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'records.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        if update_hash:
            write_json(directory / 'summary.json', {'status': 'completed', 'records_sha256': digest(path)})
        return path

    def package(self):
        return evaluation.write_blind_package(self.models, self.state)

    def task_content(self):
        return json.loads((self.evaluation_root / 'blind_review/tasks.json').read_text(encoding='utf-8'))

    def write_annotations(self, rows):
        path = self.evaluation_root / 'blind_review/annotations.csv'
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['pair_id', 'reviewer_id', 'preference', 'notes'])
            writer.writeheader()
            writer.writerows(rows)

    def annotations(self, reviewers=('fixture_reviewer_A', 'fixture_reviewer_B')):
        return [{'pair_id': pair['pair_id'], 'reviewer_id': reviewer, 'preference': 'tie',
                 'notes': 'SYNTHETIC UNIT TEST ANNOTATION; NOT A REAL REVIEW'}
                for pair in self.task_content()['pairs'] for reviewer in reviewers]


class BlindPackageTests(EvaluationGateFixture):
    def test_exactly_nine_unique_matched_pairs_and_blank_csv_is_pending(self):
        result = self.package()
        self.assertEqual(result['status'], 'awaiting_human_annotations')
        self.assertEqual((result['pairs'], result['valid_annotations']), (9, 0))
        tasks = self.task_content()
        self.assertEqual(tasks['required_reviewers_per_pair'], 2)
        self.assertEqual(len({pair['pair_id'] for pair in tasks['pairs']}), 9)
        self.assertTrue(all(set(pair) == {'pair_id', 'prompt', 'answer_A', 'answer_B'} for pair in tasks['pairs']))
        identity = json.loads((self.evaluation_root / 'blind_review/identity_key_DO_NOT_GIVE_REVIEWERS.json').read_text())
        self.assertEqual({(row['seed'], row['prompt_id']) for row in identity.values()},
                         {(seed, prompt) for seed in (42, 123, 2026) for prompt in (0, 1, 2)})
        self.assertEqual(set(identity), {pair['pair_id'] for pair in tasks['pairs']})
        before = (self.evaluation_root / 'blind_review/tasks.json').read_bytes()
        self.assertEqual(self.package()['status'], 'awaiting_human_annotations')
        self.assertEqual(before, (self.evaluation_root / 'blind_review/tasks.json').read_bytes())

    def test_missing_identity_rejected_even_with_matching_raw_hash(self):
        self.write_diagnostic('dpo', self.branch_rows[:-1])
        with self.assertRaisesRegex(ValueError, '9 distinct'):
            self.package()

    def test_duplicate_identity_rejected_even_when_total_is_nine(self):
        rows = copy.deepcopy(self.branch_rows)
        rows[-1] = copy.deepcopy(rows[0])
        self.write_diagnostic('dpo', rows)
        with self.assertRaisesRegex(ValueError, '9 distinct'):
            self.package()

    def test_wrong_seed_rejected_even_when_total_is_nine(self):
        rows = copy.deepcopy(self.branch_rows)
        rows[-1]['seed'] = 999
        self.write_diagnostic('dpo', rows)
        with self.assertRaisesRegex(ValueError, '9 distinct'):
            self.package()

    def test_raw_hash_mismatch_is_rejected_for_either_side(self):
        for name, rows in (('full_sft', self.base_rows), ('dpo', self.branch_rows)):
            with self.subTest(name=name):
                changed = copy.deepcopy(rows)
                changed[0]['answer'] += ' tampered'
                self.write_diagnostic(name, changed, update_hash=False)
                with self.assertRaisesRegex(ValueError, 'changed recorded artifact'):
                    self.package()
                self.write_diagnostic(name, rows)

    def test_same_identity_but_different_prompt_is_rejected(self):
        rows = copy.deepcopy(self.branch_rows)
        rows[0]['prompt'] = 'A different question under the same prompt ID'
        self.write_diagnostic('dpo', rows)
        with self.assertRaisesRegex(ValueError, 'different prompts'):
            self.package()

    def test_one_reviewer_remains_pending_two_distinct_complete_all_pairs(self):
        self.package()
        self.write_annotations(self.annotations(('fixture_reviewer_A',)))
        result = self.package()
        self.assertEqual((result['status'], result['valid_annotations']), ('awaiting_human_annotations', 9))
        self.write_annotations(self.annotations())
        result = self.package()
        self.assertEqual((result['status'], result['valid_annotations']), ('annotated', 18))
        self.assertTrue(result['not_population_level_claim'])

    def test_two_reviewers_only_for_some_pairs_does_not_complete_package(self):
        self.package()
        rows = self.annotations()
        rows.pop()  # Eight pairs have two reviewers; one has only one.
        self.write_annotations(rows)
        self.assertEqual(self.package()['status'], 'awaiting_human_annotations')

    def test_duplicate_reviewer_per_pair_is_rejected_including_whitespace_alias(self):
        self.package()
        rows = self.annotations(('fixture_reviewer_A', ' fixture_reviewer_A '))
        self.write_annotations(rows)
        with self.assertRaisesRegex(ValueError, 'Duplicate reviewer'):
            self.package()

    def test_blank_names_or_invalid_preferences_do_not_count_as_reviews(self):
        self.package()
        rows = self.annotations()
        for index, row in enumerate(rows):
            if index % 2:
                row['preference'] = 'model said it is good'
            else:
                row['reviewer_id'] = ' '
        self.write_annotations(rows)
        result = self.package()
        self.assertEqual((result['status'], result['valid_annotations']), ('awaiting_human_annotations', 0))

    def test_unknown_pair_id_is_rejected(self):
        self.package()
        rows = self.annotations()
        rows.append({'pair_id': 'unknown', 'reviewer_id': 'fixture_reviewer_C', 'preference': 'A', 'notes': 'fixture'})
        self.write_annotations(rows)
        with self.assertRaisesRegex(ValueError, 'Unknown blind pair'):
            self.package()


class ToolReceiptTests(EvaluationGateFixture):
    def receipt(self, *, with_lora=False):
        checkpoint = self.root / 'fixture_weights.pth'
        checkpoint.write_bytes(b'SYNTHETIC WEIGHT BYTES FOR HASH CHECK ONLY')
        source = self.root / 'fixture_evaluator.py'
        source.write_text('# synthetic source fingerprint only\n', encoding='utf-8')
        model = {'checkpoint': str(checkpoint), 'sha256': digest(checkpoint)}
        provenance = {'checkpoint': {'path': str(checkpoint), 'sha256': digest(checkpoint)},
                      'evaluator': {'path': str(source), 'sha256': digest(source)}}
        if with_lora:
            adapter = self.root / 'fixture_adapter.pth'
            adapter.write_bytes(b'SYNTHETIC ADAPTER BYTES; NOT A TRAINED MODEL')
            model.update(lora=str(adapter), lora_sha256=digest(adapter))
            provenance['lora'] = {'path': str(adapter), 'sha256': digest(adapter)}
        value = {'status': 'complete', 'seed': 42, 'max_turns': 3, 'max_new_tokens': 512,
                 'backend': 'local', 'temperature': .9, 'top_p': .9,
                 'cases': [{'fixture_case': i} for i in range(8)], 'provenance': provenance}
        path = self.root / 'fixture_results.json'
        write_json(path, value)
        return path, value, model

    def test_valid_receipt_is_verified_without_model_or_gpu_imports(self):
        path, value, model = self.receipt()
        self.assertEqual(evaluation.verify_evaluation_result('tool_seed42', path, model), value)
        self.assertEqual(evaluation.verify_evaluation_result('tool_seed42', path, model), value)

    def test_valid_lora_receipt_requires_the_matching_base_and_adapter(self):
        path, value, model = self.receipt(with_lora=True)
        evaluation.verify_evaluation_result('tool_seed42', path, model)
        changed = {**model, 'lora_sha256': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'checkpoint/adapter differs'):
            evaluation.verify_evaluation_result('tool_seed42', path, changed)

    def test_changed_checkpoint_identity_is_rejected(self):
        path, value, model = self.receipt()
        with self.assertRaisesRegex(ValueError, 'checkpoint/adapter differs'):
            evaluation.verify_evaluation_result('tool_seed42', path, {**model, 'sha256': '0' * 64})

    def test_changed_checkpoint_bytes_are_rejected_even_if_receipt_matches_expected_hash(self):
        path, value, model = self.receipt()
        Path(model['checkpoint']).write_bytes(b'changed bytes after receipt creation')
        with self.assertRaisesRegex(ValueError, 'changed recorded artifact'):
            evaluation.verify_evaluation_result('tool_seed42', path, model)

    def test_changed_seed_is_rejected(self):
        path, value, model = self.receipt()
        value['seed'] = 123
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, 'protocol differs'):
            evaluation.verify_evaluation_result('tool_seed42', path, model)

    def test_missing_cases_or_changed_decoding_or_pending_receipt_are_rejected(self):
        path, original, model = self.receipt()
        changes = ({'cases': original['cases'][:-1]}, {'max_turns': 4}, {'max_new_tokens': 128},
                   {'temperature': .7}, {'top_p': .8}, {'backend': 'api'}, {'status': 'running'})
        for change in changes:
            with self.subTest(change=change):
                write_json(path, {**original, **change})
                with self.assertRaises(ValueError):
                    evaluation.verify_evaluation_result('tool_seed42', path, model)

    def test_changed_recorded_source_file_is_rejected(self):
        path, value, model = self.receipt()
        Path(value['provenance']['evaluator']['path']).write_text('# changed source\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed recorded artifact'):
            evaluation.verify_evaluation_result('tool_seed42', path, model)


if __name__ == '__main__':
    unittest.main()
