import copy
import sys
import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evomind_product as product
import evomind_posttrain as training


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.plan = product.settings()
        self.policy = self.plan['selection']
        self.base = {'benchmarks': dict.fromkeys(product.TASKS, .5), 'eos': .75, 'repeat4': .1}

    def test_only_dpo_cispo_on_mainline(self):
        self.assertEqual([b['name'] for b in self.plan['stages']], ['dpo', 'cispo'])
        self.assertFalse(self.plan['agent']['blocks_vision'])
        self.assertEqual(self.plan['rlaif_rows'], 19502)
        self.assertEqual(self.plan['stages'][1]['data'], 'dataset/rlaif.jsonl')

    def test_equal_candidate_retains_parent(self):
        self.assertFalse(product.compare(self.base, self.base, self.policy)['promote'])

    def test_improved_candidate_promotes(self):
        other = copy.deepcopy(self.base)
        other['benchmarks']['cmmlu'] += .02
        self.assertTrue(product.compare(self.base, other, self.policy)['promote'])

    def test_better_benchmark_cannot_hide_repetition(self):
        other = copy.deepcopy(self.base)
        other['benchmarks']['cmmlu'] += .1
        other['repeat4'] += .01
        self.assertFalse(product.compare(self.base, other, self.policy)['promote'])

    def test_single_benchmark_regression_rejects(self):
        other = copy.deepcopy(self.base)
        other['benchmarks']['cmmlu'] += .1
        other['benchmarks']['piqa'] -= .02
        self.assertFalse(product.compare(self.base, other, self.policy)['promote'])

    def test_missing_nonfinite_metrics_fail(self):
        other = copy.deepcopy(self.base)
        del other['benchmarks']['piqa']
        with self.assertRaises(ValueError):
            product.compare(self.base, other, self.policy)
        with self.assertRaises(ValueError):
            product.metric(float('nan'))

    def test_sft_screening_holds_degenerate_base(self):
        self.assertTrue(product.baseline_ready(self.base, self.policy))
        self.assertFalse(product.baseline_ready({**self.base, 'eos': 0}, self.policy))

    def test_dynamic_cispo_initialization_is_not_sft_alias(self):
        branch = copy.deepcopy(self.plan['stages'][1])
        branch['options']['from_weight'] = 'dpo'
        argv = training.command(branch, branch['batch_candidates'][0], Path('test-output'),
                                Path('selected/dpo_768.pth'), probe=True)
        self.assertEqual(argv[argv.index('--from_weight')+1], 'dpo')
        self.assertEqual(argv[argv.index('--init_dir')+1], 'selected')

    def test_old_queue_cannot_launch(self):
        with self.assertRaisesRegex(RuntimeError, 'superseded'):
            training.run_posttraining()

    def test_native_metric_not_acc_norm(self):
        summary = {'scores': {'piqa': {'acc,none': .4, 'acc_norm,none': .7}}}
        self.assertEqual(product.benchmark_score(summary, 'piqa'), .4)

    def exercise_route(self, promote_dpo):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'product.json'
            plan = copy.deepcopy(self.plan)
            plan.update(run_dir=str(root / 'run'), eval_dir=str(root / 'eval'))
            config.write_text(json.dumps(plan), encoding='utf-8')
            base = root / 'full_sft_768.pth'
            base.write_bytes(b'fake test checkpoint')
            seen = []
            def train(branch, state, persist, initial):
                seen.append((branch['name'], branch['options']['from_weight'], initial.name))
                output = root / (branch['name'] + '_768.pth')
                output.write_bytes(branch['name'].encode())
                state['branches'][branch['name']].update(status='completed',
                    output={'path': str(output), 'sha256': product.sha256(output)})
                persist()
            def evaluate(models):
                scores = {n: copy.deepcopy(self.base) for n in models}
                if promote_dpo and 'dpo' in scores:
                    scores['dpo']['benchmarks']['cmmlu'] += .02
                if 'cispo' in scores:
                    scores['cispo']['repeat4'] = .9  # reject a degenerate RL model
                return scores, []
            with patch.object(product, 'CONFIG', config), \
                    patch.object(training, 'RUN', root / 'unused'), \
                    patch.object(product.evaluation, 'RUN', root / 'unused'), \
                    patch.object(product.evaluation, 'EVAL', root / 'unused'), \
                    patch.object(training, 'base_checkpoint', return_value=base), \
                    patch.object(training, 'run_branch', side_effect=train), \
                    patch.object(product, 'evaluate', side_effect=evaluate), \
                    patch.object(product, 'selected_checkpoint', return_value=base):
                product.run_product()
            receipt = json.loads((root / 'run/product_acceptance.json').read_text(encoding='utf-8'))
            return seen, receipt['selected_name']

    def test_route_inherits_promoted_dpo_then_rejects_bad_cispo(self):
        seen, selected = self.exercise_route(True)
        self.assertEqual(seen[1], ('cispo', 'dpo', 'dpo_768.pth'))
        self.assertEqual(selected, 'dpo')

    def test_route_falls_back_to_sft_before_cispo(self):
        seen, selected = self.exercise_route(False)
        self.assertEqual(seen[1], ('cispo', 'full_sft', 'full_sft_768.pth'))
        self.assertEqual(selected, 'full_sft')


if __name__ == '__main__':
    unittest.main()
