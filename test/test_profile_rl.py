import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import datasets
import torch
torch.set_num_threads(2)
from evomind_profile_rl import normalize_scores, reward_messages, topk_nucleus, fast_generate


class TestProfileRL(unittest.TestCase):
    def test_scores(self):
        self.assertEqual(normalize_scores(2.0, 1), [2.0])
        self.assertEqual(normalize_scores([1., 2.], 2), [1., 2.])
        for x, count in [(float('nan'), 1), ([1.], 2)]:
            with self.assertRaises(ValueError): normalize_scores(x, count)

    def test_messages(self):
        self.assertEqual(reward_messages([{'role':'user','content':'Q'}], 'A'),
                         [{'role':'user','content':'Q'},{'role':'assistant','content':'A'}])
        messages = [{'role':'user','content':'old'}, {'role':'assistant','content':'reply'}, {'role':'user','content':'new'}]
        self.assertEqual(reward_messages(messages, 'A')[0]['content'],
                         'user: old\nassistant: reply\n以上是对话历史。我的新问题是：\nnew')

    def test_sampler_parity(self):
        for dtype in (torch.float32, torch.bfloat16):
            for seed in range(10):
                torch.manual_seed(seed)
                scores = torch.randn(6, 128).to(dtype)
                if seed == 0: scores[:] = 1  # threshold and internal ties
                baseline = scores.clone()
                baseline[baseline < torch.topk(baseline, 50)[0][:,-1,None]] = -float('inf')
                values, indices = torch.sort(baseline, descending=True)
                mask = torch.softmax(values, -1).cumsum(-1) > .85
                mask[:,1:], mask[:,0] = mask[:,:-1].clone(), False
                baseline[mask.scatter(1,indices,mask)] = -float('inf')
                self.assertTrue(torch.equal(topk_nucleus(scores), baseline), (dtype,seed))

    def test_generate_parity(self):
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        torch.manual_seed(21)
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=32, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=128)).eval()
        ids = torch.tensor([[1,5,6], [1,7,8]])
        mask = torch.ones_like(ids)
        for sample in (False, True):
            with torch.inference_mode():
                torch.manual_seed(99)
                native = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=8,
                    temperature=.8, do_sample=sample)
                torch.manual_seed(99)
                candidate = fast_generate(model, ids, mask, 8, 2, optimized_sampling=True, do_sample=sample)
            self.assertTrue(torch.equal(native,candidate))


if __name__ == '__main__': unittest.main()
