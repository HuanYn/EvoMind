import unittest

import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.model_minimind import MiniMindModel


class TestMiniMindModel(unittest.TestCase):
    @staticmethod
    def make_config():
        return ModelConfig(
            vocab_size=64,
            dim=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            hidden_dim=64,
            max_seq_len=16,
        )

    def test_forward_and_backward(self):
        torch.manual_seed(42)
        config = self.make_config()
        model = MiniMindModel(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 8, config.vocab_size))

        targets = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertIsNotNone(model.blocks[0].self_attn.q_proj.weight.grad)

    def test_kv_cache_matches_full_forward(self):
        torch.manual_seed(7)
        config = self.make_config()
        model = MiniMindModel(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 6), dtype=torch.long)

        full_logits = model(input_ids)
        _, cache = model(input_ids[:, :3], use_cache=True)
        cached_logits, _ = model(input_ids[:, 3:], past_key_values=cache, use_cache=True)

        self.assertTrue(torch.allclose(full_logits[:, 3:], cached_logits, atol=1e-5, rtol=1e-5))

    def test_dense_model_accepts_and_ignores_token_mask(self):
        torch.manual_seed(11)
        config = self.make_config()
        model = MiniMindModel(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 6), dtype=torch.long)
        token_mask = torch.tensor(
            [[True, True, True, False, False, False], [True, True, False, False, False, False]]
        )

        expected = model(input_ids)
        actual = model(input_ids, token_mask=token_mask)

        self.assertTrue(torch.equal(expected, actual))

    def test_moe_token_mask_excludes_padding_from_routing(self):
        torch.manual_seed(13)
        config = self.make_config()
        config.use_moe = True
        model = MiniMindModel(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 6), dtype=torch.long)
        token_mask = torch.tensor(
            [[True, True, True, False, False, False], [True, True, False, False, False, False]]
        )
        changed_padding = input_ids.clone()
        changed_padding[~token_mask] = (changed_padding[~token_mask] + 17) % config.vocab_size

        logits, aux_loss, fractions = model(
            input_ids, token_mask=token_mask, return_router_loss=True
        )
        changed_logits, changed_aux_loss, changed_fractions = model(
            changed_padding, token_mask=token_mask, return_router_loss=True
        )

        self.assertTrue(torch.allclose(logits[token_mask], changed_logits[token_mask]))
        self.assertTrue(torch.equal(aux_loss, changed_aux_loss))
        self.assertTrue(torch.equal(torch.stack(fractions), torch.stack(changed_fractions)))

    def test_moe_masked_positions_receive_no_moe_gradient(self):
        torch.manual_seed(17)
        config = self.make_config()
        config.use_moe = True
        moe = MiniMindModel(config).blocks[0].mlp
        x = torch.randn(2, 4, config.dim, requires_grad=True)
        token_mask = torch.tensor(
            [[True, True, False, False], [True, False, True, False]]
        )

        output, aux_loss, expert_fraction = moe(x, token_mask=token_mask)
        (output.sum() + aux_loss).backward()

        self.assertTrue(torch.equal(output[~token_mask], torch.zeros_like(output[~token_mask])))
        self.assertTrue(torch.equal(x.grad[~token_mask], torch.zeros_like(x.grad[~token_mask])))
        self.assertAlmostEqual(expert_fraction.sum().item(), 1.0)

    def test_moe_all_true_mask_matches_unmasked_behavior(self):
        torch.manual_seed(19)
        config = self.make_config()
        config.use_moe = True
        model = MiniMindModel(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 5), dtype=torch.long)

        expected = model(input_ids, return_router_loss=True)
        actual = model(
            input_ids,
            token_mask=torch.ones_like(input_ids, dtype=torch.bool),
            return_router_loss=True,
        )

        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))
        self.assertTrue(torch.equal(torch.stack(expected[2]), torch.stack(actual[2])))

    def test_token_mask_validation(self):
        config = self.make_config()
        config.use_moe = True
        model = MiniMindModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 5), dtype=torch.long)

        with self.assertRaisesRegex(TypeError, "dtype torch.bool"):
            model(input_ids, token_mask=torch.ones_like(input_ids))
        with self.assertRaisesRegex(ValueError, "shape"):
            model(input_ids, token_mask=torch.ones(2, 4, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "at least one token"):
            model(input_ids, token_mask=torch.zeros_like(input_ids, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
