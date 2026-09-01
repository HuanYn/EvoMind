import unittest

import torch
import torch.nn.functional as F

from src.minimind.config import ModelConfig
from src.minimind.model import MiniMindModel


class TestMiniMindModel(unittest.TestCase):
    def test_forward_and_backward(self):
        torch.manual_seed(42)
        config = ModelConfig()
        model = MiniMindModel(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 8, config.vocab_size))

        targets = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertIsNotNone(model.blocks[0].attention.q_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
