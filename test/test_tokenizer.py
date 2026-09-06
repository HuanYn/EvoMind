import tempfile
import unittest
from pathlib import Path

from model.tokenizer import CharTokenizer


class TestCharTokenizer(unittest.TestCase):
    def test_encode_decode_and_persistence(self):
        tokenizer = CharTokenizer.build("MiniMind 学习")
        ids = tokenizer.encode("MiniMind", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tokenizer.bos_id)
        self.assertEqual(ids[-1], tokenizer.eos_id)
        self.assertEqual(tokenizer.decode(ids), "MiniMind")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            self.assertEqual(CharTokenizer.load(path).token_to_id, tokenizer.token_to_id)


if __name__ == "__main__":
    unittest.main()
