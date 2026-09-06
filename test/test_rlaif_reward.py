import unittest

from model.rlaif_reward import repeated_ngram_penalty, score_minimind_rlaif


class TestRLAIFReward(unittest.TestCase):
    def test_clean_answer_has_no_repeat_penalty(self):
        self.assertEqual(repeated_ngram_penalty("这是一个简短但不重复的回答。"), 0.0)

    def test_repetition_is_bounded(self):
        self.assertGreater(repeated_ngram_penalty("人工智能" * 30), 0.0)
        self.assertLessEqual(repeated_ngram_penalty("人工智能" * 30), 0.5)

    def test_composite_reward_exposes_every_component(self):
        response = "<think>这是足够长的推理内容，用于检查格式奖励与长度奖励的计算是否正确。</think>\n答案是一个清晰的解释。"
        reward = score_minimind_rlaif(response, 9.0)
        self.assertEqual(reward.reward_model_score, 3.0)
        self.assertEqual(reward.length_reward, 0.5)
        self.assertEqual(reward.think_reward, 1.25)
        self.assertAlmostEqual(reward.reward, 0.5 + 1.25 - reward.repetition_penalty + 3.0)


if __name__ == "__main__":
    unittest.main()
