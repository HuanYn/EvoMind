import unittest

from model.grpo_reward import score_final_integer


class TestGRPOMathReward(unittest.TestCase):
    def test_correct_final_integer_gets_one(self):
        result = score_final_integer("先计算，最终答案是 8。", 8)
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.parsed_answer, 8)

    def test_wrong_or_missing_final_integer_gets_zero(self):
        self.assertEqual(score_final_integer("答案是 7", 8).reward, 0.0)
        self.assertEqual(score_final_integer("我不确定", 8).reward, 0.0)

    def test_question_numbers_do_not_count_without_final_answer(self):
        result = score_final_integer("题目里有 3 和 5。", 8)
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.parsed_answer, 5)


if __name__ == "__main__":
    unittest.main()
