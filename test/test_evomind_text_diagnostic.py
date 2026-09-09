"""Read-only diagnostic resume/metric contracts; no ML imports or GPU work."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evomind_text_eval as diagnostic


class DiagnosticTests(unittest.TestCase):
    def rows(self):
        return [{"mode": mode, "seed": seed, "prompt_id": index, "prompt": prompt,
                 "expected_short_answer": answer, "thinking": False}
                for mode in diagnostic.MODES
                for seed in ([42] if mode == "greedy" else [42, 123, 2026])
                for index, (prompt, answer) in enumerate(diagnostic.PROMPTS)]

    def test_full_identity_grid(self):
        self.assertEqual(len(diagnostic.PROMPTS), 8)
        self.assertEqual(len(diagnostic.validate_saved_rows(self.rows(), [42, 123, 2026], False, complete=True)), 24)

    def test_partial_only_allowed_for_resume(self):
        diagnostic.validate_saved_rows(self.rows()[:7], [42, 123, 2026], False)
        with self.assertRaises(ValueError):
            diagnostic.validate_saved_rows(self.rows()[:7], [42, 123, 2026], False, complete=True)

    def test_duplicate_unknown_and_prompt_changes_rejected(self):
        rows = self.rows()
        for field, value in (("seed", 99), ("prompt", "changed"), ("thinking", True),
                             ("expected_short_answer", "invented")):
            altered = copy.deepcopy(rows)
            altered[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                diagnostic.validate_saved_rows(altered, [42, 123, 2026], False)
        with self.assertRaises(ValueError):
            diagnostic.validate_saved_rows(rows + rows[:1], [42, 123, 2026], False)

    def test_no_ngrams_is_undefined_not_zero_repetition(self):
        self.assertIsNone(diagnostic.repetition([1, 2], 4))
        self.assertIsNone(diagnostic.valid_mean([{"x": None}], "x"))
        self.assertEqual(diagnostic.repetition([1, 1, 1, 1, 1], 4), .5)
        self.assertEqual(diagnostic.valid_mean([{"x": None}, {"x": .2}, {"x": .4}], "x"), (.2+.4)/2)


if __name__ == "__main__":
    unittest.main()
