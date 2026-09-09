"""CPU-only recipe contracts; never import training modules or allocate CUDA."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evomind_text_recipe as recipe


class TextRecipeTests(unittest.TestCase):
    def test_defaults_do_not_execute_runtime_code(self):
        result = recipe.parser_defaults('p.add_argument("--device", default=raise_if_called()); p.add_argument("--enabled", action="store_true")')
        self.assertEqual(result["device"], {"runtime_expression": "raise_if_called()"})
        self.assertIs(result["enabled"], False)

    def test_pinned_branches_preserve_official_epochs_and_bases(self):
        data = recipe.inventory()
        self.assertFalse(data["vision_enabled"])
        rows = {row["name"]: row for row in data["branches"]}
        for name, row in rows.items():
            self.assertFalse(row["launch_enabled"])
            self.assertEqual(row["upstream_defaults"]["epochs"], {"lora": 10, "distillation": 6}.get(name, 1))
            self.assertIsNone(row["local_hardware_overrides"])
            if name != "distillation":
                self.assertEqual(row["upstream_defaults"]["from_weight"], "full_sft")
        self.assertEqual(rows["grpo"]["upstream_defaults"]["loss_type"], "cispo")
        self.assertEqual(rows["grpo"]["method_and_output_overrides"]["loss_type"], "grpo")
        self.assertNotEqual(rows["grpo"]["method_and_output_overrides"]["save_weight"], rows["cispo"]["method_and_output_overrides"]["save_weight"])
        self.assertEqual(rows["distillation"]["upstream_defaults"]["teacher_use_moe"], 1)
        scope = json.loads((recipe.ROOT / "configs/text_alignment_scope.json").read_text(encoding="utf-8"))
        self.assertTrue(set(rows).issubset(set(scope["required_coverage"])))


if __name__ == "__main__":
    unittest.main()
