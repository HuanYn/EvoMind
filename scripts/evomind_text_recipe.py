"""Export a non-executable MiniMind-3 recipe inventory without importing trainers.

Reads parser defaults directly from the pinned Git source (AST only). This is a
plan, NOT a runnable queue: missing assets, compatibility and GPU probes must be
resolved before creating any launch manifest. No training or downloads occur.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "6fc918beb68a0d8c40452338df6319fe168014ba"
BRANCHES = (
    ("dpo", "train_dpo.py", {}, ["full_sft"]),
    ("ppo", "train_ppo.py", {}, ["full_sft", "reward_model"]),
    ("grpo", "train_grpo.py", {"loss_type": "grpo", "save_weight": "grpo"}, ["full_sft", "reward_model"]),
    ("cispo", "train_grpo.py", {"loss_type": "cispo", "save_weight": "cispo"}, ["full_sft", "reward_model"]),
    ("agent_grpo", "train_agent.py", {"loss_type": "grpo", "save_weight": "agent_grpo"}, ["full_sft", "reward_model", "safe_windows_tools"]),
    ("agent_cispo", "train_agent.py", {"loss_type": "cispo", "save_weight": "agent_cispo"}, ["full_sft", "reward_model", "safe_windows_tools"]),
    ("lora", "train_lora.py", {}, ["full_sft"]),
    ("distillation", "train_distillation.py", {}, ["full_sft_dense_student", "full_sft_moe_teacher"]),
)


def parser_defaults(source):
    """Extract literal defaults, leaving runtime expressions visibly unresolved."""
    result = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        flags = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--")]
        if not flags:
            continue
        fields = {item.arg: item.value for item in node.keywords}
        key = flags[0][2:].replace("-", "_")
        if "dest" in fields:
            key = ast.literal_eval(fields["dest"])
        if "default" in fields:
            try:
                value = ast.literal_eval(fields["default"])
            except (ValueError, TypeError):
                value = {"runtime_expression": ast.unparse(fields["default"])}
        elif "action" in fields and ast.literal_eval(fields["action"]) == "store_true":
            value = False
        else:
            value = None
        result[key] = value
    return result


def inventory(root=ROOT):
    rows = []
    for name, script, overrides, dependencies in BRANCHES:
        relative = f"trainer/{script}"
        source = subprocess.check_output(["git", "show", f"{COMMIT}:{relative}"], cwd=root)
        defaults = parser_defaults(source.decode("utf-8-sig"))
        rows.append({
            "name": name, "script": relative, "upstream_defaults": defaults,
            "upstream_source_sha256": hashlib.sha256(source).hexdigest(),
            "workspace_source_sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
            "method_and_output_overrides": overrides,
            "depends_on": dependencies,
            "status": "pending_preflight", "launch_enabled": False,
            "local_hardware_overrides": None,
            "required_checks": ["source_and_data_revision_hash", "base_weight_strict_load", "loss_mask_and_reward_contract",
                                "memory_and_finite_gradient_smoke", "checkpoint_and_resume", "isolated_output_and_metrics"],
        })
    return {
        "schema_version": 1, "upstream_commit": COMMIT,
        "kind": "audit_inventory_not_launch_manifest", "vision_enabled": False,
        "active_run": "configs/text_official_mini.json",
        "weight_rule": "Branches reuse official SFT initialization, not the previous branch output. Distillation has separate teacher/student prerequisites.",
        "alignment_rule": "CLI defaults are source evidence, not proof of released-weight training history. Full mini data is not the optional full-size corpus.",
        "hardware_rule": "Do not silently shorten generations, reduce group size, model, epochs or data. Probe and record any adaptation first.",
        "branches": rows,
        "evaluation_required": ["same_prompt_and_decoding", "multiple_sampling_seeds", "eos_repetition_raw_answers",
                                "thinking_on_off", "tool_call_and_execution", "chinese_objective_benchmarks_overlap_audited",
                                "human_blind_package_and_real_annotations", "latency_memory_curves_provenance"],
        "known_preflight_items": ["Windows Agent SIGALRM tool timeout is unsupported", "Official tool demo uses unrestricted eval; require safe bounded arithmetic",
                                  "InternLM reward tokenizer/runtime compatibility", "Default RL batch/group/long response must be measured on 8GiB",
                                  "Distillation requires a compatible trained MoE teacher; never substitute random weights"],
    }


def main():
    data = inventory()
    path = ROOT / "artifacts/provenance/minimind3_text_recipe_inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(data['branches'])} audited branch recipes (no jobs launched): {path}")


if __name__ == "__main__":
    main()
