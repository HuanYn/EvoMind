"""Portable text reproduction entry: plan a direct trainer call, or check CPU code.

Plans and --help use only the standard library. No historical supervisor, remote
launcher, asset downloader, or vision pipeline is imported by this module.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAINERS = {
    "pretrain": "train_pretrain.py",
    "sft": "train_full_sft.py",
    "dpo": "train_dpo.py",
    "grpo": "train_grpo.py",
    "cispo": "train_grpo.py",
}
DATASETS = {
    "pretrain": "pretrain_t2t_mini.jsonl",
    "sft": "sft_t2t_mini.jsonl",
    "dpo": "dpo.jsonl",
    "grpo": "rlaif.jsonl",
    "cispo": "rlaif.jsonl",
}
WEIGHTS = {"pretrain": "pretrain", "sft": "full_sft", "dpo": "dpo", "grpo": "grpo", "cispo": "cispo"}
PARENTS = {"pretrain": "none", "sft": "pretrain", "dpo": "full_sft", "grpo": "full_sft", "cispo": "full_sft"}
PATH_OPTIONS = {"--data_path", "--save_dir", "--init_dir", "--resume_dir", "--tokenizer_path", "--reward_model_path", "--resume"}


def option_value(argv, name, default=None):
    """Use argparse's last-value convention for ordinary value options."""
    result = default
    for index, item in enumerate(argv):
        if item == name:
            if index + 1 == len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"{name} requires a value")
            result = argv[index + 1]
        elif item.startswith(name + "="):
            result = item.split("=", 1)[1]
    return result


def normalize_paths(argv, root):
    result = list(argv)
    for index, item in enumerate(result):
        key, equals, value = item.partition("=")
        if key not in PATH_OPTIONS:
            continue
        if not equals:
            if index + 1 >= len(result) or result[index + 1].startswith("--"):
                raise ValueError(f"{key} requires a path")
            # Normalize this occurrence, not a later repeated option.
            value = result[index + 1]
        path = Path(value).expanduser()
        path = path if path.is_absolute() else root / path
        absolute = str(path.resolve())
        if equals:
            result[index] = key + "=" + absolute
        else:
            result[index + 1] = absolute
    return result


def trainer_options(stage, root=ROOT):
    """Read declared option names without importing torch or running a trainer."""
    sources = [Path(root) / "trainer" / TRAINERS[stage]]
    if stage == "dpo":
        sources.append(Path(root) / "trainer" / "evomind_offline_runtime.py")
    if stage in {"grpo", "cispo"}:
        sources.append(Path(root) / "trainer" / "evomind_rl_runtime.py")
    options = {"--help"}
    for source in sources:
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                options.update(item.value for item in node.args if isinstance(item, ast.Constant) and isinstance(item.value, str))
    return options


def build_command(stage, extra=(), *, root=ROOT, executable=sys.executable):
    root = Path(root).resolve()
    extra = list(extra)
    known = trainer_options(stage, root)
    unknown = [item for item in extra if item.startswith("--") and item.partition("=")[0] not in known]
    if unknown:
        raise ValueError("Unknown trainer option(s): " + ", ".join(unknown))
    defaults = []
    for name, value in (("--device", "cpu"), ("--num_workers", "0"),
                        ("--data_path", str(root / "dataset" / DATASETS[stage])),
                        ("--save_dir", str(root / "out")), ("--save_weight", WEIGHTS[stage])):
        if option_value(extra, name) is None:
            defaults.extend([name, value])
    if stage in {"grpo", "cispo"}:
        if option_value(extra, "--loss_type", stage) != stage:
            raise ValueError(f"The {stage} stage requires --loss_type {stage}")
        if "--evomind_runtime" not in extra:
            defaults.append("--evomind_runtime")
        if option_value(extra, "--loss_type") is None:
            defaults += ["--loss_type", stage]
    command = [str(executable), "-B", str(root / "trainer" / TRAINERS[stage])]
    command += normalize_paths(defaults + extra, root)
    return {"stage": stage, "cwd": str(root / "trainer"), "argv": command}


def validate_inputs(plan, *, root=ROOT):
    """Check local assets and existing output names before an explicit launch."""
    root = Path(root).resolve()
    stage, argv = plan["stage"], plan["argv"][3:]
    data = Path(option_value(argv, "--data_path"))
    tokenizer = Path(option_value(argv, "--tokenizer_path", str(root / "model")))
    required = [data, tokenizer / "tokenizer.json", tokenizer / "tokenizer_config.json"]
    suffix = "_moe" if option_value(argv, "--use_moe", "0") == "1" else ""
    hidden = option_value(argv, "--hidden_size", "768")
    parent = option_value(argv, "--from_weight", PARENTS[stage])
    if parent != "none":
        # The pretrain/SFT trainers always load from ../out. DPO/RL expose init_dir.
        init_dir = Path(option_value(argv, "--init_dir", str(root / "out")))
        required.append(init_dir / f"{parent}_{hidden}{suffix}.pth")
    if stage in {"grpo", "cispo"}:
        reward = option_value(argv, "--reward_model_path")
        if not reward or not Path(reward).is_dir():
            raise ValueError("GRPO/CISPO requires --reward_model_path pointing to a local reward-model directory")
        required.append(Path(reward) / "config.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Missing local inputs: " + ", ".join(missing))
    weight = option_value(argv, "--save_weight")
    if not weight or Path(weight).name != weight or weight in {".", ".."}:
        raise ValueError("--save_weight must be a plain filename prefix")
    output = Path(option_value(argv, "--save_dir")) / f"{weight}_{hidden}{suffix}.pth"
    resume_dir = Path(option_value(argv, "--resume_dir", str(root / "checkpoints")))
    resume_file = resume_dir / f"{weight}_{hidden}{suffix}_resume.pth"
    resuming = option_value(argv, "--from_resume", "0") == "1" or option_value(argv, "--resume")
    if not resuming and (output.exists() or resume_file.exists()):
        raise ValueError("Output/checkpoint already exists; choose a new --save_weight/--save_dir or use explicit trainer resume options")


def cpu_smoke(*, use_moe=False):
    """One synthetic CPU forward/backward and in-memory SGD step; no checkpoint."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import datasets  # noqa: F401 -- must precede torch on this Windows setup.
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "model"), local_files_only=True)
    chat = tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=True)
    if not chat:
        raise AssertionError("Local tokenizer produced an empty chat template")
    config = MiniMindConfig(hidden_size=32, num_hidden_layers=1, vocab_size=64,
                            num_attention_heads=4, num_key_value_heads=2,
                            intermediate_size=64, max_position_embeddings=32,
                            flash_attn=False, use_moe=use_moe)
    model = MiniMindForCausalLM(config).cpu().train()
    ids = torch.randint(3, config.vocab_size, (2, 16), device="cpu")
    result = model(ids, labels=ids)
    loss = result.loss + result.aux_loss
    if not torch.isfinite(loss):
        raise AssertionError("Non-finite smoke loss")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("Missing or non-finite gradients")
    grad_norm = sum(gradient.float().square().sum().item() for gradient in gradients) ** 0.5
    if grad_norm == 0:
        raise AssertionError("Zero gradient norm")
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-1e-3)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise AssertionError("Non-finite parameter after SGD step")
    return {"check": "synthetic_cpu_forward_backward_sgd", "device": "cpu", "use_moe": use_moe,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "logits_shape": list(result.logits.shape), "loss": loss.item(), "gradient_norm": grad_norm,
            "tokenizer": "local_files_only", "checkpoint_written": False,
            "torch": torch.__version__}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    extra = argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__, epilog="Trainer flags go after --; their path values are relative to the repository root.")
    subs = parser.add_subparsers(dest="command", required=True)
    train = subs.add_parser("train", help="Print a direct trainer plan; --execute explicitly starts it")
    train.add_argument("stage", choices=TRAINERS)
    action = train.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--dry-run", action="store_true", help="Print JSON only (also the default)")
    helper = subs.add_parser("trainer-help", help="Run the actual trainer's --help (dependencies required)")
    helper.add_argument("stage", choices=TRAINERS)
    smoke = subs.add_parser("smoke", help="Run a tiny synthetic CPU model/tokenizer check, without training assets")
    smoke.add_argument("--use-moe", action="store_true")
    args = parser.parse_args(argv[:split])
    if extra and args.command != "train":
        parser.error("Only train accepts trainer flags after --")
    if args.command == "smoke":
        print(json.dumps(cpu_smoke(use_moe=args.use_moe), indent=2))
        return 0
    if args.command == "trainer-help":
        return subprocess.call([sys.executable, "-B", str(ROOT / "trainer" / TRAINERS[args.stage]), "--help"], cwd=ROOT / "trainer")
    try:
        plan = build_command(args.stage, extra)
        if args.execute:
            validate_inputs(plan)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"execute": args.execute, **plan}, indent=2), flush=True)
    if args.execute:
        environment = dict(os.environ)
        environment.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
        environment.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        environment.setdefault("PYTHONUNBUFFERED", "1")
        return subprocess.call(plan["argv"], cwd=plan["cwd"], env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
