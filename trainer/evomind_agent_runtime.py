"""Agent-only optimizer-boundary checkpoints; never alters live trainer_utils."""
from __future__ import annotations

from pathlib import Path
import re

import torch

from trainer.evomind_rl_runtime import atomic_json, atomic_torch_save, capture_rng, cpu_tree, sha256_file

FORMAT = "evomind.agent.optimizer-boundary/1"
ROOT = Path(__file__).resolve().parents[1]


class AgentAdamW(torch.optim.AdamW):
    """Keep CPU probes CPU-only with this installation's PyTorch 2.13 optimizer.

    AdamW's new accelerator graph-capture health check queries a CUDA stream
    even for CPU parameters. No accelerator graph can capture a CPU-only update;
    skip only that check, never the AdamW update or any finite-gradient checks.
    GPU optimizers use the upstream implementation unchanged.
    """
    def _accelerator_graph_capture_health_check(self):
        if all(parameter.device.type == 'cpu' for group in self.param_groups for parameter in group['params']):
            return
        return super()._accelerator_graph_capture_health_check()


def owned_directory(value):
    path = Path(value).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError("Agent outputs must be a dedicated directory inside evomind")
    path.mkdir(parents=True, exist_ok=True)
    return path


def agent_contract(args, config, dataset_rows):
    fields = ('epochs', 'batch_size', 'learning_rate', 'dtype', 'accumulation_steps', 'grad_clip',
              'max_seq_len', 'max_gen_len', 'max_total_len', 'num_generations', 'beta', 'loss_type',
              'epsilon', 'epsilon_high', 'thinking_ratio', 'max_turns', 'seed', 'reward_device', 'reward_dtype')
    contract = {key: getattr(args, key) for key in fields}
    contract.update(dataset_rows=dataset_rows, hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers, use_moe=config.use_moe,
                    device_type=torch.device(args.device).type,
                    probe_only=args.max_steps > 0,
                    tool_semantics="original_mock_tables_bounded_ast_arithmetic_v1",
                    mask_semantics="per_turn_eos_inclusive_tools_zero_actual_token_prefix_v1")
    initial = Path(args.init_dir) / f"{args.from_weight}_{config.hidden_size}{'_moe' if config.use_moe else ''}.pth"
    for label, path in (('initial_weights', initial), ('training_data', Path(args.data_path))):
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        contract[label] = {'path': str(path), 'sha256': sha256_file(path)}
    contract['reward_model_path'] = str(Path(args.reward_model_path).resolve())
    tokenizer = Path(args.tokenizer_path).resolve()
    contract['tokenizer'] = {name: sha256_file(tokenizer / name)
                             for name in ('tokenizer.json', 'tokenizer_config.json')}
    contract['implementation_sha256'] = {name: sha256_file(ROOT / name) for name in (
        'trainer/train_agent.py', 'trainer/evomind_tools.py', 'trainer/evomind_agent_tokens.py',
        'trainer/evomind_agent_runtime.py', 'trainer/evomind_rl_runtime.py')}
    return contract


def checkpoint_paths(args, config):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.save_weight):
        raise ValueError('save_weight must be a plain filename stem')
    save = owned_directory(args.save_dir)
    resume = owned_directory(args.resume_dir or (save / f'{args.save_weight}_runtime'))
    stem = f"{args.save_weight}_{config.hidden_size}{'_moe' if config.use_moe else ''}"
    if hasattr(args, 'init_dir') and hasattr(args, 'from_weight'):
        initial = Path(args.init_dir).resolve() / f"{args.from_weight}_{config.hidden_size}{'_moe' if config.use_moe else ''}.pth"
        if save / f'{stem}.pth' == initial:
            raise ValueError('Agent output must not overwrite its initial/reference checkpoint')
    return save / f'{stem}.pth', resume / f'{stem}_resume.pth', resume / 'state.json'


def load_checkpoint(path, contract):
    """Only load this project's explicitly selected, trusted local state file."""
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("Resume checkpoint must be an existing trusted file inside evomind")
    value = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(value, dict) or value.get('format') != FORMAT or value.get('contract') != contract:
        raise ValueError("Agent resume format/training contract differs; do not silently restart or convert")
    if value.get('pending_microsteps') != 0:
        raise ValueError("Agent checkpoint was not captured at an optimizer boundary")
    return value


def save_checkpoint(model, optimizer, scheduler, args, config, contract, *, epoch, step,
                    optimizer_updates, invocation_updates, status):
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Checkpoint requires optimizer.zero_grad(set_to_none=True) after the update")
    export_path, resume_path, state_path = checkpoint_paths(args, config)
    raw = getattr(model, '_orig_mod', model)
    state = {'format': FORMAT, 'contract': contract, 'model': cpu_tree(raw.state_dict()),
             'optimizer': cpu_tree(optimizer.state_dict()), 'scheduler': scheduler.state_dict(),
             'epoch': epoch, 'step': step, 'optimizer_updates': optimizer_updates,
             'pending_microsteps': 0, 'rng': capture_rng(args.device), 'status': status}
    # Commit the full-precision resumable state first. FP16 .pth remains the
    # upstream inference export, never the source of exact training resume.
    atomic_torch_save(state, resume_path)
    atomic_torch_save({key: value.half() if value.is_floating_point() else value
                       for key, value in state['model'].items()}, export_path)
    receipt = {'format': FORMAT, 'status': status, 'epoch': epoch, 'step': step,
                 'optimizer_updates': optimizer_updates, 'invocation_updates': invocation_updates,
                 'pending_microsteps': 0, 'resume': str(resume_path), 'export': str(export_path),
                 'resume_sha256': sha256_file(resume_path), 'export_sha256': sha256_file(export_path),
                 'contract': contract,
                 'scope': 'max_steps is a probe stop, not completion of full training'}
    atomic_json(receipt, state_path)
    receipt = {**receipt, 'status': ('probe_complete' if args.max_steps > 0 else 'completed')
               if status in ('complete', 'probe_complete') else status,
               'probe_only': args.max_steps > 0,
               'stop_reason': ('max_steps' if args.max_steps > 0 and invocation_updates >= args.max_steps
                               else 'epochs_complete' if status == 'complete' else None),
               'optimizer_updates_this_invocation': invocation_updates}
    atomic_json(receipt, export_path.with_suffix('.runtime.json'))
