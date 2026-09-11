"""Offline CPU tests; no real reward weights, dataset download or CUDA discovery."""
import datasets  # noqa: F401 -- Windows DLL ordering
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BatchEncoding

from trainer import evomind_rl_runtime as runtime
from trainer.trainer_utils import LMForRewardModel


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 6)
        self.head = torch.nn.Linear(6, 13)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(torch.tanh(self.embedding(input_ids))), aux_loss=torch.tensor(0.0))


class TinyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 6)
        self.head = torch.nn.Linear(6, 1)

    def forward(self, input_ids, attention_mask=None):
        return self.head(torch.tanh(self.embedding(input_ids))).squeeze(-1)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, prompts, **kwargs):
        ids = torch.tensor([[1, 3 + int(prompt.split(":")[0]) % 8, 3 + int(prompt[-1])] for prompt in prompts])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def batch_decode(self, ids, **kwargs):
        return [str(row.tolist()) for row in ids]


class TinyDataset:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {"prompt": f"{index}:{int(random.random() < .9)}"}


class TinyEngine:
    def __init__(self, actor):
        self.actor = actor
        self.calls = 0

    def update_policy(self, actor):
        self.actor = actor

    def rollout(self, *, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature):
        self.calls += 1
        ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        prompt_len = ids.size(1)
        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                probs = torch.softmax(self.actor(ids).logits[:, -1] / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)
                ids = torch.cat([ids, sample], dim=1)
            ids = torch.cat([ids, ids.new_full((len(ids), 1), 2)], dim=1)
            positions = torch.arange(max_new_tokens)[None] + prompt_len - 1
            positions = positions.expand(len(ids), -1)
            logps, _ = runtime.masked_completion_logps(self.actor, ids, ids.ne(0).long(), positions)
        completion = ids[:, prompt_len:]
        return SimpleNamespace(output_ids=ids, completion_ids=completion, per_token_logps=logps,
                               completions=[str(row.tolist()) for row in completion],
                               prompt_lens=ids.new_full((len(ids),), prompt_len), completion_mask=torch.ones_like(completion))


def arguments(directory, **changes):
    value = dict(device="cpu", reward_device="cpu", reward_dtype="float32", resume_dir=str(directory),
                 save_dir=str(directory), save_weight="test", hidden_size=6, use_moe=0,
                 resume=None, from_resume=0, seed=42, epochs=1, batch_size=2,
                 accumulation_steps=1, max_steps=0, max_seq_len=3, max_gen_len=4, num_generations=3,
                 save_interval=1, ratio_diagnostics=False, updates_per_rollout=1,
                 beta=.1, loss_type="grpo", epsilon=.2, epsilon_high=5., grad_clip=1.,
                 gamma=1., lam=.95, mini_batch_size=1, ppo_update_iters=2,
                 clip_epsilon=.2, kl_coef=.02, cliprange_value=.2, vf_coef=.5, early_stop_kl=10.,
                 dtype="bfloat16")
    value.update(changes)
    return SimpleNamespace(**value)


def prepared(args, algorithm, count=4):
    runtime.seed_rng(91, "cpu")
    actor = TinyActor()
    reference = copy.deepcopy(actor).eval().requires_grad_(False)
    models = {"actor": actor}
    if algorithm == "ppo":
        models["critic"] = TinyCritic()
    optimizers = {name: runtime.CPUOnlyAdamW(model.parameters(), lr=.001) for name, model in models.items()}
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=.0001)
                  for name, opt in optimizers.items()}
    session = runtime.TrainingSession(args, algorithm, models, optimizers, schedulers,
                                      {"fixture": 1, "loss_type": args.loss_type,
                                       "updates_per_rollout": getattr(args, "updates_per_rollout", 1)},
                                      total_batches=(count + args.batch_size - 1) // args.batch_size)
    engine = TinyEngine(actor)
    components = dict(actor=actor, reference=reference, tokenizer=TinyTokenizer(), dataset=TinyDataset(count),
                      engine=engine, reward_fn=lambda prompts, responses: torch.tensor([
                          (sum(ord(character) for character in response) % 19) / 10 for response in responses]))
    if algorithm == "ppo":
        components["critic"] = models["critic"]
    return session, components


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def temporary(self):
        directory = tempfile.TemporaryDirectory(prefix="rl_cpu_test_", dir=Path(__file__).parent)
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def setUp(self):
        patch = mock.patch("builtins.print")
        patch.start()
        self.addCleanup(patch.stop)
        cuda_guard = mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU suite must not initialize CUDA"))
        cuda_guard.start()
        self.addCleanup(cuda_guard.stop)

    def test_grpo_and_cispo_match_direct_upstream_objective_and_gradients(self):
        for kind in ("grpo", "cispo"):
            torch.manual_seed(3)
            policy = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
            comparison = policy.detach().clone().requires_grad_()
            old, ref = torch.randn_like(policy), torch.randn_like(policy)
            rewards = torch.tensor([1., 2., -1., .5, .5, .5], dtype=torch.float64)
            mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]] * 2, dtype=torch.float64)
            actual, advantages = runtime.grpo_objective(policy, old, ref, rewards, mask, generations=3,
                beta=.1, loss_type=kind, epsilon=.2, epsilon_high=5.)
            grouped = rewards.view(-1, 3)
            expected_adv = (rewards - grouped.mean(1).repeat_interleave(3)) / (grouped.std(1, unbiased=False).repeat_interleave(3) + 1e-4)
            ratio = torch.exp(comparison - old)
            delta = ref - comparison
            kl = torch.exp(delta) - delta - 1
            if kind == "cispo":
                tokens = -(ratio.clamp(max=5.).detach() * expected_adv[:, None] * comparison - .1 * kl)
            else:
                tokens = -(torch.min(ratio * expected_adv[:, None], ratio.clamp(.8, 1.2) * expected_adv[:, None]) - .1 * kl)
            expected = ((tokens * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
            actual.backward()
            expected.backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(policy.grad, comparison.grad, rtol=0, atol=0)
            torch.testing.assert_close(advantages, expected_adv, rtol=0, atol=0)
            self.assertTrue(torch.equal(advantages[3:], torch.zeros(3, dtype=torch.float64)))

    def test_ppo_matches_direct_upstream_objective_and_gradients(self):
        torch.manual_seed(8)
        logp = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        values = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        ref_logp, ref_values = logp.detach().clone().requires_grad_(), values.detach().clone().requires_grad_()
        old, reference, old_v, advantages, returns = [torch.randn_like(logp) for _ in range(5)]
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float64)
        p, v, _, _ = runtime.ppo_objective(logp, old, reference, values, old_v, advantages, returns, mask, mask,
            clip_epsilon=.2, kl_coef=.02, cliprange_value=.2)
        ratio = torch.exp(ref_logp - old)
        delta = reference - ref_logp
        expected_p = (torch.max(-advantages * ratio, -advantages * ratio.clamp(.8, 1.2)) * mask).sum() / mask.sum() + .02 * ((delta.exp() - delta - 1) * mask).sum() / mask.sum()
        expected_v = .5 * (torch.max((ref_values - returns) ** 2, (ref_values.clamp(old_v - .2, old_v + .2) - returns) ** 2) * mask).sum() / mask.sum()
        (p + .5 * v).backward()
        (expected_p + .5 * expected_v).backward()
        torch.testing.assert_close(p, expected_p, rtol=0, atol=0)
        torch.testing.assert_close(v, expected_v, rtol=0, atol=0)
        torch.testing.assert_close(logp.grad, ref_logp.grad, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(values.grad, ref_values.grad, rtol=0, atol=0)

    def test_grpo_resume_equal_with_accumulation_tail(self):
        directory = self.temporary()
        for kind in ("grpo", "cispo"):
            full_args = arguments(directory / f"full_{kind}", epochs=2, accumulation_steps=2, loss_type=kind)
            full, components = prepared(full_args, "grpo", count=5)
            runtime.run_grpo_prepared(full, **components)
            split_args = arguments(directory / f"split_{kind}", epochs=2, accumulation_steps=2, max_steps=2, loss_type=kind)
            first, components = prepared(split_args, "grpo", count=5)
            runtime.run_grpo_prepared(first, **components)
            self.assertEqual(first.invocation_updates, 2)
            resumed_args = arguments(directory / f"split_{kind}", epochs=2, accumulation_steps=2, from_resume=1, max_steps=100, loss_type=kind)
            resumed, components = prepared(resumed_args, "grpo", count=5)
            runtime.run_grpo_prepared(resumed, **components)
            self.assertEqual(resumed.updates, full.updates)
            for key, value in full.models["actor"].state_dict().items():
                self.assertTrue(torch.equal(value, resumed.models["actor"].state_dict()[key]), key)
            self.assertEqual(resumed.schedulers["actor"].state_dict(), full.schedulers["actor"].state_dict())

    def test_legacy_prepared_arguments_preserve_explicit_k1_behavior(self):
        directory = self.temporary()
        for kind in ("grpo", "cispo"):
            options = dict(loss_type=kind, accumulation_steps=2)
            explicit, components = prepared(arguments(directory / f"explicit_{kind}", **options), "grpo", count=5)
            runtime.run_grpo_prepared(explicit, **components)
            legacy_args = arguments(directory / f"legacy_{kind}", **options)
            del legacy_args.updates_per_rollout
            del legacy_args.ratio_diagnostics
            legacy, components = prepared(legacy_args, "grpo", count=5)
            runtime.run_grpo_prepared(legacy, **components)
            self.assertEqual(legacy.updates, explicit.updates)
            torch.testing.assert_close(legacy.models["actor"].state_dict(), explicit.models["actor"].state_dict(),
                                       rtol=0, atol=0)
            torch.testing.assert_close(legacy.optimizers["actor"].state_dict(), explicit.optimizers["actor"].state_dict(),
                                       rtol=0, atol=0)
            self.assertEqual(legacy.schedulers["actor"].state_dict(), explicit.schedulers["actor"].state_dict())

    def test_k4_resume_reuses_frozen_rollout_and_matches_uninterrupted_run(self):
        directory = self.temporary()

        def instrument_rewards(components):
            original = components["reward_fn"]

            def stochastic_rewards(prompts, responses):
                rewards = original(prompts, responses)
                # Exercise both RNG streams and make accidental rescoring visible.
                return rewards + torch.rand_like(rewards) / 10 + random.random() / 10

            components["reward_fn"] = mock.Mock(side_effect=stochastic_rewards)
            return components["reward_fn"]

        def metrics(session):
            records = [json.loads(line) for line in (session.directory / "metrics.jsonl").read_text().splitlines()]
            invocation_fields = {"invocation_id", "invocation_update", "resume_start_optimizer_update", "elapsed_seconds"}
            return [{key: value for key, value in row.items() if key not in invocation_fields} for row in records]

        for kind in ("grpo", "cispo"):
            with self.subTest(loss_type=kind):
                # Two batches (one short tail), two epochs, four updates per batch.
                options = dict(loss_type=kind, epochs=2, updates_per_rollout=4, ratio_diagnostics=True)
                full, components = prepared(arguments(directory / f"full_{kind}", **options), "grpo", count=3)
                full_rewards = instrument_rewards(components)
                runtime.run_grpo_prepared(full, **components)
                self.assertEqual(components["engine"].calls, 4)
                self.assertEqual(full_rewards.call_count, 4)
                full_state = torch.load(full.path, map_location="cpu", weights_only=False)

                split_dir = directory / f"split_{kind}"
                first, components = prepared(arguments(split_dir, max_steps=2, **options), "grpo", count=3)
                first_rewards = instrument_rewards(components)
                runtime.run_grpo_prepared(first, **components)
                self.assertEqual(components["engine"].calls, 1)
                self.assertEqual(first_rewards.call_count, 1)
                first_state = torch.load(first.path, map_location="cpu", weights_only=False)
                pending = first_state["cursor"]["pending"]
                self.assertEqual(pending["replay_index"], 2)
                self.assertEqual(pending["replay_total"], 4)
                self.assertEqual(first_state["cursor"]["next_batch"], 1)
                summary = json.loads((split_dir / "summary.json").read_text())
                self.assertTrue(summary["pending_rollout"])
                self.assertFalse(summary["pending_ppo_rollout"])

                # Resume for only replay 3: neither rollout sampling nor reward
                # scoring is permitted while the saved group is still pending.
                middle, components = prepared(arguments(split_dir, from_resume=1, max_steps=1, **options), "grpo", count=3)
                components["engine"].rollout = mock.Mock(side_effect=AssertionError("Pending rollout resampled"))
                components["reward_fn"] = mock.Mock(side_effect=AssertionError("Pending reward rescored"))
                runtime.run_grpo_prepared(middle, **components)
                components["engine"].rollout.assert_not_called()
                components["reward_fn"].assert_not_called()
                middle_state = torch.load(middle.path, map_location="cpu", weights_only=False)
                self.assertEqual(middle_state["cursor"]["pending"]["replay_index"], 3)
                for key in ("outputs", "positions", "mask", "old_logps", "ref_logps", "rewards", "advantages"):
                    self.assertTrue(torch.equal(pending[key], middle_state["cursor"]["pending"][key]), key)

                resumed, components = prepared(arguments(split_dir, from_resume=1, max_steps=100, **options), "grpo", count=3)
                resumed_rewards = instrument_rewards(components)
                runtime.run_grpo_prepared(resumed, **components)
                self.assertEqual(components["engine"].calls, 3)
                self.assertEqual(resumed_rewards.call_count, 3)
                self.assertEqual(resumed.updates, 16)
                self.assertTrue(resumed.exhausted())
                resumed_state = torch.load(resumed.path, map_location="cpu", weights_only=False)
                self.assertIsNone(resumed_state["cursor"]["pending"])
                for key in ("models", "optimizers", "schedulers"):
                    torch.testing.assert_close(resumed_state[key], full_state[key], rtol=0, atol=0)
                self.assertEqual(resumed_state["cursor"], full_state["cursor"])
                self.assertTrue(torch.equal(resumed_state["rng"]["torch"], full_state["rng"]["torch"]))
                self.assertEqual(resumed_state["rng"]["python"], full_state["rng"]["python"])
                np.testing.assert_equal(resumed_state["rng"]["numpy"], full_state["rng"]["numpy"])
                self.assertEqual(metrics(resumed), metrics(full))

    def test_ppo_resume_inside_rollout_preserves_actor_critic_rng_and_cursor(self):
        directory = self.temporary()
        full, components = prepared(arguments(directory / "full"), "ppo")
        runtime.run_ppo_prepared(full, **components)
        full_calls = components["engine"].calls
        first, components = prepared(arguments(directory / "split", max_steps=3), "ppo")
        runtime.run_ppo_prepared(first, **components)
        first_calls = components["engine"].calls
        self.assertEqual(first.invocation_updates, 3)
        state = torch.load(first.path, map_location="cpu", weights_only=False)
        self.assertIsNotNone(state["cursor"]["pending"])
        self.assertEqual(state["cursor"]["pending"]["mini_start"], 1)
        resumed, components = prepared(arguments(directory / "split", from_resume=1, max_steps=100), "ppo")
        runtime.run_ppo_prepared(resumed, **components)
        self.assertEqual(first_calls + components["engine"].calls, full_calls)
        self.assertEqual(resumed.updates, 8)
        for name in ("actor", "critic"):
            for key, value in full.models[name].state_dict().items():
                self.assertTrue(torch.equal(value, resumed.models[name].state_dict()[key]), (name, key))
            self.assertEqual(full.schedulers[name].state_dict(), resumed.schedulers[name].state_dict())
        self.assertTrue(resumed.exhausted())

    def test_ppo_resume_accumulation_tail_and_early_stop(self):
        directory = self.temporary()
        for threshold in (10., -1.):
            options = dict(epochs=2, accumulation_steps=3, early_stop_kl=threshold)
            full, components = prepared(arguments(directory / f"full_{threshold}", **options), "ppo", count=5)
            runtime.run_ppo_prepared(full, **components)
            first, components = prepared(arguments(directory / f"split_{threshold}", max_steps=2, **options), "ppo", count=5)
            runtime.run_ppo_prepared(first, **components)
            resumed, components = prepared(arguments(directory / f"split_{threshold}", from_resume=1, max_steps=100, **options), "ppo", count=5)
            runtime.run_ppo_prepared(resumed, **components)
            self.assertEqual(full.updates, resumed.updates)
            for name in ("actor", "critic"):
                for key, value in full.models[name].state_dict().items():
                    self.assertTrue(torch.equal(value, resumed.models[name].state_dict()[key]), (threshold, name, key))

    def test_max_steps_is_per_invocation_not_absolute(self):
        directory = self.temporary()
        first, components = prepared(arguments(directory, max_steps=2), "ppo")
        runtime.run_ppo_prepared(first, **components)
        second, components = prepared(arguments(directory, max_steps=2, from_resume=1), "ppo")
        runtime.run_ppo_prepared(second, **components)
        self.assertEqual(second.updates, 4)
        self.assertEqual(second.invocation_updates, 2)

    def test_cpu_rng_never_calls_cuda(self):
        with mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA discovery forbidden")), \
             mock.patch.object(torch.cuda, "get_rng_state", side_effect=AssertionError("CUDA forbidden")), \
             mock.patch.object(torch.cuda, "manual_seed", side_effect=AssertionError("CUDA forbidden")):
            runtime.seed_rng(123, "cpu")
            state = runtime.capture_rng("cpu")
            a = (random.random(), torch.rand(3))
            runtime.restore_rng(state, "cpu")
            self.assertEqual(a[0], random.random())
            self.assertTrue(torch.equal(a[1], torch.rand(3)))

    def test_nonfinite_gradient_fails_before_update(self):
        model = TinyActor()
        optimizer = runtime.CPUOnlyAdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        before = copy.deepcopy(model.state_dict())
        next(model.parameters()).grad = torch.full_like(next(model.parameters()), float("nan"))
        with self.assertRaises(FloatingPointError):
            runtime.checked_update([model], [optimizer], [scheduler], 1.)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
        self.assertEqual(len(optimizer.state), 0)

    def test_atomic_save_failure_preserves_previous_checkpoint(self):
        path = self.temporary() / "latest.pt"
        runtime.atomic_torch_save({"step": 1}, path)
        with mock.patch.object(torch, "save", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                runtime.atomic_torch_save({"step": 2}, path)
        self.assertEqual(torch.load(path, weights_only=True), {"step": 1})

    def test_resume_rejects_changed_contract_and_existing_run(self):
        directory = self.temporary()
        first, components = prepared(arguments(directory, max_steps=1), "ppo")
        runtime.run_ppo_prepared(first, **components)
        with self.assertRaises(FileExistsError):
            prepared(arguments(directory), "ppo")
        with self.assertRaisesRegex(ValueError, "contract changed"):
            prepared(arguments(directory, from_resume=1, max_steps=1, loss_type="cispo"), "ppo")

    def test_probe_checkpoint_cannot_be_promoted_to_full_run(self):
        directory = self.temporary()
        first, components = prepared(arguments(directory, max_steps=2), "ppo")
        runtime.run_ppo_prepared(first, **components)
        with self.assertRaisesRegex(ValueError, "Probe/full"):
            prepared(arguments(directory, from_resume=1, max_steps=0), "ppo")

    def test_checkpoint_cadence_three_keeps_metrics_at_every_update(self):
        directory = self.temporary()
        session, _ = prepared(arguments(directory, save_interval=3), "ppo")
        for update in (1, 2, 3):
            for model in session.models.values():
                for parameter in model.parameters():
                    parameter.grad = torch.zeros_like(parameter)
            runtime.checked_update(list(session.models.values()), list(session.optimizers.values()), list(session.schedulers.values()), 1.)
            session.commit({"loss": 0.})
            self.assertEqual(session.path.exists(), update == 3)
        records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["checkpoint_committed"] for row in records], [False, False, True])
        self.assertEqual(session.checkpoint_optimizer_update, 3)

    def test_probe_tail_is_saved_even_before_checkpoint_interval(self):
        directory = self.temporary()
        session, components = prepared(arguments(directory, max_steps=2, save_interval=100), "ppo")
        runtime.run_ppo_prepared(session, **components)
        state = torch.load(session.path, map_location="cpu", weights_only=False)
        self.assertEqual(state["optimizer_updates"], 2)
        self.assertTrue(state["probe_only"])

    def test_full_loops_match_upstream_on_tiny_fixed_rollouts(self):
        from trainer import train_grpo, train_ppo
        directory = self.temporary()
        for algorithm, loss_type, accumulation in (("grpo", "grpo", 2), ("grpo", "cispo", 2), ("ppo", "grpo", 1), ("ppo", "grpo", 3)):
            label = f"{algorithm}_{loss_type}_{accumulation}"
            overrides = dict(loss_type=loss_type, accumulation_steps=accumulation)
            ours, ours_components = prepared(arguments(directory / (label + "_ours"), **overrides), algorithm, count=5)
            if algorithm == "ppo":
                runtime.run_ppo_prepared(ours, **ours_components)
            else:
                runtime.run_grpo_prepared(ours, **ours_components)
            original, components = prepared(arguments(directory / (label + "_upstream"), **overrides), algorithm, count=5)
            args = original.args
            args.debug_mode, args.debug_log_ratio = False, False
            args.debug_interval, args.save_interval, args.log_interval = 9999, 9999, 9999
            runtime.seed_rng(args.seed, "cpu")
            order = torch.randperm(5).tolist()
            loader = [{"prompt": [components["dataset"][index]["prompt"] for index in order[start:start + args.batch_size]]}
                      for start in range(0, 5, args.batch_size)]
            shared = dict(args=args, tokenizer=components["tokenizer"], lm_config=SimpleNamespace(use_moe=False),
                          autocast_ctx=runtime.nullcontext(), Logger=lambda *a, **kw: None,
                          calculate_rewards=lambda prompts, responses, ignored: components["reward_fn"](prompts, responses))
            if algorithm == "grpo":
                shared.update(model=components["actor"], optimizer=original.optimizers["actor"], scheduler=original.schedulers["actor"])
                with mock.patch.multiple(train_grpo, create=True, **shared):
                    # iters is deliberately larger: avoid upstream native exports in this in-memory objective test.
                    train_grpo.grpo_train_epoch(0, loader, 9999, components["engine"], components["reference"], None)
            else:
                shared.update(actor_model=components["actor"], critic_model=components["critic"],
                              actor_optimizer=original.optimizers["actor"], critic_optimizer=original.optimizers["critic"])
                with mock.patch.multiple(train_ppo, create=True, **shared):
                    train_ppo.ppo_train_epoch(0, loader, 9999, components["engine"], components["reference"],
                        original.schedulers["actor"], original.schedulers["critic"], None)
            for name in ours.models:
                for key, value in ours.models[name].state_dict().items():
                    torch.testing.assert_close(value, original.models[name].state_dict()[key], rtol=0, atol=0,
                                               msg=f"{label}:{name}:{key}")
                self.assertEqual(ours.schedulers[name].state_dict(), original.schedulers[name].state_dict())

    def test_real_minimind_and_unmodified_torch_rollout_cpu_integration(self):
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        from trainer.train_ppo import CriticModel
        from trainer.rollout_engine import TorchRolloutEngine
        directory = self.temporary()
        for algorithm in ("grpo", "ppo"):
            args = arguments(directory / algorithm, max_steps=2, mini_batch_size=2)
            config = MiniMindConfig(hidden_size=16, num_hidden_layers=1, vocab_size=64,
                num_attention_heads=2, num_key_value_heads=1, intermediate_size=32, max_position_embeddings=32)
            actor = MiniMindForCausalLM(config)
            reference = copy.deepcopy(actor).eval().requires_grad_(False)
            models = {"actor": actor}
            if algorithm == "ppo":
                critic = CriticModel(config)
                missing = critic.load_state_dict(actor.state_dict(), strict=False)
                self.assertEqual(set(missing.missing_keys), {"value_head.weight", "value_head.bias"})
                self.assertEqual(missing.unexpected_keys, [])
                models["critic"] = critic
            optimizers = {name: runtime.CPUOnlyAdamW(model.parameters(), lr=.001) for name, model in models.items()}
            schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10) for name, opt in optimizers.items()}
            session = runtime.TrainingSession(args, algorithm, models, optimizers, schedulers,
                                              {"tiny_real_model": True}, total_batches=2)
            engine = TorchRolloutEngine(actor, TinyTokenizer(), device="cpu", autocast_ctx=runtime.nullcontext())
            components = dict(actor=actor, reference=reference, tokenizer=TinyTokenizer(), dataset=TinyDataset(4), engine=engine,
                reward_fn=lambda prompts, responses: torch.arange(len(responses), dtype=torch.float32) / 10)
            if algorithm == "ppo":
                runtime.run_ppo_prepared(session, critic=models["critic"], **components)
            else:
                runtime.run_grpo_prepared(session, **components)
            self.assertEqual(session.invocation_updates, 2)
            self.assertTrue(session.path.is_file())

    def test_reward_loader_uses_slow_local_tokenizer_and_original_score_method(self):
        self.assertIs(runtime.LocalRewardModel.get_score, LMForRewardModel.get_score)
        directory = self.temporary()
        fake_model = mock.MagicMock()
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model
        fake_model.requires_grad_.return_value = fake_model
        fake_model.get_score.return_value = 9.
        with mock.patch.object(runtime.importlib.util, "find_spec", return_value=object()), \
             mock.patch.object(runtime.AutoTokenizer, "from_pretrained", return_value="slow") as tokenizer, \
             mock.patch.object(runtime.AutoModel, "from_pretrained", return_value=fake_model):
            reward = runtime.LocalRewardModel(directory, device="cpu", dtype=torch.float32)
            self.assertFalse(tokenizer.call_args.kwargs["use_fast"])
            self.assertTrue(tokenizer.call_args.kwargs["local_files_only"])
            self.assertEqual(reward.get_score([{"role": "user", "content": "hello"}], "answer"), 3.)
            fake_model.requires_grad_.assert_called_with(False)

    def test_raw_nonfinite_reward_is_rejected_before_original_clamp(self):
        directory = self.temporary()
        for invalid in (float("nan"), float("inf"), -float("inf")):
            fake_model = mock.MagicMock()
            fake_model.to.return_value = fake_model
            fake_model.eval.return_value = fake_model
            fake_model.requires_grad_.return_value = fake_model
            fake_model.get_score.return_value = invalid
            with mock.patch.object(runtime.importlib.util, "find_spec", return_value=object()), \
                 mock.patch.object(runtime.AutoTokenizer, "from_pretrained", return_value="slow"), \
                 mock.patch.object(runtime.AutoModel, "from_pretrained", return_value=fake_model):
                reward = runtime.LocalRewardModel(directory)
                with self.assertRaises(FloatingPointError):
                    reward.get_score([], "answer")

    def test_completion_mask_includes_real_eos_and_excludes_after(self):
        result = SimpleNamespace(output_ids=torch.tensor([[1, 3, 4, 2, 0], [1, 3, 4, 5, 6]]),
            completion_ids=torch.tensor([[4, 2, 0], [4, 5, 6]]), prompt_lens=torch.tensor([2, 2]),
            completion_mask=torch.ones(2, 3), per_token_logps=torch.zeros(2, 3))
        state = runtime.completion_state(result, TinyTokenizer(), "cpu")
        self.assertTrue(torch.equal(state["mask"], torch.tensor([[1., 1., 0.], [1., 1., 1.]])))


if __name__ == "__main__":
    unittest.main()
