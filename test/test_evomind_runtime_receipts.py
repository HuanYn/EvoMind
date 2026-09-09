"""Native-schema receipt fixtures only: stdlib, no tensor loading or GPU calls."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evomind_posttrain as pipeline
from evomind_run import atomic_json, sha256


def digest_contract(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def emit_receipt_fixture(branch, directory, *, probe, epochs=None, updates=2):
    """Mirror current runtime JSON schemas; .pth bytes are NOT tensor fixtures."""
    directory = Path(directory).resolve()
    runtime, name = branch["runtime"], branch["name"]
    export = directory / "weights" / f"{name}_768.pth"
    resume = (directory / "checkpoints/latest.resume.pt" if runtime == "rl" else
              directory / "checkpoints" / f"{name}_768_resume.pth")
    export.parent.mkdir(parents=True, exist_ok=True)
    resume.parent.mkdir(parents=True, exist_ok=True)
    export.write_bytes(b"stdlib receipt fixture: not real model weights")
    resume.write_bytes(b"stdlib receipt fixture: not a tensor checkpoint")
    epochs = branch["options"]["epochs"] if epochs is None else epochs
    candidate = branch["batch_candidates"][0]
    args = {**branch["options"], **candidate, "epochs": epochs,
            "max_steps": 2 if probe else 0, "device": "cpu",
            "save_dir": str(export.parent), "resume_dir": str(resume.parent),
            "init_dir": str(directory / "base"), "tokenizer_path": str(pipeline.ROOT / "model"),
            "data_path": str(pipeline.ROOT / branch["data"]), "from_resume": 0,
            "save_interval": 1 if probe else 100, "log_interval": 1 if probe else 100}
    args["lora_name" if name == "lora" else "save_weight"] = name
    total = max(updates, 8 * epochs) if not probe else updates
    data = {"status": "probe_complete" if probe else "completed", "probe_only": probe,
            "stop_reason": "max_steps" if probe else "epochs_complete",
            "optimizer_updates": total, "optimizer_updates_this_invocation": updates,
            "epoch": 0 if probe else epochs - 1, "step": 2 if probe else 8,
            "export_sha256": sha256(export), "resume_sha256": sha256(resume)}

    def file_record(path):
        return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}

    if runtime == "rl":
        algorithm = "ppo" if name == "ppo" else "grpo"
        args["evomind_runtime"] = True
        args["resume"] = None
        excluded = {"max_steps", "resume", "from_resume", "resume_dir", "save_dir", "debug_mode",
                    "debug_interval", "debug_log_ratio", "log_interval", "save_interval"}
        settings = {key: value for key, value in args.items() if key not in excluded}
        planned = 16 * epochs if algorithm == "ppo" else 8 * epochs
        contract = {"algorithm": algorithm, "settings": settings,
                    "initial": file_record(export), "data": file_record(resume),
                    "tokenizer": {"tokenizer.json": file_record(export)},
                    "reward_model": {"config.json": file_record(resume)},
                    "runtime_source": file_record(pipeline.ROOT / "trainer/evomind_rl_runtime.py"),
                    "upstream_source": file_record(pipeline.ROOT / "trainer" / branch["script"]),
                    "torch_version": "fixture-not-imported",
                    "hardware_adaptation": {"reward_device": "cpu", "reward_dtype": "float32",
                        "policy_recomputation": "full_batch_upstream", "data_workers": 0},
                    "planned_optimizer_steps": planned, "dataset_rows": candidate["batch_size"] * 8}
        config = {"algorithm": algorithm, "contract": contract, "invocation": args}
        atomic_json(resume.parent / "run_config.json", config)
        data = {"schema_version": "evomind.rl.run/1", "status": "probe_complete" if probe else "complete",
                "algorithm": algorithm, "loss_type": args.get("loss_type"),
                "contract_sha256": digest_contract(contract), "planned_optimizer_updates": planned,
                "probe_only": probe, "stop_reason": data["stop_reason"], "optimizer_updates": total,
                "invocation_optimizer_updates": updates, "resume_start_optimizer_update": total - updates,
                "checkpoint_optimizer_update": total, "requested_max_steps": args["max_steps"],
                "configured_epochs": epochs, "cursor_epoch": 0 if probe else epochs,
                "cursor_next_batch": 2 if probe else 0, "pending_ppo_rollout": False,
                "resume_checkpoint": str(resume), "final_checkpoint": str(export),
                "elapsed_seconds": 1.0, "reward_device": "cpu", "reward_dtype": "float32",
                "peak_allocated_bytes": None, "peak_reserved_bytes": None}
    elif runtime == "offline":
        excluded = {"from_resume", "save_dir", "resume_dir", "init_dir", "device", "log_interval",
                    "save_interval", "use_wandb", "wandb_project", "data_path"}
        sources = [pipeline.ROOT / "trainer/evomind_offline_runtime.py",
                   pipeline.ROOT / "trainer" / branch["script"]]
        code = {str(source): sha256(source) for source in sources}
        data.update(contract={"version": 1, "branch": name,
                              "parameters": {key: value for key, value in args.items() if key not in excluded},
                              "inputs": {"data": file_record(resume)}, "world_size": 1,
                              "torch_version": "fixture-not-imported", "code_sha256": code},
                    optimizer_boundary=True, iters=8, code_sha256=code,
                    export_path=str(export), resume_path=str(resume))
    else:
        fields = ("epochs", "batch_size", "learning_rate", "dtype", "accumulation_steps", "grad_clip",
                  "max_seq_len", "max_gen_len", "max_total_len", "num_generations", "beta", "loss_type",
                  "epsilon", "epsilon_high", "thinking_ratio", "max_turns", "seed", "reward_device", "reward_dtype")
        contract = {key: args[key] for key in fields}
        contract.update(dataset_rows=candidate["batch_size"] * 8, hidden_size=768, num_hidden_layers=8,
                        use_moe=False, device_type="cpu", probe_only=probe,
                        reward_model_path=args["reward_model_path"], initial_weights=file_record(export),
                        training_data=file_record(resume), tokenizer={"tokenizer.json": "0" * 64},
                        implementation_sha256={name: sha256(pipeline.ROOT / name) for name in (
                            "trainer/train_agent.py", "trainer/evomind_agent_runtime.py",
                            "trainer/evomind_rl_runtime.py")})
        data.update(format="evomind.agent.optimizer-boundary/1", contract=contract,
                    pending_microsteps=0, invocation_updates=updates,
                    export=str(export), resume=str(resume))
    receipt = pipeline.receipt_path(branch, directory)
    atomic_json(receipt, data)
    return receipt


class RuntimeReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.branches = {b["name"]: b for b in pipeline.build_plan()["branches"]}
        # Retain native PPO receipt regression coverage; this is NOT a queued training branch.
        self.branches["ppo"] = {**self.branches["grpo"], "name": "ppo", "script": "train_ppo.py",
                                "options": {k: v for k, v in self.branches["grpo"]["options"].items()
                                            if k not in ("loss_type", "num_generations")}}

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, name="grpo", *, probe=False, updates=2):
        branch = self.branches[name]
        directory = self.root / name
        path = emit_receipt_fixture(branch, directory, probe=probe, updates=updates)
        return branch, directory, path

    def change(self, path, **values):
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(values)
        atomic_json(path, data)

    def change_rl_contract(self, directory, path, mutator):
        config_path = directory / "checkpoints/run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        mutator(config)
        atomic_json(config_path, config)
        self.change(path, contract_sha256=digest_contract(config["contract"]))

    def test_native_schemas_accept_full_and_two_update_probe(self):
        for name in self.branches:
            for probe in (False, True):
                with self.subTest(name=name, probe=probe):
                    branch, directory, _ = self.fixture(name, probe=probe)
                    record = pipeline.check_receipt(branch, directory, probe=probe)
                    self.assertEqual(record["probe_only"], probe)
                    self.assertEqual(len(record["resume_checkpoint_sha256"]), 64)
                    self.assertEqual(len(record["contract_sha256"]), 64)

    def test_full_requires_explicit_boolean_probe_identity(self):
        for value in (None, 0, 1, "false", "", [], True):
            with self.subTest(value=value):
                branch, directory, path = self.fixture()
                self.change(path, probe_only=value)
                with self.assertRaisesRegex(ValueError, "Probe/full"):
                    pipeline.check_receipt(branch, directory, probe=False)
        branch, directory, path = self.fixture()
        data = json.loads(path.read_text()); del data["probe_only"]; atomic_json(path, data)
        with self.assertRaisesRegex(ValueError, "Probe/full"):
            pipeline.check_receipt(branch, directory, probe=False)

    def test_rl_rejects_wrong_algorithm_loss_or_contract_hash(self):
        for mutation in ({"algorithm": "ppo"}, {"loss_type": "cispo"}, {"contract_sha256": "0" * 64},
                         {"contract_sha256": None}, {"schema_version": "upstream"}):
            with self.subTest(mutation=mutation):
                branch, directory, path = self.fixture()
                self.change(path, **mutation)
                with self.assertRaises(ValueError):
                    pipeline.check_receipt(branch, directory, probe=False)

    def test_rehashed_rl_contract_cannot_change_G_or_authorized_batch(self):
        for key, value in (("num_generations", 2), ("batch_size", 3), ("save_weight", "cispo")):
            with self.subTest(key=key):
                branch, directory, path = self.fixture()
                def mutate(config):
                    config["contract"]["settings"][key] = value
                    config["invocation"][key] = value
                self.change_rl_contract(directory, path, mutate)
                with self.assertRaises(ValueError):
                    pipeline.check_receipt(branch, directory, probe=False)

    def test_rl_invocation_must_match_hashed_settings_and_output_paths(self):
        for key, value in (("num_generations", 2), ("max_steps", 2), ("save_dir", str(self.root)),
                           ("resume_dir", str(self.root))):
            with self.subTest(key=key):
                branch, directory, path = self.fixture()
                self.change_rl_contract(directory, path, lambda config: config["invocation"].update({key: value}))
                with self.assertRaises(ValueError):
                    pipeline.check_receipt(branch, directory, probe=False)

    def test_rl_committed_update_accounting_and_cursor_are_required(self):
        for mutation in ({"checkpoint_optimizer_update": 7}, {"checkpoint_optimizer_update": None},
                         {"optimizer_updates": True}, {"invocation_optimizer_updates": 9},
                         {"resume_start_optimizer_update": 0}, {"pending_ppo_rollout": None},
                         {"pending_ppo_rollout": True}, {"cursor_epoch": 0}, {"cursor_next_batch": 1},
                         {"planned_optimizer_updates": 999}, {"requested_max_steps": 2}):
            with self.subTest(mutation=mutation):
                branch, directory, path = self.fixture()
                self.change(path, **mutation)
                with self.assertRaises(ValueError):
                    pipeline.check_receipt(branch, directory, probe=False)

    def test_rl_requires_exact_nonempty_export_and_resume_paths(self):
        for key in ("final_checkpoint", "resume_checkpoint"):
            for bad in (None, str(self.root / "unrelated.pth")):
                with self.subTest(key=key, value=bad):
                    branch, directory, path = self.fixture()
                    self.change(path, **{key: bad})
                    with self.assertRaises(ValueError):
                        pipeline.check_receipt(branch, directory, probe=False)
        for relative in ("weights/grpo_768.pth", "checkpoints/latest.resume.pt"):
            branch, directory, _ = self.fixture()
            (directory / relative).write_bytes(b"")
            with self.assertRaises(FileNotFoundError):
                pipeline.check_receipt(branch, directory, probe=False)

    def test_ppo_kl_early_stop_does_not_require_nominal_update_equality(self):
        branch, directory, path = self.fixture("ppo")
        data = json.loads(path.read_text())
        self.assertLess(data["optimizer_updates"], data["planned_optimizer_updates"])
        pipeline.check_receipt(branch, directory, probe=False)

    def test_final_resume_can_complete_without_new_optimizer_update(self):
        for name in ("grpo", "ppo", "dpo", "agent_cispo"):
            with self.subTest(name=name):
                branch, directory, _ = self.fixture(name, updates=0)
                pipeline.check_receipt(branch, directory, probe=False)

    def test_probe_requires_two_completed_updates_not_just_a_status(self):
        for name in ("grpo", "ppo", "dpo", "agent_cispo"):
            with self.subTest(name=name):
                branch, directory, _ = self.fixture(name, probe=True, updates=1)
                with self.assertRaises(ValueError):
                    pipeline.check_receipt(branch, directory, probe=True)

    def test_offline_agent_both_hashes_are_required_and_bind_file_contents(self):
        for name in ("dpo", "lora", "distillation", "agent_cispo"):
            for key in ("export_sha256", "resume_sha256"):
                with self.subTest(name=name, key=key):
                    branch, directory, path = self.fixture(name)
                    self.change(path, **{key: None})
                    with self.assertRaisesRegex(ValueError, "SHA256"):
                        pipeline.check_receipt(branch, directory, probe=False)
                    branch, directory, path = self.fixture(name)
                    relative = f"weights/{name}_768.pth" if key == "export_sha256" else f"checkpoints/{name}_768_resume.pth"
                    (directory / relative).write_bytes(b"changed after receipt")
                    with self.assertRaisesRegex(ValueError, "SHA256"):
                        pipeline.check_receipt(branch, directory, probe=False)

    def test_offline_agent_reject_wrong_plan_fields_and_nonfinal_cursor(self):
        for name in ("dpo", "agent_cispo"):
            for field in ("epochs", "batch_size"):
                with self.subTest(name=name, field=field):
                    branch, directory, path = self.fixture(name)
                    data = json.loads(path.read_text())
                    params = data["contract"].get("parameters", data["contract"])
                    params[field] = 7
                    atomic_json(path, data)
                    with self.assertRaises(ValueError):
                        pipeline.check_receipt(branch, directory, probe=False)
            branch, directory, path = self.fixture(name)
            self.change(path, step=7)
            with self.assertRaises(ValueError):
                pipeline.check_receipt(branch, directory, probe=False)

    def test_offline_agent_reject_non_optimizer_boundary(self):
        for name, values in (("dpo", {"optimizer_boundary": False}),
                             ("agent_cispo", {"pending_microsteps": 1}),
                             ("agent_cispo", {"invocation_updates": 1})):
            with self.subTest(name=name, values=values):
                branch, directory, path = self.fixture(name)
                self.change(path, **values)
                with self.assertRaises(ValueError):
                    pipeline.check_receipt(branch, directory, probe=False)

    def test_modified_implementation_hash_is_not_accepted(self):
        branch, directory, path = self.fixture()
        self.change_rl_contract(directory, path, lambda config: config["contract"]["runtime_source"].update(sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "implementation"):
            pipeline.check_receipt(branch, directory, probe=False)


if __name__ == "__main__":
    unittest.main()
