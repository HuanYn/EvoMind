"""CPU-only checks for the native dense single-image launch contract."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dense_single_launcher_under_test", ROOT / "scripts" / "launch_dense_single.py"
)
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)

REJECTED = (ValueError, RuntimeError)
SELECTED_UUID = "GPU-00000000-0000-0000-0000-000000000001"
OTHER_UUID = "GPU-00000000-0000-0000-0000-000000000002"


class AttestationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.weights = self.root / "selected_text.pth"
        self.weights.write_bytes(b"selected native text checkpoint fixture\n")
        self.weights_sha256 = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        self.receipt_path = self.root / "acceptance.json"

    def write_receipt(self, status="accepted_text_screening", selected=None):
        receipt = {
            "status": status,
            "selected": selected if selected is not None else {"sha256": self.weights_sha256},
        }
        self.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return receipt

    def test_acceptance_is_bound_to_exact_receipt_and_weight_bytes(self):
        receipt = self.write_receipt()
        result = launcher.verify_attestation(self.receipt_path, self.weights)
        self.assertEqual(result["receipt"], receipt)
        self.assertEqual(result["weights_sha256"], self.weights_sha256)
        self.assertEqual(
            result["receipt_sha256"],
            hashlib.sha256(self.receipt_path.read_bytes()).hexdigest(),
        )

    def test_unaccepted_receipts_cannot_open_the_vision_gate(self):
        for status in ("pending", "failed", "accepted", None):
            with self.subTest(status=status):
                self.write_receipt(status=status)
                with self.assertRaises(REJECTED):
                    launcher.verify_attestation(self.receipt_path, self.weights)

    def test_checkpoint_changed_after_acceptance_is_rejected(self):
        self.write_receipt()
        self.weights.write_bytes(b"different checkpoint\n")
        with self.assertRaises(REJECTED):
            launcher.verify_attestation(self.receipt_path, self.weights)

    def test_receipt_without_selected_weight_digest_is_rejected(self):
        self.write_receipt(selected={})
        with self.assertRaises(REJECTED):
            launcher.verify_attestation(self.receipt_path, self.weights)


class ProbePlanTests(unittest.TestCase):
    @staticmethod
    def report(batch_size=4, accumulation_steps=1, status="success", iterations=2):
        report = {
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "status": status,
        }
        if iterations is not None:
            report["iterations"] = [
                {"optimizer_update": index + 1, "last_microbatch_loss": 2.3}
                for index in range(iterations)
            ]
        return report

    def test_initial_probe_uses_batch_four(self):
        self.assertEqual(launcher.choose_probe_plan([]), (4, 1))

    def test_interrupted_probe_retries_its_config_without_an_oom_adaptation(self):
        reports = [self.report(status="interrupted", iterations=1)]
        self.assertEqual(launcher.choose_probe_plan(reports), (4, 1))

    def test_observed_batch_four_oom_enables_batch_one_accumulation_four(self):
        reports = [self.report(status="oom", iterations=0)]
        self.assertEqual(launcher.choose_probe_plan(reports), (1, 4))

    def test_interrupted_fallback_retains_its_observed_oom_evidence(self):
        reports = [
            self.report(status="oom", iterations=0),
            self.report(1, 4, status="interrupted", iterations=1),
        ]
        self.assertEqual(launcher.choose_probe_plan(reports), (1, 4))

    def test_exact_two_iteration_success_finishes_the_probe(self):
        histories = (
            [self.report()],
            [self.report(status="oom", iterations=0), self.report(1, 4)],
        )
        for reports in histories:
            with self.subTest(reports=reports):
                self.assertIsNone(launcher.choose_probe_plan(reports))

    def test_success_label_without_exactly_two_iterations_does_not_finish_probe(self):
        for iterations in (0, 1, 3, None):
            with self.subTest(iterations=iterations):
                try:
                    plan = launcher.choose_probe_plan([self.report(iterations=iterations)])
                except REJECTED:
                    continue
                self.assertIsNotNone(plan)

    def test_other_training_failures_do_not_enable_an_oom_fallback(self):
        for batch_size, accumulation_steps in ((4, 1), (1, 4)):
            with self.subTest(batch_size=batch_size):
                reports = [] if batch_size == 4 else [self.report(status="oom", iterations=0)]
                reports.append(self.report(batch_size, accumulation_steps, status="failed", iterations=0))
                with self.assertRaises(REJECTED):
                    launcher.choose_probe_plan(reports)

    def test_both_configs_oom_prevents_full_training(self):
        reports = [
            self.report(status="oom", iterations=0),
            self.report(1, 4, status="oom", iterations=0),
        ]
        with self.assertRaises(REJECTED):
            launcher.choose_probe_plan(reports)


class MetricTests(unittest.TestCase):
    @staticmethod
    def native_line(epoch=1, step=100, iters=636245):
        return (
            f"Epoch:[{epoch}/2]({step}/{iters}), loss: 2.3456, "
            "logits_loss: 2.3456, aux_loss: 0.0000, "
            "lr: 0.00000500, epoch_time: 141.5min\n"
        )

    def test_native_console_metric_preserves_invocation_and_loss(self):
        metric = launcher.parse_metric(self.native_line(), 1, "formal-invocation-1")
        self.assertIsNotNone(metric)
        self.assertEqual(metric["epoch"], 1)
        self.assertEqual(metric["microstep"], 100)
        self.assertEqual(metric["optimizer_update"], 100)
        self.assertEqual(metric["invocation"], "formal-invocation-1")
        self.assertAlmostEqual(metric["loss"], 2.3456)

    def test_microstep_is_not_mislabeled_as_optimizer_update_under_accumulation(self):
        metric = launcher.parse_metric(self.native_line(iters=2544979), 4, "fallback")
        self.assertEqual(metric["microstep"], 100)
        self.assertEqual(metric["optimizer_update"], 25)

    def test_second_epoch_update_count_includes_previous_partial_accumulation(self):
        metric = launcher.parse_metric(self.native_line(epoch=2, iters=2544979), 4, "resumed")
        self.assertEqual(metric["optimizer_update"], 636245 + 25)

    def test_final_microstep_counts_the_native_tail_optimizer_update(self):
        metric = launcher.parse_metric(
            self.native_line(epoch=2, step=2544979, iters=2544979), 4, "fallback"
        )
        self.assertEqual(metric["optimizer_update"], 1272490)

    def test_nonfinite_loss_is_rejected_before_writing_metrics(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                line = self.native_line().replace("loss: 2.3456,", f"loss: {value},", 1)
                with self.assertRaises(REJECTED):
                    launcher.parse_metric(line, 1, "invalid")

    def test_initialization_messages_are_not_metrics(self):
        for line in ("", "Loading checkpoint...", "Trainable parameters: 15931776", "Epoch 1 started"):
            with self.subTest(line=line):
                self.assertIsNone(launcher.parse_metric(line, 1, "invocation"))


class ProbeCudaInitializationTests(unittest.TestCase):
    class FakeCuda:
        def __init__(self, available=True, count=1, init_error=None):
            self.available, self.count, self.init_error = available, count, init_error
            self.calls = []
            self.initialized = False

        def is_available(self):
            return self.available

        def device_count(self):
            return self.count

        def set_device(self, device):
            self.calls.append(("set_device", device))

        def init(self):
            self.calls.append(("init",))
            if self.init_error:
                raise self.init_error
            self.initialized = True

        def reset_peak_memory_stats(self, device):
            if not self.initialized:
                raise RuntimeError("Invalid device argument: CUDA allocator not initialized")
            self.calls.append(("reset_peak_memory_stats", device))

    def test_context_is_initialized_before_resetting_allocator_statistics(self):
        cuda = self.FakeCuda()
        launcher.initialize_probe_cuda(SimpleNamespace(cuda=cuda))
        self.assertEqual(cuda.calls, [("set_device", 0), ("init",), ("reset_peak_memory_stats", 0)])

    def test_missing_or_ambiguous_visible_device_does_not_initialize_cuda(self):
        for available, count in ((False, 0), (True, 0), (True, 2)):
            with self.subTest(available=available, count=count):
                cuda = self.FakeCuda(available=available, count=count)
                with self.assertRaises(RuntimeError):
                    launcher.initialize_probe_cuda(SimpleNamespace(cuda=cuda))
                self.assertEqual(cuda.calls, [])

    def test_initialization_failure_does_not_reset_allocator_statistics(self):
        cuda = self.FakeCuda(init_error=RuntimeError("CUDA context initialization failed"))
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            launcher.initialize_probe_cuda(SimpleNamespace(cuda=cuda))
        self.assertEqual(cuda.calls, [("set_device", 0), ("init",)])


class CudaUuidTests(unittest.TestCase):
    def test_torch_uuid_with_or_without_nvidia_prefix_matches_assigned_card(self):
        for observed in (SELECTED_UUID, SELECTED_UUID.removeprefix("GPU-")):
            with self.subTest(observed=observed):
                self.assertTrue(launcher.cuda_uuid_matches(observed, SELECTED_UUID))

    def test_normalization_does_not_accept_a_different_card(self):
        self.assertFalse(launcher.cuda_uuid_matches(OTHER_UUID.removeprefix("GPU-"), SELECTED_UUID))


class FakeNvidiaSmi:
    """Answer requested CSV columns without inspecting any real hardware."""

    def __init__(self, processes=(), memory_used=1, recovery_action="None"):
        self.calls = []
        self.processes = processes
        self.hardware = {
            "uuid": SELECTED_UUID,
            "index": "2",
            "name": "NVIDIA GeForce RTX 3090",
            "memory.used": str(memory_used),
            "memory.free": str(24576 - memory_used),
            "memory.total": "24576",
            "utilization.gpu": "0",
            "gpu_recovery_action": recovery_action,
        }

    @staticmethod
    def option(argv, name):
        for index, argument in enumerate(argv):
            if argument.startswith(name + "="):
                return argument.split("=", 1)[1]
            if argument == name:
                return argv[index + 1]
        return None

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        selected = self.option(argv, "--id") or self.option(argv, "-i")
        fields = self.option(argv, "--query-gpu")
        if fields is not None:
            if selected != SELECTED_UUID:
                raise AssertionError("Hardware availability must query the selected GPU UUID")
            rows = [self.hardware]
        else:
            fields = self.option(argv, "--query-compute-apps")
            if fields is None:
                raise AssertionError(f"Unexpected nvidia-smi invocation: {argv}")
            rows = [
                {"gpu_uuid": gpu_uuid, "pid": str(pid), "used_memory": "20480",
                 "used_gpu_memory": "20480", "process_name": "other_training.py"}
                for gpu_uuid, pid in self.processes
                if selected is None or gpu_uuid == selected
            ]
        output = "\n".join(
            ", ".join(row[field.strip()] for field in fields.split(",")) for row in rows
        )
        return subprocess.CompletedProcess(argv, 0, stdout=output + ("\n" if output else ""), stderr="")


class GpuGuardTests(unittest.TestCase):
    def test_idle_selected_gpu_is_available_while_other_users_card_is_busy(self):
        command = FakeNvidiaSmi(processes=[(OTHER_UUID, 99999)])
        hardware = launcher.gpu_guard(SELECTED_UUID, command=command)
        self.assertIsInstance(hardware, dict)
        self.assertEqual(hardware["uuid"], SELECTED_UUID)
        self.assertEqual(hardware["physical_index"], 2)
        self.assertTrue(command.calls)

    def test_live_process_on_selected_gpu_blocks_launch(self):
        command = FakeNvidiaSmi(processes=[(OTHER_UUID, 99999), (SELECTED_UUID, 12345)])
        with self.assertRaises(REJECTED):
            launcher.gpu_guard(SELECTED_UUID, command=command)

    def test_physical_index_cannot_replace_explicit_uuid_binding(self):
        command = FakeNvidiaSmi()
        with self.assertRaises(REJECTED):
            launcher.gpu_guard("2", command=command)
        self.assertEqual(command.calls, [])


class NativeCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name)
        self.data = str(self.run_dir / "official data" / "train.parquet")
        (self.run_dir / "contract.json").write_text(
            json.dumps({"data": {"path": self.data}}), encoding="utf-8"
        )

    def test_both_probe_choices_preserve_the_actual_native_training_recipe(self):
        for batch, accumulation in ((4, 1), (1, 4)):
            with self.subTest(batch=batch):
                argv = launcher.native_argv(
                    self.run_dir, {"batch_size": batch, "accumulation_steps": accumulation}
                )
                self.assertEqual(Path(argv[0]).resolve(), ROOT / "trainer" / "train_sft_vlm.py")
                self.assertEqual(len(argv[1:]) % 2, 0)
                flags = dict(zip(argv[1::2], argv[2::2]))
                expected = {
                    "--epochs": "2", "--batch_size": str(batch),
                    "--accumulation_steps": str(accumulation),
                    "--learning_rate": "5e-6", "--hidden_size": "768",
                    "--num_hidden_layers": "8", "--max_seq_len": "768",
                    "--freeze_llm": "1", "--dtype": "bfloat16", "--device": "cuda:0",
                    "--num_workers": "0", "--log_interval": "100", "--save_interval": "1000",
                    "--data_path": self.data,
                }
                for flag, value in expected.items():
                    self.assertEqual(flags[flag], value, flag)
                self.assertNotIn("--from_resume", flags)

    def test_resume_uses_native_optimizer_checkpoint_restore(self):
        selection = {"batch_size": 4, "accumulation_steps": 1}
        fresh = launcher.native_argv(self.run_dir, selection)
        resumed = launcher.native_argv(self.run_dir, selection, resume=True)
        self.assertEqual(resumed, fresh + ["--from_resume", "1"])


class OwnershipTests(unittest.TestCase):
    def test_second_launcher_cannot_acquire_the_same_live_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "gpu.lock"
            with launcher.run_lock(lock):
                with self.assertRaises(RuntimeError):
                    with launcher.run_lock(lock):
                        self.fail("A concurrent launcher acquired an owned GPU lock")
            with launcher.run_lock(lock):
                self.assertTrue(lock.is_file())

    def test_lock_is_released_after_owner_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "run.lock"
            with self.assertRaisesRegex(ValueError, "fixture failure"):
                with launcher.run_lock(lock):
                    raise ValueError("fixture failure")
            with launcher.run_lock(lock):
                self.assertTrue(lock.is_file())


if __name__ == "__main__":
    unittest.main()
