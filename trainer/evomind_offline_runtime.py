"""Local safety adapter for official DPO/LoRA/distillation; no loss changes.

Exports retain upstream FP16 weights (LoRA exports adapters only). Resume files
retain native parameter precision, optimizer/scaler and a strict input/config
contract. RNG replay is NOT promised: upstream reseeds/reorders each epoch and
SFT augmentation draws are not skipped/replayed. Full data/epochs are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch


class OfflineAdamW(torch.optim.AdamW):
    """Avoid Torch 2.13 CUDA capture probing for an entirely CPU optimizer.

    CUDA parameters keep the upstream implementation; Adam update math and
    defaults are unchanged. This also makes fail-fast CUDA-guarded CPU tests safe.
    """
    def _cuda_graph_capture_health_check(self):
        if any(parameter.is_cuda for group in self.param_groups for parameter in group["params"]):
            return super()._cuda_graph_capture_health_check()

    def _accelerator_graph_capture_health_check(self):
        if all(parameter.device.type == "cpu" for group in self.param_groups for parameter in group["params"]):
            return
        return super()._accelerator_graph_capture_health_check()


def add_offline_arguments(parser):
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Probe only: stop after N complete optimizer updates (N>=2); 0 preserves full epochs")
    parser.add_argument("--init_dir", default="../out", help="Strict base/reference/teacher weight directory")
    parser.add_argument("--resume_dir", default=None,
                        help="Default ../checkpoints for full training, SAVE_DIR/checkpoints for isolated probes")
    parser.add_argument("--seed", type=int, default=42)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, value, *, tensor=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        if tensor:
            with temporary.open("wb") as stream:
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class OfflineRuntime:
    def __init__(self, args, config, branch, weight, inputs):
        """inputs: [(role, model_config, prefix[, override_init_directory]), ...]."""
        self.args, self.config, self.branch = args, config, branch
        if args.max_steps < 0 or args.max_steps == 1:
            raise ValueError("--max_steps must be 0 (full) or >=2 complete optimizer updates (probe)")
        if min(args.batch_size, args.accumulation_steps, args.epochs, args.log_interval, args.save_interval) <= 0:
            raise ValueError("Batch/accumulation/epochs/log/save intervals must be positive")
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError("This audited local runtime requires world_size=1; do not silently convert resume steps")
        if args.max_steps and args.from_resume:
            raise ValueError("Probes start from verified base weights, never promote or resume probe checkpoints")
        if not weight or Path(weight).name != weight or weight in (".", ".."):
            raise ValueError("Output weight must be a plain filename prefix")
        save_dir = Path(args.save_dir).resolve()
        init_dir = Path(args.init_dir).resolve()
        resume_dir = Path(args.resume_dir).resolve() if args.resume_dir else (
            save_dir / "checkpoints" if args.max_steps else Path("../checkpoints").resolve())
        if args.max_steps and (save_dir == Path("../out").resolve() or save_dir == init_dir
                               or resume_dir == Path("../checkpoints").resolve()):
            raise ValueError("Probe output/resume directories must be isolated from normal out/checkpoints/init_dir")
        args.resume_dir = str(resume_dir)
        suffix = "_moe" if config.use_moe else ""
        stem = f"{weight}_{config.hidden_size}{suffix}"
        self.export_path = save_dir / f"{stem}.pth"
        self.resume_path = resume_dir / f"{stem}_resume.pth"
        self.report_path = save_dir / f"{stem}.runtime.json"
        if not args.from_resume and any(p.exists() for p in (self.export_path, self.resume_path, self.report_path)):
            raise FileExistsError("Outputs already exist: use a fresh directory/prefix or explicit --from_resume 1")
        required = {"data": Path(args.data_path).resolve()}
        for role, input_config, prefix, *directory in inputs:
            if prefix == "none" or not prefix or Path(prefix).name != prefix:
                raise ValueError(f"{role} requires an explicit real checkpoint prefix; random initialization is not allowed")
            input_suffix = "_moe" if input_config.use_moe else ""
            input_dir = Path(directory[0]).resolve() if directory and directory[0] else init_dir
            required[role] = input_dir / f"{prefix}_{input_config.hidden_size}{input_suffix}.pth"
        tokenizer_dir = Path("../model").resolve()
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(f"Tokenizer directory missing: {tokenizer_dir}")
        for path in tokenizer_dir.glob("*.json"):
            required[f"tokenizer/{path.name}"] = path
        for role, path in required.items():
            if not path.is_file():
                raise FileNotFoundError(f"Required {role} input missing: {path}")
            if path in (self.export_path, self.resume_path):
                raise ValueError(f"Output would overwrite input {role}: {path}")
        excluded = {"from_resume", "save_dir", "resume_dir", "init_dir", "device", "log_interval",
                    "save_interval", "use_wandb", "wandb_project", "data_path"}
        parameters = {key: value for key, value in vars(args).items() if key not in excluded}
        self.contract = {"version": 1, "branch": branch, "parameters": parameters,
            "inputs": {role: {"path": str(path), "sha256": sha256(path)} for role, path in required.items()},
            "world_size": 1, "torch_version": str(torch.__version__),
            "code_sha256": {str(path): sha256(path) for path in (
                Path(__file__).resolve(), Path(__file__).with_name(f"train_{branch}.py").resolve(),
                Path(__file__).with_name("trainer_utils.py").resolve(),
                Path(__file__).resolve().parents[1] / "dataset" / "lm_dataset.py",
                Path(__file__).resolve().parents[1] / "model" / "model_minimind.py",
                Path(__file__).resolve().parents[1] / "model" / "model_lora.py")}}
        self.updates, self.invocation_updates, self.microbatches = 0, 0, 0
        self.last_update = None
        self.started = None
        self.last_report = None

    @property
    def probe_complete(self):
        return bool(self.args.max_steps and self.invocation_updates >= self.args.max_steps)

    def load_resume(self):
        if not self.args.from_resume:
            return None
        if not self.resume_path.is_file():
            raise FileNotFoundError(f"Explicit resume checkpoint missing: {self.resume_path}")
        data = torch.load(self.resume_path, map_location="cpu", weights_only=True)
        required = {"model", "optimizer", "scaler", "epoch", "step", "world_size", "evomind"}
        if not isinstance(data, dict) or not required.issubset(data):
            raise ValueError("Resume lacks model/optimizer/scaler/epoch/step/world_size/strict evomind metadata")
        metadata = data["evomind"]
        if not isinstance(metadata, dict) or not {"contract", "probe_only", "optimizer_updates", "iters", "scaler_enabled"}.issubset(metadata):
            raise ValueError("Malformed evomind resume metadata")
        if metadata.get("probe_only") or metadata.get("contract") != self.contract:
            raise ValueError("Refusing probe promotion or changed batch/accumulation/data/model/tokenizer/loss resume contract")
        if data["world_size"] != 1 or not metadata.get("optimizer_boundary"):
            raise ValueError("Resume must be a single-process optimizer-boundary checkpoint")
        if not all(isinstance(data[key], dict) for key in ("model", "optimizer", "scaler")):
            raise ValueError("Malformed model/optimizer/scaler states")
        if (type(data["epoch"]) is not int or not 0 <= data["epoch"] < self.args.epochs
                or type(data["step"]) is not int or data["step"] <= 0):
            raise ValueError("Invalid saved epoch/microstep")
        if not data["optimizer"].get("state") or not data["optimizer"].get("param_groups"):
            raise ValueError("Resume does not contain initialized Adam optimizer state")
        if type(metadata["optimizer_updates"]) is not int or metadata["optimizer_updates"] <= 0:
            raise ValueError("Invalid completed optimizer-update count")
        for state in data["optimizer"]["state"].values():
            if not isinstance(state, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
                raise ValueError("Resume is missing Adam moments or step state")
            if any(torch.is_tensor(value) and not bool(torch.isfinite(value).all()) for value in state.values()):
                raise ValueError("Nonfinite saved optimizer state")
        if any(torch.is_tensor(value) and not bool(torch.isfinite(value).all()) for value in data["model"].values()):
            raise ValueError("Nonfinite saved model state")
        return data

    def restore(self, model, optimizer, scaler, data, dataset_length):
        iters = math.ceil(dataset_length / self.args.batch_size)
        if iters <= 0:
            raise ValueError("Training dataset is empty")
        if self.args.max_steps and iters < self.args.max_steps * self.args.accumulation_steps:
            raise ValueError("Probe requires N*accumulation_steps microbatches in one epoch; tail flush does not count")
        if data is None:
            return 0, 0
        metadata = data["evomind"]
        step, epoch = data["step"], data["epoch"]
        if metadata.get("iters") != iters or not 0 < step <= iters:
            raise ValueError("Saved data position/epoch length does not match the complete dataset")
        if step % self.args.accumulation_steps and step != iters:
            raise ValueError("Partial-accumulation checkpoints are allowed only at the completed epoch tail")
        if bool(scaler.is_enabled()) != metadata.get("scaler_enabled"):
            raise ValueError("Scaler mode changed on resume")
        model.load_state_dict(data["model"], strict=True)
        optimizer.load_state_dict(data["optimizer"])
        scaler.load_state_dict(data["scaler"])
        self.updates = metadata["optimizer_updates"]
        report = {**metadata, "epoch": epoch, "step": step}
        if epoch + 1 == self.args.epochs and step == iters:
            # save() publishes the native resume before the FP16 export. A
            # crash between those replaces leaves a final resume but a stale
            # (or absent) export. No epoch will execute after this restore, so
            # regenerate from the strictly loaded model before finish can emit
            # a completed receipt. LoRA must remain adapter-only.
            self.last_report = None
            native_state = self._native_state(model)
            atomic_write(self.export_path, self._export_state(native_state, self.branch == "lora"), tensor=True)
            report.update(self._file_binding(), recovered_final_export=True)
        self.last_report = report
        return (epoch + 1, 0) if step == iters else (epoch, step)

    def start_training(self):
        if str(self.args.device).startswith("cuda"):
            torch.cuda.synchronize(self.args.device)
            torch.cuda.reset_peak_memory_stats(self.args.device)
        self.started = time.perf_counter()

    def check_loss(self, loss):
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("Nonfinite offline-posttrain loss; refusing backward/update/checkpoint")
        self.microbatches += 1

    def update(self, model, optimizer, scaler, parameters, epoch, step, iters):
        parameters = list(parameters)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, self.args.grad_clip, error_if_nonfinite=True)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            raise FloatingPointError("Scaler skipped optimizer update; probe/full checkpoint refused")
        optimizer.zero_grad(set_to_none=True)
        self.updates += 1
        self.invocation_updates += 1
        self.last_update = (epoch, step, iters)

    def save(self, model, optimizer, scaler, epoch, step, iters, *, adapter_only=False):
        if self.last_update != (epoch, step, iters) or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Checkpoint requires a completed optimizer update and cleared gradients")
        native_state = self._native_state(model)
        export = self._export_state(native_state, adapter_only)
        metadata = {"contract": self.contract, "probe_only": bool(self.args.max_steps),
            "optimizer_boundary": True, "optimizer_updates": self.updates,
            "optimizer_updates_this_invocation": self.invocation_updates,
            "microbatches_this_invocation": self.microbatches, "iters": iters,
            "scaler_enabled": bool(scaler.is_enabled()),
            "resume_parameter_precision": "native; FP16 inference/adapter export is separate",
            "rng_policy": "Not bitwise replay: upstream epoch reseed + deterministic index shuffle; augmentation/worker RNG is not restored",
            "budget_policy": "max_steps=0 preserves official full dataset and epochs; partial tail loss still divided by configured accumulation_steps",
            "supervision_policy": "No validation/quality scores inferred from probe loss",
            "code_sha256": {str(Path(__file__).resolve()): sha256(__file__),
                str(Path(__file__).with_name(f"train_{self.branch}.py")): sha256(Path(__file__).with_name(f"train_{self.branch}.py"))}}
        payload = {"model": native_state, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                   "epoch": epoch, "step": step, "world_size": 1, "wandb_id": None, "evomind": metadata}
        # A failed publication must not leave an older in-memory report eligible
        # for finish(), even if callers accidentally catch the I/O exception.
        self.last_report = None
        atomic_write(self.resume_path, payload, tensor=True)
        atomic_write(self.export_path, export, tensor=True)
        self.last_report = {**metadata, "epoch": epoch, "step": step,
                            **self._file_binding(), "recovered_final_export": False}
        self._write_report("probe_complete" if self.probe_complete else "checkpoint")
        print(f"EVOMIND_CHECKPOINT optimizer_updates={self.updates} epoch={epoch + 1} microstep={step} probe_only={bool(self.args.max_steps)} path={self.export_path}", flush=True)

    @staticmethod
    def _native_state(model):
        raw = getattr(model, "module", model)
        raw = getattr(raw, "_orig_mod", raw)
        return {key: value.detach().cpu() for key, value in raw.state_dict().items()}

    @staticmethod
    def _export_state(native_state, adapter_only):
        export = {key: value.half() for key, value in native_state.items() if not adapter_only or ".lora." in key}
        if not export:
            raise ValueError("Refusing empty model/adapter export")
        return export

    def _file_binding(self):
        # These hashes belong to the external receipt, never to the resume
        # payload itself (which cannot include its own cryptographic hash).
        return {"export_path": str(self.export_path), "export_sha256": sha256(self.export_path),
                "resume_path": str(self.resume_path), "resume_sha256": sha256(self.resume_path)}

    def _write_report(self, status):
        report = dict(self.last_report or {})
        report.update(status=status, stop_reason="max_steps" if self.probe_complete else
                      "epochs_complete" if status == "completed" else "optimizer_boundary",
                      optimizer_updates_this_invocation=self.invocation_updates,
                      microbatches_this_invocation=self.microbatches,
                      elapsed_training_seconds=time.perf_counter() - self.started if self.started else None)
        if str(self.args.device).startswith("cuda"):
            torch.cuda.synchronize(self.args.device)
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.args.device)
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.args.device)
        atomic_write(self.report_path, report)

    def finish(self):
        if self.args.max_steps and not self.probe_complete:
            raise RuntimeError("Probe failed to complete the requested full optimizer updates")
        if self.last_report is None:
            raise RuntimeError("No optimizer-boundary checkpoint was produced/restored")
        if not self.args.max_steps and (self.last_report.get("epoch") != self.args.epochs - 1
                                       or self.last_report.get("step") != self.last_report.get("iters")):
            raise RuntimeError("Full training has not reached the final configured epoch tail")
        binding = self._file_binding()
        if any(self.last_report.get(key) != value for key, value in binding.items()):
            raise RuntimeError("Export/resume changed or missing file binding; refusing completed receipt")
        self._write_report("probe_complete" if self.probe_complete else "completed")
