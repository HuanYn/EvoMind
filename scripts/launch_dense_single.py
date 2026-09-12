"""Owned, offline launcher for the accepted native Dense single-image baseline.

Run in a dedicated screen session. The imported product receipt is a deployment
attestation, not permission to invent or repeat text evaluations. Only the named
GPU is checked. Probe updates are discarded; formal training starts afresh from
the accepted text checkpoint. Native resume restores optimizer/scaler/epoch/step,
but upstream does not persist augmentation RNG: resume is not bitwise replay.

The formal worker executes the actual trainer with runpy. Its hooks change only
input/checkpoint locations and snapshot retention, never the training math.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

VISION = Path(__file__).resolve().parents[1]
OFFICIAL_ROWS = 2544979
ENCODER_FILES = {
    "config.json": "5ad8dda7d55541c7749f9b1cc43fe8eb8c70d8664588d89f710242ce06b3167e",
    "preprocessor_config.json": "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff",
    "model.safetensors": "c1e9cc19ed6704b87353ee00b9ff5d6191886d741898339984364f789c62810d",
}
TOKENIZER_FILES = {
    "tokenizer.json": "8bf5868abfc7ea919186b57e2b411adefe3b0922b53052c5839937e166b9d395",
    "tokenizer_config.json": "e42b762555734279158ca292431c831fa387c48c35d9b1b205f4fc6c83b79d34",
}
SOURCE_FILES = ("scripts/launch_dense_single.py", "trainer/train_sft_vlm.py",
                "trainer/trainer_utils.py", "model/model_vlm.py",
                "model/model_minimind.py", "dataset/lm_dataset.py")
RECIPE = {"epochs": 2, "effective_batch": 4, "max_seq_len": 768,
          "learning_rate": 5e-6, "hidden_size": 768, "num_hidden_layers": 8,
          "freeze_llm": 1, "dtype": "bfloat16", "seed": 42,
          "num_workers": 0, "log_interval": 100, "save_interval": 1000,
          "probe_optimizer_updates": 2}
METRIC_RE = re.compile(
    r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), loss: ([^,]+), "
    r"logits_loss: ([^,]+), aux_loss: ([^,]+), lr: ([^,]+), epoch_time: ([^m]+)min")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_json(path, data):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def verify_attestation(acceptance_path, weights_path):
    receipt = read_json(acceptance_path)
    if receipt.get("status") != "accepted_text_screening":
        raise ValueError("Deployment requires accepted_text_screening product acceptance")
    expected = receipt.get("selected", {}).get("sha256", "")
    if not re.fullmatch(r"[a-fA-F0-9]{64}", expected):
        raise ValueError("Acceptance has no valid selected.sha256")
    observed = sha256(weights_path)
    if observed != expected.lower():
        raise ValueError("Initialization weights differ from accepted selected checkpoint")
    return {"receipt": receipt, "receipt_sha256": sha256(acceptance_path),
            "weights_sha256": observed}


def gpu_guard(gpu_uuid, command=subprocess.run):
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("Use the full GPU UUID, not a physical index")
    uuid.UUID(gpu_uuid[4:])

    def query(fields, kind):
        result = command(["nvidia-smi", "--id", gpu_uuid,
                          f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(f"Selected GPU query failed: {result.stderr.strip()}")
        return [list(map(str.strip, row)) for row in csv.reader(io.StringIO(result.stdout))
                if row and any(item.strip() for item in row)]

    devices = query("index,uuid,name,memory.used,memory.total", "gpu")
    if len(devices) != 1 or len(devices[0]) != 5 or devices[0][1] != gpu_uuid:
        raise RuntimeError("nvidia-smi did not resolve exactly the selected GPU UUID")
    busy = [row for row in query("gpu_uuid,pid,process_name,used_gpu_memory", "compute-apps")
            if row[0] == gpu_uuid]
    if busy:
        raise RuntimeError(f"Selected GPU has an active compute process; nothing started: {busy}")
    row = devices[0]
    return {"physical_index": int(row[0]), "uuid": row[1], "name": row[2],
            "used_mib": float(row[3]), "total_mib": float(row[4]), "checked_at": now()}


@contextmanager
def run_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                if path.stat().st_size == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"A launcher already owns this run/GPU lock: {path}") from error
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def build_contract(args):
    attestation = verify_attestation(args.acceptance, args.init_weights)
    if not re.fullmatch(r"[a-fA-F0-9]{64}", args.data_sha256):
        raise ValueError("--data-sha256 must be the verified official train parquet hash")
    print("Verifying complete official train parquet hash...", flush=True)
    if sha256(args.data) != args.data_sha256.lower():
        raise ValueError("Official train parquet SHA256 mismatch")
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(args.data)
    if parquet.metadata.num_rows != OFFICIAL_ROWS:
        raise ValueError(f"Expected all {OFFICIAL_ROWS} official train rows")
    if not {"image_bytes", "conversations"}.issubset(parquet.schema_arrow.names):
        raise ValueError("Official training parquet lacks native columns")
    assets = {}
    for directory, expected_files in ((VISION / "model", TOKENIZER_FILES),
                                      (VISION / "model/siglip2-base-p32-256-ve", ENCODER_FILES)):
        for name, expected in expected_files.items():
            path = directory / name
            if sha256(path) != expected:
                raise ValueError(f"Pinned offline model/tokenizer asset changed: {path}")
            assets[str(path.relative_to(VISION))] = expected
    return {"schema_version": 1, "owner": "evomind_dense_single_native_v1",
            "run_dir": str(args.run_dir), "gpu_uuid": args.gpu_uuid,
            "acceptance_sha256": attestation["receipt_sha256"],
            "acceptance": attestation["receipt"],
            "init_weights": {"path": str(args.init_weights), "sha256": attestation["weights_sha256"]},
            "data": {"path": str(args.data), "sha256": args.data_sha256.lower(),
                     "bytes": args.data.stat().st_size, "rows": OFFICIAL_ROWS,
                     "row_groups": parquet.metadata.num_row_groups},
            "assets_sha256": assets,
            "source_sha256": {name: sha256(VISION / name) for name in SOURCE_FILES},
            "recipe": RECIPE,
            "resume_limitations": ["Native augmentation RNG is not checkpointed; not bitwise replay.",
                                   "Unsaved updates are recomputed from the last saved optimizer boundary.",
                                   "No automatic fresh restart after formal training began without a checkpoint."],
            "evaluation_scope": "Training completion is not six-image evaluation or a quality improvement claim."}


def choose_probe_plan(reports):
    for batch, accumulation in ((4, 1), (1, 4)):
        matches = [r for r in reports if (r.get("batch_size"), r.get("accumulation_steps")) == (batch, accumulation)]
        if not matches:
            return batch, accumulation
        report = matches[-1]
        status = report.get("status")
        if status == "success":
            if not isinstance(report.get("iterations"), list) or len(report["iterations"]) != 2:
                raise ValueError("A successful probe must contain exactly two optimizer updates")
            return None
        if status in ("checking", "running", "interrupted"):
            return batch, accumulation
        if status != "oom":
            raise RuntimeError("A non-OOM probe failure must be diagnosed; no fallback allowed")
    raise RuntimeError("Both predefined probe configurations ran out of memory")


def parse_metric(line, accumulation, invocation):
    match = METRIC_RE.search(line)
    if not match:
        return None
    epoch, epochs, step, steps = map(int, match.groups()[:4])
    loss, logits, aux, lr, eta = map(float, match.groups()[4:])
    if not all(math.isfinite(x) for x in (loss, logits, aux, lr, eta)):
        raise ValueError("Native trainer logged nonfinite metrics")
    updates = math.ceil(steps / accumulation)
    completed = math.ceil(step / accumulation) if step == steps else step // accumulation
    return {"event": "native_train_metric", "at": now(), "invocation": invocation,
            "epoch": epoch, "epochs": epochs, "microstep": step, "microsteps_per_epoch": steps,
            "optimizer_update": (epoch - 1) * updates + completed, "loss": loss,
            "logits_loss": logits, "aux_loss": aux, "learning_rate": lr,
            "epoch_eta_minutes": eta, "definition": "Native logged last-microbatch loss, not validation loss"}


def write_curves(path, records):
    if not records:
        return
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="650" viewBox="0 0 1000 650">',
             '<rect width="1000" height="650" fill="white"/>',
             '<g font-family="sans-serif" font-size="14" fill="#222">',
             '<text x="60" y="30">Native Dense single-image training</text>']
    maximum = max(1, max(r["optimizer_update"] for r in records))
    for field, title, top in (("loss", "Logged microbatch loss", 60), ("learning_rate", "Learning rate", 350)):
        values = [r[field] for r in records]
        low, high = min(values), max(values)
        span = max(high - low, abs(high) * .01, 1e-12)
        parts += [f'<text x="60" y="{top}">{title}</text>',
                  f'<path d="M60 {top + 15} V{top + 220} H950" fill="none" stroke="#888"/>',
                  f'<text x="65" y="{top + 35}">{high:.6g}</text>',
                  f'<text x="65" y="{top + 210}">{low:.6g}</text>']
        by_invocation = {}
        for record in records:
            by_invocation.setdefault(record["invocation"], []).append(record)
        for series in by_invocation.values():
            sampled = series[::max(1, len(series) // 1200)]
            if sampled[-1] is not series[-1]:
                sampled.append(series[-1])
            points = " ".join(f'{60 + 890 * r["optimizer_update"] / maximum:.1f},{top + 220 - 200 * (r[field] - low) / span:.1f}' for r in sampled)
            parts.append(f'<polyline points="{points}" fill="none" stroke="#2563a6" stroke-width="1.5"/>')
        parts.append(f'<text x="700" y="{top + 245}">Optimizer updates; latest {maximum:,}</text>')
    parts.append('<text x="60" y="625">All invocation histories retained; unsaved updates may be recomputed after resume.</text></g></svg>')
    temporary = Path(str(path) + ".tmp")
    temporary.write_text("\n".join(parts), encoding="utf-8")
    os.replace(temporary, path)


def native_argv(run_dir, selection, resume=False):
    run_dir = Path(run_dir)
    contract = read_json(run_dir / "contract.json")
    result = [str(VISION / "trainer/train_sft_vlm.py"), "--from_weight", "accepted_text",
              "--save_dir", str(run_dir / "exports"), "--save_weight", "dense_single",
              "--epochs", "2", "--batch_size", str(selection["batch_size"]),
              "--accumulation_steps", str(selection["accumulation_steps"]),
              "--learning_rate", "5e-6", "--hidden_size", "768", "--num_hidden_layers", "8",
              "--max_seq_len", "768", "--freeze_llm", "1", "--dtype", "bfloat16",
              "--device", "cuda:0", "--num_workers", "0", "--log_interval", "100",
              "--save_interval", "1000", "--data_path", contract["data"]["path"]]
    if resume:
        result += ["--from_resume", "1"]
    return result


def worker_environment(contract, run_dir):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=contract["gpu_uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       PYTHONPATH=str(VISION), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                       HF_HOME=str(VISION.parent / ".cache/huggingface"),
                       HF_DATASETS_CACHE=str(VISION.parent / ".cache/huggingface/datasets"),
                       EVOMIND_DENSE_PARENT_PID=str(os.getpid()))
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    return environment


def run_worker(argv, log_path, environment, state, run_dir, records, accumulation=1):
    process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), *argv],
                               cwd=VISION / "trainer", env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               start_new_session=(os.name != "nt"))
    state.update(child_pid=process.pid, child_argv=argv, updated_at=now())
    atomic_json(run_dir / "state.json", state)
    try:
        with Path(log_path).open("x", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                if "--_worker" in argv and argv[argv.index("--_worker") + 1] == "train":
                    metric = parse_metric(line, accumulation, state["invocation"])
                    if metric:
                        append_json(run_dir / "metrics.jsonl", metric)
                        records.append(metric)
                        state.update(last_metric=metric, updated_at=now())
                        atomic_json(run_dir / "state.json", state)
                        if len(records) == 1 or len(records) % 10 == 0:
                            write_curves(run_dir / "curves.svg", records)
        return process.wait()
    finally:
        if process.poll() is None:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        state.update(child_pid=None, updated_at=now())
        atomic_json(run_dir / "state.json", state)
        write_curves(run_dir / "curves.svg", records)


def protect_worker_parent():
    expected = int(os.environ.get("EVOMIND_DENSE_PARENT_PID", "0"))
    if not expected:
        raise RuntimeError("Internal workers must be started by the owned launcher")
    if sys.platform == "linux":
        import ctypes
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0:
            raise RuntimeError("Could not enable worker parent-death protection")
    if os.getppid() != expected:
        raise RuntimeError("Launcher disappeared before worker startup")


def cuda_uuid_matches(observed_uuid, assigned_uuid):
    """Torch may omit the GPU- prefix that nvidia-smi includes."""
    return (uuid.UUID(str(observed_uuid).removeprefix("GPU-")) ==
            uuid.UUID(str(assigned_uuid).removeprefix("GPU-")))


def load_native_runtime(contract):
    import datasets  # Windows Arrow DLL ordering; harmless on Linux.
    import torch
    from model.model_vlm import VLMConfig
    from trainer.trainer_utils import init_vlm_model, setup_seed
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly the UUID-bound CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native BF16 is unavailable on the selected GPU")
    observed_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    if observed_uuid is not None and not cuda_uuid_matches(observed_uuid, contract["gpu_uuid"]):
        raise RuntimeError("Torch device UUID disagrees with the assigned GPU")
    setup_seed(42)
    config = VLMConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=768, use_moe=False)
    model, tokenizer, processor = init_vlm_model(
        config, from_weight="none", tokenizer_path=str(VISION / "model"),
        vision_model_path=str(VISION / "model/siglip2-base-p32-256-ve"), device="cuda:0", freeze_llm=1)
    if model.vision_encoder is None or processor is None:
        raise RuntimeError("Pinned local SigLIP encoder failed to load")
    if len(tokenizer) != 6400 or tokenizer.encode("<|image_pad|>", add_special_tokens=False) != [12]:
        raise ValueError("Tokenizer does not match new64M vocabulary/image marker")
    weights = torch.load(contract["init_weights"]["path"], map_location="cpu", weights_only=True)
    target = model.state_dict()
    keys = {key for key in target if not key.startswith(("vision_encoder.", "vision_proj."))}
    if not isinstance(weights, dict) or set(weights) != keys:
        raise ValueError("Accepted text tensor keys do not exactly match the native new64M LLM")
    if any(not isinstance(weights[key], torch.Tensor) or target[key].shape != weights[key].shape for key in keys):
        raise ValueError("Accepted text checkpoint has incompatible tensor shapes")
    model.load_state_dict(weights, strict=False)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith(("vision_proj.", "model.layers.0.", "model.layers.7.")) for name in trainable):
        raise ValueError("Native freeze policy changed")
    return model, tokenizer, processor, config


def initialize_probe_cuda(torch):
    """Initialize the UUID-bound CUDA allocator before resetting its counters."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly the UUID-bound CUDA device")
    torch.cuda.set_device(0)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)


def probe_worker(args, contract):
    report_path = Path(args.report)
    report = {"status": "checking", "started_at": now(), "batch_size": args.batch,
              "accumulation_steps": args.accumulation, "iterations": [], "probe_only": True,
              "contract_sha256": sha256(args.run_dir / "contract.json"),
              "note": "Two repeated real-data batch updates; probe weights are discarded."}
    atomic_json(report_path, report)
    torch = None
    try:
        gpu_guard(contract["gpu_uuid"])
        import datasets
        import torch as torch_module
        torch = torch_module
        import pyarrow.parquet as pq
        from dataset.lm_dataset import VLMDataset
        from trainer.trainer_utils import get_lr, vlm_collate_fn
        initialize_probe_cuda(torch)
        model, tokenizer, processor, config = load_native_runtime(contract)
        rows = next(pq.ParquetFile(contract["data"]["path"]).iter_batches(
            batch_size=args.batch, columns=["image_bytes", "conversations"])).to_pylist()
        if len(rows) != args.batch:
            raise ValueError("Insufficient real rows for the prescribed probe batch")
        dataset = VLMDataset.__new__(VLMDataset)
        dataset.dataset, dataset.tokenizer, dataset.preprocess = rows, tokenizer, processor
        dataset.max_length = 768
        dataset.image_special_token = config.image_special_token * 64
        dataset.bos_id = tokenizer(tokenizer.bos_token + "assistant\n", add_special_tokens=False).input_ids
        dataset.eos_id = tokenizer(tokenizer.eos_token + "\n", add_special_tokens=False).input_ids
        batch = vlm_collate_fn([dataset[index] for index in range(args.batch)])
        if not torch.all((batch[0] == 12).sum(1) == 64) or not torch.all((batch[1][:, 1:] != -100).sum(1) > 0):
            raise ValueError("Real probe rows have truncated image markers or no assistant labels")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        report.update(status="running", torch=str(torch.__version__), cuda=torch.version.cuda,
                      gpu=contract["gpu_uuid"], input_shapes={name: list(value.shape) for name, value in batch[2].items()},
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        atomic_json(report_path, report)
        for update in range(2):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for microstep in range(args.accumulation):
                step = update * args.accumulation + microstep + 1
                for group in optimizer.param_groups:
                    group["lr"] = get_lr(step, 2 * math.ceil(OFFICIAL_ROWS / args.batch), 5e-6)
                ids, labels = (value.to("cuda:0") for value in batch[:2])
                pixels = {name: value.to("cuda:0") for name, value in batch[2].items()}
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output = model(ids, labels=labels, pixel_values=pixels)
                    native_loss = output.loss + output.aux_loss
                    loss = native_loss / args.accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite real-batch native probe loss")
                scaler.scale(loss).backward()
                last_loss = float(native_loss.detach())
                del ids, labels, pixels, output, loss, native_loss
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite native probe gradient")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            observation = {"optimizer_update": update + 1, "last_microbatch_loss": last_loss,
                           "seconds": time.perf_counter() - started, "gradient_norm": float(norm)}
            report["iterations"].append(observation)
            report.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
            atomic_json(report_path, report)
            print("DENSE_PROBE " + json.dumps(observation), flush=True)
        report.update(status="success", completed_at=now())
        return_code = 0
    except Exception as error:
        oom = torch is not None and isinstance(error, torch.cuda.OutOfMemoryError)
        report.update(status="oom" if oom else "failed", error=str(error), traceback=traceback.format_exc(), completed_at=now())
        return_code = 3 if oom else 1
    atomic_json(report_path, report)
    print("DENSE_PROBE_RESULT " + json.dumps(report), flush=True)
    return return_code


def preserve_file(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Immutable artifact already exists: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256(destination)}


def train_worker(args, contract):
    gpu_guard(contract["gpu_uuid"])
    import datasets
    import torch
    import trainer.trainer_utils as native
    selection = read_json(args.run_dir / "preflight.json")["selected"]
    model, tokenizer, processor, config = load_native_runtime(contract)
    original_checkpoint = native.vlm_checkpoint
    checkpoint_dir = args.run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    (args.run_dir / "exports").mkdir(exist_ok=True)
    (args.run_dir / "snapshots").mkdir(exist_ok=True)

    def initialize(vlm_config, **kwargs):
        if (vlm_config.hidden_size, vlm_config.num_hidden_layers, kwargs.get("freeze_llm")) != (768, 8, 1):
            raise ValueError("Native trainer requested a changed model/freeze recipe")
        return model, tokenizer, processor

    def checkpoint(vlm_config, *positional, **kwargs):
        kwargs["save_dir"] = str(checkpoint_dir)
        result = original_checkpoint(vlm_config, *positional, **kwargs)
        if kwargs.get("model") is not None:
            epoch, step = kwargs["epoch"], kwargs["step"]
            directory = args.run_dir / "snapshots" / f"epoch{epoch + 1}_step{step}"
            directory.mkdir(exist_ok=False)
            record = {"epoch": epoch, "step": step, "saved_at": now(),
                      "weights": preserve_file(checkpoint_dir / "dense_single_768.pth", directory / "weights.pth"),
                      "resume": preserve_file(checkpoint_dir / "dense_single_768_resume.pth", directory / "resume.pth")}
            atomic_json(directory / "receipt.json", record)
            append_json(args.run_dir / "checkpoints.jsonl", record)
        elif result is not None:
            if result.get("world_size") != 1 or result.get("epoch") not in (0, 1):
                raise ValueError("Native resume world size or epoch is incompatible")
            steps = math.ceil(OFFICIAL_ROWS / selection["batch_size"])
            step = result.get("step", -1)
            if not isinstance(step, int) or not 0 < step <= steps:
                raise ValueError("Native resume microstep is out of bounds")
            if step != steps and step % selection["accumulation_steps"]:
                raise ValueError("Native resume is not at an optimizer boundary")
            if set(result["model"]) != set(model.state_dict()):
                raise ValueError("Native resume tensor keys changed")
        return result

    native.init_vlm_model = initialize
    native.vlm_checkpoint = checkpoint
    sys.argv = native_argv(args.run_dir, selection, resume=args.resume)
    print("NATIVE_TRAIN_ARGV " + json.dumps(sys.argv), flush=True)
    runpy.run_path(str(VISION / "trainer/train_sft_vlm.py"), run_name="__main__")
    resume_path = checkpoint_dir / "dense_single_768_resume.pth"
    payload = torch.load(resume_path, map_location="cpu", weights_only=True)
    if (payload["epoch"], payload["step"]) != (1, math.ceil(OFFICIAL_ROWS / selection["batch_size"])):
        raise RuntimeError("Native trainer returned without the complete two-epoch checkpoint")
    del payload
    atomic_json(args.run_dir / "training_completed.json", {
        "status": "training_complete_evaluation_pending", "completed_at": now(),
        "contract_sha256": sha256(args.run_dir / "contract.json"), "epochs": 2,
        "optimizer_updates": 2 * math.ceil(OFFICIAL_ROWS / 4),
        "weights": {"path": str(args.run_dir / "exports/dense_single_768.pth"),
                    "sha256": sha256(args.run_dir / "exports/dense_single_768.pth")},
        "resume_sha256": sha256(resume_path)})
    return 0


def launch(args):
    if not args.gpu_uuid.startswith("GPU-"):
        raise ValueError("Use the full GPU UUID, not a physical index")
    uuid.UUID(args.gpu_uuid[4:])
    if args.run_dir in (VISION, VISION.parent) or not args.run_dir.is_relative_to(VISION.parent):
        raise ValueError("Use a dedicated run directory inside the evomind project")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(args.run_dir / "run.lock"), run_lock(VISION / "artifacts/gpu_leases" / (args.gpu_uuid + ".lock")):
        contract_path = args.run_dir / "contract.json"
        existing = contract_path.exists()
        if existing and not args.resume:
            raise RuntimeError("Run already exists; use --resume after inspecting its state")
        if not existing and any(path.name != "run.lock" for path in args.run_dir.iterdir()):
            raise RuntimeError("Refusing an unowned nonempty run directory")
        hardware = gpu_guard(args.gpu_uuid)
        contract = build_contract(args)
        if existing and read_json(contract_path) != contract:
            raise ValueError("Inputs, source code or recipe changed; cannot resume this run")
        if not existing:
            atomic_json(contract_path, contract)
            atomic_json(args.run_dir / "product_acceptance.json", contract["acceptance"])
        state_path = args.run_dir / "state.json"
        state = read_json(state_path) if state_path.exists() else {"created_at": now(), "formal_started": False}
        if state.get("status") == "training_complete_evaluation_pending":
            completion = read_json(args.run_dir / "training_completed.json")
            if sha256(completion["weights"]["path"]) != completion["weights"]["sha256"]:
                raise ValueError("Completed model export changed")
            print("Training already complete; evaluation remains a separate gate.", flush=True)
            return 0
        invocation = str(time.time_ns())
        state.update(status="preflight", invocation=invocation, pid=os.getpid(), hardware=hardware, updated_at=now())
        atomic_json(state_path, state)
        for name in ("logs", "probes"):
            (args.run_dir / name).mkdir(exist_ok=True)
        environment = worker_environment(contract, args.run_dir)
        metrics_path = args.run_dir / "metrics.jsonl"
        records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()] if metrics_path.exists() else []
        try:
            preflight_path = args.run_dir / "preflight.json"
            while not preflight_path.exists():
                report_files = sorted((args.run_dir / "probes").glob("*.json"))
                reports = [read_json(path) for path in report_files]
                plan = choose_probe_plan(reports)
                if plan is None:
                    successes = [(path, report) for path, report in zip(report_files, reports) if report.get("status") == "success"]
                    path, report = successes[-1]
                    atomic_json(preflight_path, {"contract_sha256": sha256(contract_path),
                        "selected": {"batch_size": report["batch_size"], "accumulation_steps": report["accumulation_steps"]},
                        "probe": {"path": str(path), "sha256": sha256(path)},
                        "fallback_policy": "B1/acc4 only after observed CUDA OOM at B4/acc1"})
                    break
                batch, accumulation = plan
                gpu_guard(args.gpu_uuid)
                report = args.run_dir / "probes" / f"{time.time_ns()}_b{batch}a{accumulation}.json"
                code = run_worker(["--_worker", "probe", "--run-dir", str(args.run_dir), "--batch", str(batch),
                                   "--accumulation", str(accumulation), "--report", str(report)],
                                  args.run_dir / "logs" / (report.stem + ".log"), environment, state, args.run_dir, records)
                if not report.is_file() or code not in (0, 3):
                    raise RuntimeError(f"Probe failed with exit {code}; inspect preserved report/log")
            preflight = read_json(preflight_path)
            if preflight["contract_sha256"] != sha256(contract_path) or sha256(preflight["probe"]["path"]) != preflight["probe"]["sha256"]:
                raise ValueError("Successful preflight no longer matches this run")
            selected = preflight["selected"]
            if (selected["batch_size"], selected["accumulation_steps"]) not in ((4, 1), (1, 4)):
                raise ValueError("Preflight selected an unapproved recipe")
            success = read_json(preflight["probe"]["path"])
            if (success.get("status") != "success" or len(success.get("iterations", [])) != 2
                    or success.get("contract_sha256") != sha256(contract_path)
                    or (success.get("batch_size"), success.get("accumulation_steps")) !=
                    (selected["batch_size"], selected["accumulation_steps"])):
                raise ValueError("Preflight does not contain the selected two-update success evidence")
            checkpoint = args.run_dir / "checkpoints/dense_single_768_resume.pth"
            resume_training = bool(state.get("formal_started"))
            if resume_training and not checkpoint.is_file():
                raise RuntimeError("Formal training started but has no saved checkpoint; preserve this run and inspect it")
            if not resume_training and checkpoint.exists():
                raise RuntimeError("Unexpected checkpoint before formal training ownership")
            gpu_guard(args.gpu_uuid)
            state.update(status="formal_training", formal_started=True, selected=selected, updated_at=now())
            atomic_json(state_path, state)
            worker_args = ["--_worker", "train", "--run-dir", str(args.run_dir)]
            if resume_training:
                worker_args.append("--resume")
            code = run_worker(worker_args, args.run_dir / "logs" / f"{invocation}_formal.log", environment,
                              state, args.run_dir, records, selected["accumulation_steps"])
            if code or not (args.run_dir / "training_completed.json").is_file():
                raise RuntimeError(f"Native trainer exited {code}; no completion claim")
            state.update(status="training_complete_evaluation_pending", completed_at=now())
            atomic_json(state_path, state)
            return 0
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         error=str(error), updated_at=now())
            atomic_json(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--data-sha256")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--_worker", choices=("probe", "train"), help=argparse.SUPPRESS)
    parser.add_argument("--batch", type=int, choices=(1, 4), help=argparse.SUPPRESS)
    parser.add_argument("--accumulation", type=int, choices=(1, 4), help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args._worker:
        protect_worker_parent()
        contract = read_json(args.run_dir / "contract.json")
        verify_attestation(args.run_dir / "product_acceptance.json", contract["init_weights"]["path"])
        return probe_worker(args, contract) if args._worker == "probe" else train_worker(args, contract)
    for name in ("acceptance", "init_weights", "data", "data_sha256", "gpu_uuid"):
        if getattr(args, name) is None:
            parser.error("--" + name.replace("_", "-") + " is required")
    for name in ("acceptance", "init_weights", "data"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}; owned child will be stopped")
    signal.signal(signal.SIGTERM, interrupted)
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
