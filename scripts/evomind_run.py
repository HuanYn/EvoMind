"""Run an auditable, fail-closed sequence of local Python stages.

Usage: python scripts/evomind_run.py --manifest manifests/run.json [--resume]
The manifest contains run_dir and stages [{name, argv, cwd?, expected_weight?}].
argv excludes the interpreter; relative cwd and output paths are repo-relative.
Without cwd, stages run in trainer/. No GPU libraries are imported here.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


REPO_ROOT = Path(__file__).resolve().parents[1]
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)"
LOG_PATTERN = re.compile(
    r"Epoch:\s*\[(?P<epoch>\d+)\s*/\s*(?P<epochs>\d+)\]\s*"
    r"[\[(](?P<microstep>\d+)\s*/\s*(?P<microsteps_per_epoch>\d+)[\])]"
    rf",\s*loss:\s*(?P<loss>{NUMBER}),\s*logits_loss:\s*(?P<logits_loss>{NUMBER}),"
    rf"\s*aux_loss:\s*(?P<aux_loss>{NUMBER}),\s*lr:\s*(?P<learning_rate>{NUMBER})",
    re.IGNORECASE,
)
OFFICIAL_TRAINERS = {"train_pretrain.py", "train_full_sft.py"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def parse_official_log(line):
    """Parse observed training losses, never infer validation or optimizer steps."""
    match = LOG_PATTERN.search(line)
    if not match:
        return None
    result = {}
    for key, value in match.groupdict().items():
        if key in {"epoch", "epochs", "microstep", "microsteps_per_epoch"}:
            result[key] = int(value)
        else:
            number = float(value)
            result[key] = number if math.isfinite(number) else None
    result["nonfinite"] = any(result[key] is None for key in
                              ("loss", "logits_loss", "aux_loss", "learning_rate"))
    result["global_microstep"] = ((result["epoch"] - 1) * result["microsteps_per_epoch"]
                                  + result["microstep"])
    result["metric_scope"] = "training_current_microbatch"
    return result


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied: conservatively live.
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class WindowsJobObject:
    """Own only newly created children, never the supervisor or existing training.

    The noninheritable handle has KILL_ON_JOB_CLOSE, including on abrupt owner
    exit. Children start suspended so a venv redirector cannot spawn outside the
    job before assignment. Nested jobs are supported on Windows 8 and newer.
    https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
    """
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("WindowsJobObject is available only on Windows.")
        import ctypes
        from ctypes import wintypes
        self.ctypes, self.wintypes = ctypes, wintypes
        kernel = self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                        ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_longlong) for name in (
                "TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")]
            _fields_ += [(name, wintypes.DWORD) for name in (
                "TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]

        self.Accounting = Accounting
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateJobObject.restype = wintypes.BOOL
        kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                     wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        kernel.QueryInformationJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
        if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel.CloseHandle(self.handle)
            self.handle = None
            raise error

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def resume_suspended_process(self, process):
        """Popen closes its initial-thread handle; reopen that owned thread."""
        ctypes, wintypes, kernel = self.ctypes, self.wintypes, self.kernel

        class ThreadEntry(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
                        ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG), ("dwFlags", wintypes.DWORD)]

        kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        for name in ("Thread32First", "Thread32Next"):
            function = getattr(kernel, name)
            function.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
            function.restype = wintypes.BOOL
        kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenThread.restype = wintypes.HANDLE
        kernel.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel.ResumeThread.restype = wintypes.DWORD
        snapshot = kernel.CreateToolhelp32Snapshot(0x4, 0)  # TH32CS_SNAPTHREAD.
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = ThreadEntry()
            entry.dwSize = ctypes.sizeof(entry)
            found = kernel.Thread32First(snapshot, ctypes.byref(entry))
            while found:
                if entry.th32OwnerProcessID == process.pid:
                    thread = kernel.OpenThread(0x0002, False, entry.th32ThreadID)  # THREAD_SUSPEND_RESUME.
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        previous = kernel.ResumeThread(thread)
                        if previous == 0xFFFFFFFF:
                            raise ctypes.WinError(ctypes.get_last_error())
                        if previous != 1:
                            raise RuntimeError(f"Unexpected initial-thread suspend count: {previous}")
                    finally:
                        kernel.CloseHandle(thread)
                    return
                found = kernel.Thread32Next(snapshot, ctypes.byref(entry))
            raise RuntimeError(f"Suspended child {process.pid} has no discoverable initial thread.")
        finally:
            kernel.CloseHandle(snapshot)

    def active_processes(self):
        data = self.Accounting()
        if not self.kernel.QueryInformationJobObject(self.handle, 1, self.ctypes.byref(data),
                                                    self.ctypes.sizeof(data), None):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        return data.ActiveProcesses

    def close(self, timeout=15):
        """Terminate the owned tree and confirm its exit before releasing the job."""
        if self.handle is None:
            return
        try:
            if not self.kernel.TerminateJobObject(self.handle, 1):
                raise self.ctypes.WinError(self.ctypes.get_last_error())
            deadline = time.monotonic() + timeout
            while self.active_processes():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Owned Windows process tree did not exit within the cleanup deadline.")
                time.sleep(0.01)
        finally:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


@contextlib.contextmanager
def managed_child(argv, **popen_kwargs):
    """Yield Popen; close its owned Windows process tree on every exit path.

    Call child.wait()/communicate() inside the context to preserve its actual
    exit code. The Windows job is not inherited, so owner death closes its last
    handle and kills descendants. Linux retains direct-child cleanup semantics.
    """
    job = WindowsJobObject() if os.name == "nt" else None
    child = None
    if job is not None:
        flags = popen_kwargs.get("creationflags", 0)
        if flags & 0x01000000:  # CREATE_BREAKAWAY_FROM_JOB defeats ownership.
            job.close()
            raise ValueError("managed_child forbids CREATE_BREAKAWAY_FROM_JOB")
        popen_kwargs["creationflags"] = flags | 0x00000004  # CREATE_SUSPENDED.
    try:
        child = subprocess.Popen(argv, **popen_kwargs)
        if job is not None:
            job.assign(child)
            job.resume_suspended_process(child)
        yield child
    finally:
        try:
            if job is not None:
                job.close()
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=15)
            if child is not None:
                for stream in (child.stdin, child.stdout, child.stderr):
                    if stream is not None:
                        stream.close()


@contextlib.contextmanager
def exclusive_run_lock(run_dir):
    """OS-owned lock automatically releases if the supervisor crashes."""
    with (run_dir / "run.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("This run already has an active supervisor.") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def resolve_path(value, repo):
    path = Path(value)
    return (path if path.is_absolute() else repo / path).resolve()


def weight_path(value, repo):
    if not value:
        return None
    path = Path(value)
    if len(path.parts) == 1:
        name = path.name if path.suffix == ".pth" else path.name + "_768.pth"
        return (repo / "out" / name).resolve()
    return resolve_path(value, repo)


def flag_value(argv, flag, default):
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return default


def set_flag(argv, flag, value):
    result = list(argv)
    for index, item in enumerate(result):
        if item == flag:
            result[index + 1] = value
            return result
        if item.startswith(flag + "="):
            result[index] = flag + "=" + value
            return result
    return result + [flag, value]


def continuation_resume_argv(argv, repo, resume_requested):
    """Forward explicit outer resume only when the continuation has inner state."""
    if not resume_requested or Path(argv[0]).name != "evomind_continue.py" or "--resume" in argv:
        return argv
    nested_manifest = repo / "configs" / "vision_pipeline.json"
    if not nested_manifest.is_file():
        return argv
    manifest = json.loads(nested_manifest.read_text(encoding="utf-8-sig"))
    if (resolve_path(manifest["run_dir"], repo) / "state.json").is_file():
        return [*argv, "--resume"]
    return argv


def checkpoint_path(stage, cwd):
    script = Path(stage["argv"][0]).name
    if script not in OFFICIAL_TRAINERS:
        return None
    prefix = flag_value(stage["argv"], "--save_weight",
                        "pretrain" if script == "train_pretrain.py" else "full_sft")
    hidden = flag_value(stage["argv"], "--hidden_size", "768")
    suffix = "_moe" if flag_value(stage["argv"], "--use_moe", "0") == "1" else ""
    return (cwd / "../checkpoints" / f"{prefix}_{hidden}{suffix}_resume.pth").resolve()


def file_record(path):
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def provenance(repo, stages):
    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=15)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    files = {Path(__file__).resolve()}
    tracked = git("ls-files", "*.py", "requirements.txt", "model/*.json") or ""
    files.update(repo / name for name in tracked.splitlines())
    for stage in stages:
        cwd = resolve_path(stage.get("cwd", "trainer"), repo)
        script = Path(stage["argv"][0])
        files.add(script if script.is_absolute() else cwd / script)
    packages = {}
    for name in ("torch", "transformers", "datasets", "numpy", "pyarrow", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    safe_environment = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM", "PYTHONIOENCODING",
                        "HF_HOME", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "VIRTUAL_ENV")
    return {"captured_at": now(), "code_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"), "python": sys.version,
            "executable": sys.executable, "platform": platform.platform(), "packages": packages,
            "environment": {key: os.environ[key] for key in safe_environment if key in os.environ},
            "environment_policy": "Child inherits the complete parent environment; only safe keys are recorded.",
            "file_hashes": {str(path.resolve()): sha256(path) for path in sorted(files) if path.is_file()}}


def render_curve(metrics_path, destination):
    """Draw only observed training metrics. A missing plotting dependency is recorded."""
    if not metrics_path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        return
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for key in ("loss", "logits_loss", "aux_loss"):
        axes[0].plot([row["global_microstep"] for row in rows],
                     [row[key] for row in rows], label=key, linewidth=1)
    axes[0].set_ylabel("Observed training loss")
    axes[0].legend()
    axes[1].plot([row["global_microstep"] for row in rows],
                 [row["learning_rate"] for row in rows], linewidth=1)
    axes[1].set_ylabel("Learning rate")
    axes[1].set_xlabel("Global microstep (not optimizer step)")
    figure.suptitle("Training current-microbatch metrics; no validation estimates")
    figure.tight_layout()
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        figure.savefig(temporary, format="png", dpi=140)
        os.replace(temporary, destination)
    finally:
        plt.close(figure)


def validate_manifest(manifest, repo):
    if not isinstance(manifest.get("run_dir"), str) or not manifest["run_dir"]:
        raise ValueError("Manifest must specify a nonempty run_dir.")
    stages = manifest.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Manifest must contain at least one stage.")
    names = set()
    for stage in stages:
        name = stage.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError("Stage names must be unique, filesystem-safe identifiers.")
        names.add(name)
        argv = stage.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
            raise ValueError(f"Stage {name}: argv must be a nonempty list of strings.")
        if not resolve_path(stage.get("cwd", "trainer"), repo).is_dir():
            raise ValueError(f"Stage {name}: cwd does not exist.")


def run_manifest(manifest_path, resume=False, repo_root=REPO_ROOT):
    repo = Path(repo_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    validate_manifest(manifest, repo)
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                                             separators=(",", ":")).encode("utf-8")).hexdigest()
    run_dir = resolve_path(manifest["run_dir"], repo)
    run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(run_dir):
        return _run_locked(manifest, manifest_path, manifest_hash, run_dir, repo, resume)


def _run_locked(manifest, manifest_path, manifest_hash, run_dir, repo, resume):
    state_path = run_dir / "state.json"
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    if previous:
        if not resume:
            raise RuntimeError("Run state already exists; use --resume explicitly or a new run_dir.")
        if previous.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("Refusing resume: manifest hash differs (including batch/accumulation).")
        for key in ("pid", "child_pid"):
            if previous.get("status") == "running" and pid_alive(previous.get(key)):
                raise RuntimeError(f"Refusing resume: recorded {key} is still running.")
        if previous.get("status") == "completed":
            return 0
    elif resume:
        raise RuntimeError("Cannot resume a run without state.json.")
    elif any(path.name != "run.lock" for path in run_dir.iterdir()):
        raise RuntimeError("Run directory is nonempty; refusing to overwrite untracked artifacts.")

    for name in ("logs", "metrics", "curves", "snapshots"):
        (run_dir / name).mkdir(exist_ok=True)
    atomic_json(run_dir / "manifest.json", manifest)
    environment = provenance(repo, manifest["stages"])
    append_json(run_dir / "provenance.jsonl", environment)
    state = previous or {"schema_version": 1, "run_dir": str(run_dir), "started_at": now(),
                         "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash,
                         "stages": {}}
    state.update(status="running", pid=os.getpid(), child_pid=None, current_stage=None,
                 last_started_at=now(), exit_code=None, error=None)
    child = None
    current = None

    def persist():
        state["updated_at"] = now()
        atomic_json(state_path, state)

    def plot(stage_name):
        try:
            render_curve(run_dir / "metrics" / f"{stage_name}.jsonl",
                         run_dir / "curves" / f"{stage_name}.png")
        except Exception as error:
            append_json(run_dir / "events.jsonl", {"timestamp": now(), "stage": stage_name,
                        "event": "curve_render_failed", "error": str(error)})

    persist()
    try:
        for stage in manifest["stages"]:
            name = stage["name"]
            current = state["stages"].get(name)
            expected = weight_path(stage.get("expected_weight"), repo)
            if current and current.get("status") == "completed":
                if expected and (not expected.is_file() or
                                 sha256(expected) != current["output"]["sha256"]):
                    raise RuntimeError(f"Completed stage {name}: expected output is missing or changed.")
                continue
            state["current_stage"] = name
            cwd = resolve_path(stage.get("cwd", "trainer"), repo)
            argv = list(stage["argv"])
            argv = continuation_resume_argv(argv, repo, resume or bool(
                current and current.get("status") in {"running", "failed", "interrupted"}))
            resume_checkpoint = checkpoint_path(stage, cwd)
            verified_resume = None
            if current and current.get("status") in {"running", "failed", "interrupted"}:
                if resume_checkpoint is not None:
                    valid = (resume_checkpoint.is_file() and
                             (resume_checkpoint.stat().st_mtime_ns >= current["started_unix_ns"] or
                              sha256(resume_checkpoint) == (current.get("initial_resume_checkpoint") or {}).get("sha256")))
                    recorded = current.get("resume_checkpoint")
                    if recorded and valid:
                        valid = sha256(resume_checkpoint) == recorded["sha256"]
                    if not valid:
                        raise RuntimeError(f"Stage {name}: no verified checkpoint for resume; use a new run_dir to restart.")
                    verified_resume = file_record(resume_checkpoint)
                    argv = set_flag(argv, "--from_resume", "1")
            required = [resolve_path(item, repo) for item in stage.get("requires", [])]
            if Path(argv[0]).name in OFFICIAL_TRAINERS:
                source = flag_value(argv, "--from_weight",
                                    "none" if Path(argv[0]).name == "train_pretrain.py" else "pretrain")
                if source != "none":
                    hidden = flag_value(argv, "--hidden_size", "768")
                    suffix = "_moe" if flag_value(argv, "--use_moe", "0") == "1" else ""
                    required.append((cwd / "../out" / f"{source}_{hidden}{suffix}.pth").resolve())
            for required_path in required:
                if not required_path.is_file():
                    raise RuntimeError(f"Stage {name}: required input missing: {required_path}")
            before = file_record(expected) if expected and expected.is_file() else None
            command = [sys.executable, "-u", *argv]
            attempt = (current or {}).get("attempt", 0) + 1
            current = {"status": "running", "attempt": attempt, "started_at": now(),
                       "started_unix_ns": time.time_ns(), "argv": command, "cwd": str(cwd),
                       "exit_code": None, "metrics_count": 0,
                       "initial_resume_checkpoint": verified_resume}
            state["stages"][name] = current
            persist()
            command_record = {"timestamp": now(), "stage": name, "attempt": attempt,
                              "argv": command, "cwd": str(cwd), "manifest_sha256": manifest_hash,
                              "code_commit": environment["code_commit"],
                              "script_sha256": environment["file_hashes"].get(str((cwd / argv[0]).resolve()))}
            append_json(run_dir / "commands.jsonl", command_record)
            with (run_dir / "logs" / f"{name}.log").open("a", encoding="utf-8") as log:
                log.write("\n# EVOMIND " + json.dumps(command_record, ensure_ascii=False) + "\n")
                log.flush()
                with managed_child(command, cwd=cwd, env=os.environ.copy(),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1) as child:
                    state["child_pid"] = child.pid
                    current["child_pid"] = child.pid
                    current["process_tree_managed"] = os.name == "nt"
                    persist()
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                        metric = parse_official_log(line)
                        if metric:
                            metric.update(timestamp=now(), stage=name, attempt=attempt)
                            append_json(run_dir / "metrics" / f"{name}.jsonl", metric)
                            current["metrics_count"] += 1
                            current["last_metric"] = metric
                            persist()
                            if current["metrics_count"] % 10 == 0:
                                plot(name)
                            if metric["nonfinite"]:
                                raise RuntimeError(f"Stage {name}: nonfinite training metric detected.")
                    returncode = child.wait()  # EOF does not imply successful process exit.
                    current["exit_code"] = returncode
                state["child_pid"] = None
                child = None
            plot(name)
            if returncode != 0:
                raise RuntimeError(f"Stage {name} exited with code {returncode}; later stages were not launched.")
            if expected:
                if not expected.is_file() or expected.stat().st_size == 0:
                    raise RuntimeError(f"Stage {name} succeeded but expected weight is missing/empty: {expected}")
                output = file_record(expected)
                if before and output == before:
                    raise RuntimeError(f"Stage {name} did not refresh its expected weight: {expected}")
                snapshot_dir = run_dir / "snapshots" / name
                snapshot_dir.mkdir(exist_ok=True)
                destination = snapshot_dir / expected.name
                shutil.copy2(expected, destination)
                if sha256(destination) != output["sha256"]:
                    raise RuntimeError(f"Stage {name}: checkpoint snapshot hash verification failed.")
                current.update(output=output, snapshot=file_record(destination))
                if resume_checkpoint and resume_checkpoint.is_file():
                    shutil.copy2(resume_checkpoint, snapshot_dir / resume_checkpoint.name)
            current.update(status="completed", finished_at=now())
            persist()
        state.update(status="completed", current_stage=None, exit_code=0, finished_at=now())
    except BaseException as error:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=15)
            if child.stdout:
                child.stdout.close()
            if current is not None:
                current["exit_code"] = child.returncode
        interrupted = isinstance(error, KeyboardInterrupt)
        state.update(status="interrupted" if interrupted else "failed", child_pid=None,
                     exit_code=130 if interrupted else 1, error=str(error), finished_at=now())
        if current is not None and current.get("status") != "completed":
            current.update(status=state["status"], error=str(error), finished_at=now())
            if "resume_checkpoint" in locals() and resume_checkpoint and resume_checkpoint.is_file():
                current["resume_checkpoint"] = file_record(resume_checkpoint)
        append_json(run_dir / "events.jsonl", {"timestamp": now(), "event": state["status"],
                    "stage": state["current_stage"], "error": str(error), "traceback": traceback.format_exc()})
        if state["current_stage"]:
            plot(state["current_stage"])
        print(f"EVOMIND {state['status']}: {error}", file=sys.stderr, flush=True)
    finally:
        persist()
        atomic_json(run_dir / "summary.json", {**state, "provenance": environment,
                    "metrics_note": "Observed training microbatch metrics only; no validation metrics inferred."})
    return state["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        return run_manifest(args.manifest, args.resume)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"EVOMIND refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
