"""Atomic, content-addressed caches of FROZEN encoder outputs, not projections."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile

import torch

from .views import VIEW_VERSION, num_views

CACHE_VERSION = "evomind-encoder-features-v1"
_DTYPES = {"float32": torch.float32, "float16": torch.float16}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_vision(encoder_path, processor):
    """Hash local encoder weight/config CONTENT and exact preprocessing config.

    Call once per cache preparation run, not once per image. Model weights must
    already be local; this function does not download or execute any model code.
    Software versions participate because processing implementations can change.
    """
    root = Path(encoder_path)
    if root.is_file():
        files = [root]
        base = root.parent
    elif root.is_dir():
        base = root
        files = sorted(
            path for path in root.rglob("*")
            if path.is_file() and (path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}
                                   or path.name in {"config.json", "preprocessor_config.json",
                                                    "processor_config.json"}
                                   or path.name.endswith(".index.json"))
        )
    else:
        raise FileNotFoundError(f"local vision encoder not found: {root}")
    if not any(p.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"} for p in files):
        raise ValueError(f"no local vision encoder weight files under {root}")
    if not hasattr(processor, "to_dict"):
        raise TypeError("processor must expose to_dict() for a reproducible fingerprint")
    versions = {}
    for package in ("transformers", "Pillow", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    payload = {
        "weights_and_configs": [(str(p.relative_to(base)).replace("\\", "/"), _sha256_file(p)) for p in files],
        "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
        "processor": processor.to_dict(),
        "software": versions,
        "view_version": VIEW_VERSION,
        "encoder_compute_dtype": "float32",
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


class VisionFeatureCache:
    """Store one CPU tensor [V, 64, D] per original image.

    Entries contain only tensors and primitive metadata. Reads explicitly use
    torch.load(weights_only=True); arbitrary pickle objects are never accepted.
    Float16 storage is lossy and therefore has a distinct key namespace.
    Corruption/mismatched metadata raises an error instead of silently reusing
    an invalid tensor. A missing key returns None.
    """

    def __init__(self, root, fingerprint, mode="single", dtype="float32",
                 tokens_per_view=64, hidden_size=768):
        if dtype not in _DTYPES:
            raise ValueError("cache dtype must be float32 or float16")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("a nonempty encoder/preprocessing fingerprint is required")
        if tokens_per_view != 64 or hidden_size < 1:
            raise ValueError("expected 64 tokens per view and a positive hidden size")
        self.root = Path(root)
        self.fingerprint = fingerprint
        self.mode = mode
        self.dtype = dtype
        self.shape = (num_views(mode), tokens_per_view, hidden_size)
        self.metadata = {
            "cache_version": CACHE_VERSION,
            "view_version": VIEW_VERSION,
            "fingerprint": fingerprint,
            "mode": mode,
            "dtype": dtype,
            "shape": list(self.shape),
            "stage": "frozen_vision_encoder_output_before_trainable_projector",
            "encoder_compute_dtype": "float32",
        }

    def key_for(self, image):
        """Use raw image file bytes (or supplied bytes), never its path/mtime."""
        if isinstance(image, (bytes, bytearray, memoryview)):
            image_hash = hashlib.sha256(image).hexdigest()
        else:
            image_hash = _sha256_file(image)
        return hashlib.sha256(_canonical({**self.metadata, "image_sha256": image_hash})).hexdigest()

    def path_for(self, key):
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("cache key must be a lowercase SHA256 digest")
        return self.root / key[:2] / f"{key}.pt"

    def _validate(self, features):
        if not isinstance(features, torch.Tensor) or tuple(features.shape) != self.shape:
            raise ValueError(f"encoder features must have shape {self.shape}")
        if features.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("encoder features must be floating point")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("encoder features contain non-finite values")

    def load(self, key):
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("metadata") != {**self.metadata, "key": key}:
                raise ValueError("metadata/fingerprint mismatch")
            features = payload.get("features")
            self._validate(features)
            if features.dtype != _DTYPES[self.dtype]:
                raise ValueError("tensor dtype does not match cache metadata")
            return features.detach().contiguous()
        except Exception as exc:
            raise ValueError(f"invalid vision feature cache entry {path}: {exc}") from exc

    def save(self, key, features):
        self._validate(features)
        if features.requires_grad:
            raise ValueError("cache accepts detached frozen encoder outputs only")
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = features.detach().to(device="cpu", dtype=_DTYPES[self.dtype]).contiguous()
        self._validate(stored)  # Detect float16 conversion overflow as well.
        payload = {"metadata": {**self.metadata, "key": key}, "features": stored}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=f".{key}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return path
