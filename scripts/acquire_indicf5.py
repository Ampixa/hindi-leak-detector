"""Acquire pinned IndicF5 code/weights without exposing existing credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "ai4bharat/IndicF5"
REVISION = "fb652a96e508091045890dae5bb99fe511facbcf"
LOG = logging.getLogger(__name__)


def credential(path: Path) -> str:
    found = None
    for line in path.read_text().splitlines():
        fields = shlex.split(line, comments=True)
        if fields and fields[0] == "export":
            fields = fields[1:]
        for field in fields:
            key, separator, value = field.partition("=")
            if separator and key in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
                found = value
    if not found:
        raise ValueError("credential file contains no recognized Hugging Face token")
    return found


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def acquire(destination: Path, token: str) -> None:
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, token=token)
    if info.sha != REVISION:
        raise ValueError("unexpected IndicF5 revision")
    selected = [f for f in info.siblings if f.rfilename in {
        "README.md", "config.json", "model.py", "model.safetensors", "checkpoints/vocab.txt",
        "f5_tts/api.py", "f5_tts/infer/utils_infer.py", "f5_tts/infer/__init__.py",
    } or (f.rfilename.startswith("f5_tts/model/") and f.rfilename.endswith(".py"))]
    if not {"model.safetensors", "config.json", "model.py", "f5_tts/api.py"}.issubset({f.rfilename for f in selected}):
        raise ValueError("required inspected model files are absent")
    destination.mkdir(parents=True, exist_ok=True)
    inventory = []
    for entry in selected:
        path = Path(hf_hub_download(REPO, entry.rfilename, revision=REVISION,
                                   local_dir=destination, token=token))
        if path.stat().st_size != entry.size:
            raise ValueError(f"size mismatch: {entry.rfilename}")
        checksum = digest(path)
        expected = entry.lfs.sha256 if entry.lfs else None
        if expected and expected != checksum:
            raise ValueError(f"LFS checksum mismatch: {entry.rfilename}")
        inventory.append({"path": entry.rfilename, "size_bytes": entry.size,
                          "sha256": checksum, "expected_lfs_sha256": expected})
        LOG.info("Verified %s (%s bytes)", entry.rfilename, entry.size)
    record = {"repo": REPO, "revision": REVISION, "created_utc": datetime.now(timezone.utc).isoformat(),
              "files": inventory, "weights_lfs_verified": True}
    (destination / "inventory.json").write_text(json.dumps(record, indent=2) + "\n")
    acquire_vocos_config(destination)


def acquire_vocos_config(destination: Path) -> None:
    repo = "charactr/vocos-mel-24khz"
    revision = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
    directory = destination / "vocos"
    path = Path(hf_hub_download(repo, "config.yaml", revision=revision, local_dir=directory, token=False))
    if path.stat().st_size != 461:
        raise ValueError("unexpected pinned Vocos config size")
    record = {"repo": repo, "revision": revision, "files": [{"path": "config.yaml",
              "size_bytes": 461, "sha256": digest(path)}],
              "weights": "Already bundled in IndicF5 model.safetensors under vocoder._orig_mod.; no duplicate download"}
    (directory / "inventory.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--credential-env", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        acquire(args.destination, credential(args.credential_env))
    except Exception:
        LOG.exception("IndicF5 acquisition failed; no unverified model may be used")
        raise
