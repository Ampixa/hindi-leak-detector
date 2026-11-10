"""Acquire the inspected phoneme model in this experiment's external-disk cache."""
from __future__ import annotations

import hashlib
import json
import logging
import socket
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
REVISION = "2c733782da5604684829819a5eb744c193fe9398"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models/phoneme_probe"
FILES = ["config.json", "preprocessor_config.json", "vocab.json", "tokenizer_config.json",
         "special_tokens_map.json", "README.md", "pytorch_model.bin"]


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    inventory = []
    for name in FILES:
        path = Path(hf_hub_download(MODEL, name, revision=REVISION, local_dir=DEST, token=False))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        inventory.append({"path": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        logging.info("Verified local file %s (%d bytes)", name, path.stat().st_size)
    (DEST / "inventory.json").write_text(json.dumps({"model": MODEL, "revision": REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
        "files": inventory}, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        main()
    except Exception:
        logging.exception("Pinned model acquisition failed; preserve traceback and inspect partial downloads")
        raise
