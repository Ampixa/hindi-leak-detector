"""Verify task-selected L40 audio and stream a tar archive; no scratch archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

ALLOWED_SOURCE = Path("/home/ampixa/tts_bench_v0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    args = parser.parse_args()
    entries = json.loads(args.selection.read_text())
    for entry in entries:
        path = Path(entry["source"]).resolve(strict=True)
        destination = PurePosixPath(entry["destination"])
        if not path.is_relative_to(ALLOWED_SOURCE) or destination.is_absolute() or ".." in destination.parts:
            raise ValueError("audio transfer path outside verified scope")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"source hash mismatch: {path}")
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        for entry in entries:
            archive.add(entry["source"], arcname=entry["destination"], recursive=False)


if __name__ == "__main__":
    main()
