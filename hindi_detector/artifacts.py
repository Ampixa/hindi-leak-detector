"""Small reproducible output groups with explicit retention documentation."""
from __future__ import annotations

import hashlib
import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path


def write_group(directory: Path, files: dict[str, object], purpose: str, inputs: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in [*files, "README.md"]:
        if (directory / name).exists():
            raise FileExistsError(f"refusing to overwrite retained artifact: {directory / name}")
    inventory = []
    for name, value in files.items():
        path = directory / name
        text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        path.write_text(text, encoding="utf-8")
        inventory.append(f"- `{name}`: {path.stat().st_size} bytes; SHA-256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`.")
    now = datetime.now(timezone.utc)
    doc = f"""# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: {socket.gethostname()}. Original directory: `{directory.resolve()}`.
Creation and last review (UTC): {now.isoformat()}. Status: active.
Next review: {(now + timedelta(days=7)).date().isoformat()}.

Purpose and current consumer: {purpose}

## File inventory

{chr(10).join(inventory)}

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

{chr(10).join('- ' + s for s in inputs)}

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
"""
    (directory / "README.md").write_text(doc, encoding="utf-8")
