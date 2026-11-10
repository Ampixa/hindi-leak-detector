"""Start the task-owned loopback review service and verify readiness."""
from __future__ import annotations

import json
import argparse
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--pack", type=Path, default=ROOT / "data/indicf5_audio8_v1")
    parser.add_argument("--labels", type=Path, default=ROOT / "annotations/labels.jsonl")
    parser.add_argument("--name", default="review_server")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not re.fullmatch(r"[a-z0-9_]+", args.name):
        raise ValueError("invalid task service port/name")
    label_path = args.labels.resolve()
    if not label_path.is_relative_to((ROOT / "annotations").resolve()):
        raise ValueError("review labels must remain under this project's annotations directory")
    directory = ROOT / "annotations"
    if not (directory / "README.md").is_file():
        raise ValueError("annotation provenance README is required")
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("port already in use; inspect the existing service before starting another")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with (directory / f"{args.name}.log").open("a") as log:
        process = subprocess.Popen([
            sys.executable, "-m", "hindi_detector.review_server",
            "--pack", str(args.pack.resolve(strict=True)),
            "--page", str(ROOT / "review/annotate.html"),
            "--labels", str(label_path), "--port", str(args.port),
        ], cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (directory / f"{args.name}.pid").write_text(str(process.pid) + "\n")
    last_error = None
    for _ in range(40):
        if process.poll() is not None:
            raise RuntimeError(f"review server exited with {process.returncode}; inspect annotations/review_server.log")
        try:
            with urlopen(f"http://127.0.0.1:{args.port}/api/items", timeout=2) as response:
                data = json.load(response)
            print(json.dumps({"pid": process.pid, "ready_clips": len(data["items"]),
                              "labels": len(data["labels"]), "url": f"http://127.0.0.1:{args.port}/"}))
            return
        except URLError as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"review server readiness timed out: {last_error}")


if __name__ == "__main__":
    main()
