"""Download only inspected source files, checking Git object hashes and sizes."""
from __future__ import annotations

import hashlib
import json
import logging
import socket
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def fetch() -> None:
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    inventory = []
    for source in lock["sources"]:
        directory = ROOT / source["directory"]
        for entry in source["files"]:
            path = directory / entry["path"]
            url = f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/{entry['path']}"
            if path.exists():
                payload = path.read_bytes()
            else:
                with urlopen(url, timeout=45) as response:
                    payload = response.read(entry["size_bytes"] + 1)
            digest = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            if len(payload) != entry["size_bytes"] or digest != entry["git_blob_sha1"]:
                raise ValueError(f"source verification failed: {path}; remove corrupt copy and retry")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            inventory.append({"path": str(path.relative_to(ROOT)), "url": url,
                              "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        LOG.info("Verified %s @ %s", source["repository"], source["revision"])
    (ROOT / ".sources/inventory.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "files": inventory,
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        fetch()
    except (OSError, ValueError, KeyError, URLError, socket.timeout):
        LOG.exception("Source acquisition failed; no unverified input may be used")
        raise SystemExit(1)
