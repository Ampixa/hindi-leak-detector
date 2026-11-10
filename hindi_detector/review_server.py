"""Loopback-only blind review; append-only human labels stay on k2."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import soundfile as sf

LOG = logging.getLogger(__name__)
LABELS = {"native_acceptable", "hindi_like", "other_error", "uncertain"}
REALIZATIONS = {"retained", "deleted", "reduced", "uncertain"}


def read_items(pack: Path) -> tuple[dict[str, dict], list[dict]]:
    selection = json.loads((pack / "selection.json").read_text())
    jobs = {j["id"]: j for j in selection["jobs"]}
    permitted_roots = [pack.resolve(strict=True)]
    declared_roots = list(selection.get("external_audio_roots", []))
    access_path = pack / "review_access.json"
    if access_path.exists():
        access = json.loads(access_path.read_text())
        if access.get("schema_version") != 1:
            raise ValueError("unsupported review access policy")
        declared_roots.extend(access["external_audio_roots"])
    for root in declared_roots:
        path = Path(root).resolve(strict=True)
        if not path.is_dir() or path == Path(path.anchor):
            raise ValueError("external audio root must be an explicit corpus directory")
        permitted_roots.append(path)
    items = {row["clip_id"]: dict(row) for row in selection["clips"]}
    receipt = pack / "indicf5/items.jsonl"
    if receipt.exists():
        for line in receipt.read_text().splitlines():
            row = json.loads(line)
            if row["status"] != "ok_unreviewed":
                raise ValueError("unexpected IndicF5 receipt status")
            job = jobs[row["id"]]
            if any(row[key] != job[key] for key in ["text", "speaker_id", "reference_sha256"]):
                raise ValueError("IndicF5 receipt does not match selected text, voice and reference")
            clip_id = hashlib.sha256(f"indicf5-stock:{row['id']}".encode()).hexdigest()[:24]
            items[clip_id] = {"clip_id": clip_id, "audio": row["audio"], "sha256": row["sha256"],
                              "system": "indicf5_stock", "sentence_id": row["id"],
                              "speaker_id": row["speaker_id"], "text": job["text"],
                              "split": job["split"]}
    public = []
    for clip_id, row in items.items():
        path = (pack / row["audio"]).resolve(strict=True)
        if not any(path.is_relative_to(root) for root in permitted_roots):
            raise ValueError("audio path outside pack and declared corpus roots")
        info = sf.info(path)
        row["duration"] = info.duration
        job = jobs[row["sentence_id"]]
        if row["split"] != job["split"] or row["split"] not in {"train", "development", "reserved_evaluation"}:
            raise ValueError("clip/job split mismatch")
        public.append({"id": clip_id, "text": row["text"], "duration": info.duration,
                       "audio": f"/audio/{clip_id}", "contrast_word": job.get("contrast_word"),
                       "category": job.get("category", "development"), "sort": hashlib.sha256(("blind-v1:" + clip_id).encode()).hexdigest()})
    return items, sorted(public, key=lambda row: row["sort"])


def validate_label(payload: dict, item: dict) -> dict:
    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, str) or not re.fullmatch(r"[\w .-]{1,64}", reviewer):
        raise ValueError("enter a reviewer name or initials (1–64 letters/numbers)")
    variety = payload.get("variety")
    if not isinstance(variety, str) or not 2 <= len(variety.strip()) <= 120:
        raise ValueError("describe the Nepali variety you are judging against")
    if payload.get("native_nepali") is not True:
        raise ValueError("this review round requires self-reported native Nepali competence")
    if payload.get("judgment") not in LABELS:
        raise ValueError("select an overall pronunciation judgment")
    if payload.get("prosody") not in LABELS:
        raise ValueError("select a separate melody/rhythm judgment, or uncertain")
    note = payload.get("note", "")
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("notes must be text under 2,000 characters")
    spans = payload.get("spans", [])
    if not isinstance(spans, list) or len(spans) > 20:
        raise ValueError("at most 20 word spans are allowed")
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("each span must be an object")
        start, end = span.get("start"), span.get("end")
        if any(type(v) not in {int, float} or not math.isfinite(v) for v in [start, end]):
            raise ValueError("span times must be finite seconds")
        if not 0 <= start < end <= item["duration"] + 0.02:
            raise ValueError("word span must fit inside the clip")
        if not isinstance(span.get("word"), str) or not 1 <= len(span["word"]) <= 120:
            raise ValueError("each span needs a target word")
        if span.get("realization") not in REALIZATIONS or span.get("judgment") not in LABELS:
            raise ValueError("invalid schwa realization or word judgment")
    return {"reviewer": reviewer.strip(), "variety": variety.strip(), "native_nepali": True,
            "judgment": payload["judgment"], "prosody": payload["prosody"], "note": note, "spans": spans}


def serve(pack: Path, page: Path, labels: Path, port: int) -> None:
    pack = pack.resolve(strict=True)
    page = page.resolve(strict=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    if not (labels.parent / "README.md").is_file():
        raise ValueError("annotation retention README is required before serving")
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, message: str, *args: object) -> None:
            LOG.info("%s %s", self.client_address[0], message % args)

        def respond(self, code: int, value: object) -> None:
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            try:
                route = urlparse(self.path).path
                if route in {"/", "/results"}:
                    document = page if route == "/" else page.parent / "results.html"
                    if not document.is_file():
                        self.respond(404, {"error": "results not available yet"})
                        return
                    content = document.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return
                items, public = read_items(pack)
                if route == "/api/items":
                    with lock:
                        rows = [json.loads(line) for line in labels.read_text().splitlines()] if labels.exists() else []
                    splits = {item["split"] for item in items.values()}
                    notice = ("Reserved evaluation: these reviews never enter training." if splits == {"reserved_evaluation"}
                              else "Separate training/development set: benchmark recordings are excluded.")
                    self.respond(200, {"items": public, "labels": rows, "notice": notice})
                elif route.startswith("/audio/") and route.removeprefix("/audio/") in items:
                    item = items[route.removeprefix("/audio/")]
                    path = pack / item["audio"]
                    payload = path.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                        raise ValueError("audio checksum mismatch")
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.respond(404, {"error": "resource not found"})
            except (BrokenPipeError, ConnectionResetError):
                LOG.info("Review client disconnected during audio/page transfer")
            except Exception:
                LOG.exception("Review read failed")
                self.respond(500, {"error": "server read failed; inspect the logged traceback"})

        def do_POST(self) -> None:
            try:
                if urlparse(self.path).path != "/api/labels":
                    self.respond(404, {"error": "resource not found"})
                    return
                origin = self.headers.get("Origin")
                host = self.headers.get("Host")
                if not host or (origin and origin != f"http://{host}"):
                    raise ValueError("cross-origin label writes are not allowed")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 32768:
                    raise ValueError("label request must be between 1 and 32,768 bytes")
                payload = json.loads(self.rfile.read(size))
                items, _ = read_items(pack)
                if not isinstance(payload, dict) or payload.get("clip_id") not in items:
                    raise ValueError("unknown clip ID")
                item = items[payload["clip_id"]]
                label = validate_label(payload, item)
                label.update({"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                              "source": "human_blind_review", "clip_id": item["clip_id"],
                              "audio_sha256": item["sha256"], "sentence_id": item["sentence_id"],
                              "speaker_id": item["speaker_id"], "split": item["split"]})
                with lock, labels.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(label, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self.respond(201, {"saved": True, "label": label})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                LOG.warning("Invalid review submission: %s", exc)
                self.respond(400, {"error": str(exc)})
            except Exception:
                LOG.exception("Review write failed")
                self.respond(500, {"error": "label was not confirmed saved; inspect the logged traceback"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    LOG.info("Blind review listening on http://127.0.0.1:%s", port)
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    serve(args.pack, args.page, args.labels, args.port)
