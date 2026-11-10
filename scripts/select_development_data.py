"""Select independent native development texts; never infer leakage labels."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/Volumes/T9/indicvoices_r_ne/groups.jsonl")
ALLOWED_AUDIO = Path("/Volumes/T9/indicvoices_r_ne/audio")
OUT = ROOT / "data/development_v1"


def normalized(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFC", text))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    if not (OUT / "README.md").is_file():
        raise ValueError("development data provenance README is required")
    protected = [normalized(r["text_devanagari"]) for r in json.loads((ROOT / ".sources/neptts/benchmark/sentences.json").read_text())]
    groups = defaultdict(list)
    for number, line in enumerate(SOURCE.open(), 1):
        row = json.loads(line)
        if set(row) != {"path", "duration", "transcript", "source"}:
            raise ValueError(f"unexpected native manifest schema at line {number}")
        if not 2 <= row["duration"] <= 8 or not 12 <= len(row["transcript"]) <= 160:
            continue
        if not row["source"].startswith("indicvoices_r/") or re.search(r"[A-Za-z]", row["transcript"]):
            continue
        text = normalized(row["transcript"])
        overlap = any(text == benchmark or (min(len(text), len(benchmark)) >= 15 and
                      (text in benchmark or benchmark in text or SequenceMatcher(None, text, benchmark).ratio() >= 0.9))
                      for benchmark in protected)
        if overlap:
            continue
        path = Path(row["path"]).resolve(strict=True)
        if not path.is_relative_to(ALLOWED_AUDIO) or path.suffix != ".wav":
            raise ValueError("native source path outside inspected corpus")
        groups[row["source"]].append(row)
    speakers = [name for name, rows in groups.items() if len({normalized(r["transcript"]) for r in rows}) >= 3]
    speakers.sort(key=lambda name: hashlib.sha256(("dev-v1:" + name).encode()).hexdigest())
    if len(speakers) < 16:
        raise ValueError("fewer than sixteen eligible native speaker groups")
    jobs, native = [], []
    for index, name in enumerate(speakers[:16]):
        rows = sorted(groups[name], key=lambda row: (row["duration"], row["path"]))
        unique = {}
        for row in rows:
            unique.setdefault(normalized(row["transcript"]), row)
        chosen = list(unique.values())[:3]
        reference, targets = chosen[0], chosen[1:]
        split = "train" if index < 12 else "development"
        speaker_id = name.removeprefix("indicvoices_r/")
        for target_index, target in enumerate(targets):
            path = Path(target["path"])
            info = sf.info(path)
            if info.channels != 1 or abs(info.duration - target["duration"]) > 0.02:
                raise ValueError("native duration/channel metadata mismatch")
            item_id = f"dev_{index:02d}_{target_index}"
            job = {"id": item_id, "text": target["transcript"], "speaker_id": speaker_id,
                   "reference_audio": reference["path"], "reference_text": reference["transcript"],
                   "reference_sha256": digest(Path(reference["path"])), "seed": 0, "split": split,
                   "native_audio": target["path"], "native_sha256": digest(path),
                   "source_group": name, "review_status": "unreviewed"}
            jobs.append(job)
            native.append({"clip_id": hashlib.sha256(("native-dev:" + item_id).encode()).hexdigest()[:24],
                           "system": "native_recording", "sentence_id": item_id,
                           "speaker_id": speaker_id, "text": target["transcript"],
                           "audio": target["path"], "sha256": job["native_sha256"],
                           "split": split, "automatic_leakage_label": None})
    path = OUT / "selection.json"
    train_texts = {normalized(j["text"]) for j in jobs if j["split"] == "train"}
    development_texts = {normalized(j["text"]) for j in jobs if j["split"] == "development"}
    if train_texts & development_texts:
        raise ValueError("normalized text overlap across native training/development speakers")
    if path.exists():
        raise FileExistsError("inspect the existing development selection before replacement")
    record = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "creator": "Codex GPT-6 /root", "source_manifest": str(SOURCE),
              "source_manifest_sha256": digest(SOURCE), "jobs": jobs, "clips": native,
              "external_audio_roots": [str(ALLOWED_AUDIO)],
              "speaker_id_evidence": "Inspected prepare_indicvoices_r.py maps source to original row speaker_id; historical execution revision unknown",
              "counts": {"speakers": 16, "native_targets": 32, "train_speakers": 12, "development_speakers": 4},
              "benchmark_text_overlap_after_screen": 0,
              "limitations": ["Corpus transcripts and natural speech require review, not automatic negative labels.",
                              "Cross-corpus speaker identity versus opaque benchmark voice IDs is unverified."]}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record["counts"]))


if __name__ == "__main__":
    main()
