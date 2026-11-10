"""Select a reserved evaluation/review pack from inspected source receipts."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/indicf5_audio8_v1"


def opaque(role: str, item: str) -> str:
    return hashlib.sha256(f"hindi-leakage-v1:{role}:{item}".encode()).hexdigest()[:24]


def main() -> None:
    source = json.loads((DEST / "source_manifest.json").read_text())
    sentences = {r["sent_id"]: r for r in json.loads((ROOT / ".sources/neptts/benchmark/sentences.json").read_text())}
    receipts = {}
    for name in ["r2", "r3"]:
        rows = [json.loads(line) for line in (DEST / f"source_audio8_{name}.jsonl").read_text().splitlines()]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"duplicate synthesis receipts in {name}")
        receipts[name] = {r["id"]: r for r in rows}
    eligible = source["human_jobs"]
    for row in eligible:
        benchmark = sentences[row["id"]]
        if row["text"] != benchmark["text_devanagari"]:
            raise ValueError(f"benchmark/source text mismatch for {row['id']}")
        for name, rows in receipts.items():
            receipt = rows[row["id"]]
            if receipt["status"] != "ok" or receipt["speaker_id"] != row["speaker_id"]:
                raise ValueError(f"invalid matched receipt {name}/{row['id']}")
    schwa = [r for r in eligible if "schwa" in str(sentences[r["id"]].get("phonetic_targets", ""))]
    questions = [r for r in eligible if r["category"] == "question_intonation"]
    focus = [r for r in eligible if r["category"] == "contrastive_stress"][:6]
    selected = schwa + questions + focus
    if len(selected) != 24 or len({r["id"] for r in selected}) != 24:
        raise ValueError(f"unexpected pilot shape: {len(selected)} items")
    if len({r["speaker_id"] for r in selected}) != 8:
        raise ValueError("pilot does not cover all eight available voices")
    files: dict[str, dict] = {}
    jobs = []

    def add(role: str, item: str, path: str, checksum: str) -> str:
        destination = f"audio/{opaque(role, item)}.wav"
        previous = files.get(destination)
        entry = {"source_host": "khalti-gpu", "source": path, "destination": destination, "sha256": checksum}
        if previous and previous != entry:
            raise ValueError("conflicting copy destination")
        files[destination] = entry
        return destination

    clips = []
    for row in selected:
        sid, voice = row["id"], row["speaker_id"]
        reference = row["reference"]
        ref_path = add("reference", voice, reference["audio"], reference["sha256"])
        benchmark = sentences[sid]
        jobs.append({"id": sid, "text": row["text"], "speaker_id": voice,
                     "reference_audio": ref_path, "reference_text": reference["text"],
                     "reference_sha256": reference["sha256"], "seed": row["seed"],
                     "category": row["category"], "phonetic_targets": benchmark.get("phonetic_targets"),
                     "contrast_word": benchmark.get("contrast_word"), "split": "reserved_evaluation"})
        sources = [("native_recording", row["audio"], row["sha256"]),
                   *[(f"audio8_dpo_{name.upper()}", receipts[name][sid]["audio"], receipts[name][sid]["sha256"])
                     for name in ["r2", "r3"]]]
        for role, path, checksum in sources:
            audio = add(role, sid, path, checksum)
            clips.append({"clip_id": opaque(role, sid), "system": role, "sentence_id": sid,
                          "speaker_id": voice, "text": row["text"], "audio": audio,
                          "sha256": checksum, "review_status": "unreviewed",
                          "split": "reserved_evaluation", "automatic_leakage_label": None})
    payload = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "created_by": "Codex GPT-6 /root", "purpose": "Matched IndicF5/audio8/native evaluation and blind review",
               "source_manifest_sha256": hashlib.sha256((DEST / "source_manifest.json").read_bytes()).hexdigest(),
               "jobs": jobs, "clips": clips, "benchmark_excluded_from_training": True,
               "counts": {"sentences": len(jobs), "voices": 8, "existing_clips": len(clips),
                          "categories": dict(Counter(j["category"] for j in jobs))},
               "limitations": ["Source benchmark text/contrast metadata is not acoustic gold.",
                               "Native recordings require feature-specific review too.",
                               "No label is assigned from TTS system identity."]}
    for name, value in [("selection.json", payload), ("transfer_files.json", list(files.values()))]:
        path = DEST / name
        if path.exists():
            raise FileExistsError(f"inspect existing selection before changing it: {path}")
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["counts"]))
    print("audio transfers", len(files))


if __name__ == "__main__":
    main()
