"""Consume genuine review events without turning source identity into labels."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

CLASS_IDS = {"native_acceptable": 0, "hindi_like": 1}


def approved_labels(path: Path, *, variety: str, split: str, minimum_raters: int = 2) -> tuple[list[dict], dict]:
    if minimum_raters < 1 or not variety.strip():
        raise ValueError("a named target variety and at least one rater are required")
    latest = {}
    counts = Counter()
    if not path.exists():
        return [], {"human_review_events": 0, "approved_clips": 0}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank annotation record at line {number}")
        event = json.loads(line)
        if event.get("source") != "human_blind_review" or event.get("native_nepali") is not True:
            raise ValueError(f"non-human/unqualified supervision at line {number}")
        for key in ["clip_id", "reviewer", "audio_sha256", "speaker_id", "sentence_id", "variety", "split"]:
            if not isinstance(event.get(key), str) or not event[key]:
                raise ValueError(f"missing annotation field {key}, line {number}")
        if len(event["audio_sha256"]) != 64:
            raise ValueError("annotation audio hash must be SHA-256")
        counts["human_review_events"] += 1
        if event["variety"] == variety:
            latest[event["clip_id"], event["reviewer"]] = event
    groups = defaultdict(list)
    for (clip_id, _), event in latest.items():
        groups[clip_id].append(event)
    output = []
    for clip_id, events in groups.items():
        if any(e["split"] != split for e in events):
            counts["excluded_wrong_split"] += 1
            continue
        if split == "train" and any(e["split"] == "reserved_evaluation" for e in events):
            raise ValueError("reserved benchmark reviews cannot enter training")
        if len({e["audio_sha256"] for e in events}) != 1:
            raise ValueError(f"different audio versions were reviewed under {clip_id}")
        for key in ["speaker_id", "sentence_id"]:
            if len({e[key] for e in events}) != 1:
                raise ValueError(f"conflicting {key} in reviews of {clip_id}")
        if len(events) < minimum_raters:
            counts["awaiting_additional_raters"] += 1
            continue
        decisions = {e["judgment"] for e in events}
        if len(decisions) != 1 or not decisions.issubset(CLASS_IDS):
            counts["excluded_uncertain_other_or_disputed"] += 1
            continue
        output.append({"clip_id": clip_id, "class_id": CLASS_IDS[events[0]["judgment"]],
                       "audio_sha256": events[0]["audio_sha256"], "speaker_id": events[0]["speaker_id"],
                       "sentence_id": events[0]["sentence_id"], "variety": variety,
                       "split": split, "rater_count": len(events)})
    counts["approved_clips"] = len(output)
    counts["minimum_raters"] = minimum_raters
    return output, dict(counts)


def require_training_labels(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no approved training labels; evaluation reviews and model identity cannot substitute")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("training received an evaluation/development item")
    counts = Counter(row["class_id"] for row in rows)
    if set(counts) != {0, 1} or min(counts.values()) < 8:
        raise ValueError(f"need at least eight human-approved clips in each class; got {dict(counts)}")
    if len({row["speaker_id"] for row in rows}) < 4:
        raise ValueError("need at least four distinct labeled speaker groups for a training pilot")
