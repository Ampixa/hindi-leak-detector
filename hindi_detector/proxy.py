"""Explicit source-level proxy supervision, separate from human judgments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from hindi_detector.evaluation import require_feature_recipe
from hindi_detector.review_server import read_items

PROXY_LABELS = {"indicf5_stock": 1, "audio8_r3": 0}
LABEL_SOURCE = "user_authorized_source_proxy_not_human_gold"


def load_pack(directory: Path, pack: Path) -> tuple[list[dict], np.ndarray]:
    manifest = json.loads((directory / "manifest.json").read_text())
    require_feature_recipe(manifest)
    source, _ = read_items(pack)
    rows, vectors = [], []
    if set(source) != {row["clip_id"] for row in manifest["items"]}:
        raise ValueError("source/feature coverage mismatch")
    for row in manifest["items"]:
        item = source[row["clip_id"]]
        if row["audio_sha256"] != item["sha256"]:
            raise ValueError("source/feature audio identity mismatch")
        for key in ["speaker_id", "sentence_id", "split"]:
            if row[key] != item[key]:
                raise ValueError(f"source/feature metadata mismatch: {key}")
        if hashlib.sha256((pack / item["audio"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("source audio checksum mismatch")
        path = (directory / row["path"]).resolve(strict=True)
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("feature path outside cache")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("feature checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            vector = data["speech"].copy()
        if vector.shape != (4096,) or not np.isfinite(vector).all():
            raise ValueError("invalid speech feature vector")
        # The inspected original benchmark calls the same R3 checkpoint
        # audio8_dpo_R3; retain that raw name alongside the explicit alias.
        system = {"audio8_dpo_R3": "audio8_r3"}.get(item["system"], item["system"])
        rows.append({**row, "system": system, "source_system": item["system"], "text": item["text"],
                     "class_id": PROXY_LABELS.get(system), "label_source": LABEL_SOURCE})
        vectors.append(vector)
    return rows, np.stack(vectors)


def check_proxy_pairs(rows: list[dict], split: str) -> None:
    groups = {}
    for row in rows:
        if row["split"] != split or row["system"] not in PROXY_LABELS:
            raise ValueError("unexpected proxy split/system")
        if row["class_id"] != PROXY_LABELS[row["system"]] or row["label_source"] != LABEL_SOURCE:
            raise ValueError("proxy labels must retain their explicit source assumption")
        key = (row["speaker_id"], row["sentence_id"], row["text_sha256"])
        systems = groups.setdefault(key, [])
        systems.append(row["system"])
    if not groups or any(sorted(s) != sorted(PROXY_LABELS) for s in groups.values()):
        raise ValueError("each text/voice needs exactly one IndicF5 and one Audio8 R3 output")


def check_disjoint(left: list[dict], right: list[dict]) -> None:
    for key in ["speaker_id", "sentence_id", "text_sha256", "audio_sha256"]:
        if {r[key] for r in left} & {r[key] for r in right}:
            raise ValueError(f"proxy split leakage: {key}")
