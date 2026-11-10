"""Evaluate a real trained head on independent human-reviewed reserved audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import binary_metrics, grouped_intervals, predict_linear, require_feature_recipe, require_held_out
from hindi_detector.review_server import read_items
from hindi_detector.supervision import approved_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--minimum-raters", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.model.read_text())
    labels, counts = approved_labels(args.annotations, variety=model["variety"],
                                     split="reserved_evaluation", minimum_raters=args.minimum_raters)
    if not labels:
        raise ValueError("no approved independent evaluation judgments; accuracy cannot be computed")
    manifest = json.loads((args.features / "manifest.json").read_text())
    require_feature_recipe(manifest)
    if manifest["model_revision"] != model["encoder_revision"] or manifest["feature_dimension"] != 4096:
        raise ValueError("evaluation encoder does not match the trained head")
    features = {row["clip_id"]: row for row in manifest["items"]}
    items, _ = read_items(args.pack)
    rows, vectors = [], []
    for label in labels:
        row, source = features[label["clip_id"]], items[label["clip_id"]]
        if row["audio_sha256"] != label["audio_sha256"] or source["sha256"] != label["audio_sha256"]:
            raise ValueError("evaluation source/review/feature audio identity mismatch")
        for key in ["speaker_id", "sentence_id", "split"]:
            if row[key] != label[key] or source[key] != label[key]:
                raise ValueError(f"evaluation source/review/feature metadata mismatch: {key}")
        path = (args.features / row["path"]).resolve(strict=True)
        if not path.is_relative_to(args.features.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("evaluation feature checksum/path mismatch")
        with np.load(path, allow_pickle=False) as stored:
            vectors.append(stored["speech"].copy())
        rows.append({**row, "class_id": label["class_id"], "system": source["system"],
                     "raters": label["rater_count"]})
    require_held_out(model, rows)
    probabilities = predict_linear(model, np.stack(vectors))
    y = np.array([row["class_id"] for row in rows])
    per_system = {}
    for system in sorted({row["system"] for row in rows}):
        indices = np.array([i for i, row in enumerate(rows) if row["system"] == system])
        per_system[system] = binary_metrics(y[indices], probabilities[indices])
    seen = set(model["training_systems"]) | set(model["development_systems"])
    unseen = np.array([i for i, row in enumerate(rows) if row["system"] not in seen], dtype=int)
    report = {"status": "human_reviewed_pilot_evaluation", "annotation_counts": counts,
              "threshold": 0.5, "threshold_selected_on_test": False,
              "overall": binary_metrics(y, probabilities), "per_system": per_system,
              "uncertainty": grouped_intervals(y, probabilities, [r["speaker_id"] for r in rows]),
              "unseen_systems": sorted({rows[i]["system"] for i in unseen}),
              "unseen_system_performance": binary_metrics(y[unseen], probabilities[unseen]) if len(unseen) else None,
              "limits": ["Speaker IDs and normalized texts are disjoint, but cross-corpus real-person identity remains unverified.",
                         "Source identity never supplies a reference label; disputed/uncertain labels remain excluded and counted.",
                         "A Hindi-like judgment does not prove its causal origin in a Hindi initialization.",
                         "Do not interpret tiny/perfect pilot confidence intervals as production qualification."]}
    predictions = [{"clip_id": row["clip_id"], "reference_class": row["class_id"],
                    "hindi_like_score": float(score), "predicted_class": int(score >= 0.5),
                    "system": row["system"], "rater_count": row["raters"]}
                   for row, score in zip(rows, probabilities, strict=True)]
    write_group(args.out, {"evaluation.json": report, "predictions.json": predictions},
                "Independent human-reviewed pilot evaluation of the actual trained pronunciation head.",
                [f"model sha256={hashlib.sha256(args.model.read_bytes()).hexdigest()}",
                 f"labels sha256={hashlib.sha256(args.annotations.read_bytes()).hexdigest()}",
                 f"feature directory {args.features.resolve()}"])
    print(json.dumps(report))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception("Evaluation failed; no accuracy result is available")
        raise
