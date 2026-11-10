"""Fit a small frozen-encoder head only after real training reviews exist."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import ENCODER_REVISION, require_feature_recipe
from hindi_detector.supervision import approved_labels, require_training_labels
from hindi_detector.review_server import read_items

LOG = logging.getLogger(__name__)


def load_features(directory: Path, labels: list[dict]) -> tuple[np.ndarray, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    require_feature_recipe(manifest)
    indexed = {row["clip_id"]: row for row in manifest["items"]}
    features, selected = [], []
    for label in labels:
        row = indexed[label["clip_id"]]
        for key in ["audio_sha256", "speaker_id", "sentence_id", "split"]:
            if row[key] != label[key]:
                raise ValueError(f"feature/review metadata mismatch: {key}")
        path = (directory / row["path"]).resolve(strict=True)
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("feature path outside cache")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("feature checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            vector = data["speech"].copy()
        if vector.shape != (4096,) or not np.isfinite(vector).all():
            raise ValueError("invalid feature vector")
        features.append(vector)
        selected.append(row)
    return np.stack(features), selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--variety", required=True)
    parser.add_argument("--minimum-raters", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train, train_counts = approved_labels(args.annotations, variety=args.variety, split="train",
                                          minimum_raters=args.minimum_raters)
    require_training_labels(train)
    dev, dev_counts = approved_labels(args.annotations, variety=args.variety, split="development",
                                      minimum_raters=args.minimum_raters)
    if len(dev) < 4 or {r["class_id"] for r in dev} != {0, 1}:
        raise ValueError("need at least four approved development clips covering both classes")
    if {r["speaker_id"] for r in train} & {r["speaker_id"] for r in dev}:
        raise ValueError("speaker leakage between training and development")
    if {r["sentence_id"] for r in train} & {r["sentence_id"] for r in dev}:
        raise ValueError("sentence leakage between training and development")
    x_train, train_meta = load_features(args.features, train)
    x_dev, dev_meta = load_features(args.features, dev)
    if {r["text_sha256"] for r in train_meta} & {r["text_sha256"] for r in dev_meta}:
        raise ValueError("same normalized sentence appears under different IDs across training/development")
    if any(row.get("split") == "reserved_evaluation" for row in train_meta + dev_meta):
        raise ValueError("reserved evaluation features reached the training pipeline")
    source_items, _ = read_items(args.pack)
    for label in train + dev:
        if source_items[label["clip_id"]]["sha256"] != label["audio_sha256"]:
            raise ValueError("review/pack audio identity mismatch")
    mean, scale = x_train.mean(0), np.maximum(x_train.std(0), 1e-4)
    x = torch.tensor((x_train - mean) / scale, dtype=torch.float32)
    y = torch.tensor([r["class_id"] for r in train], dtype=torch.float32)
    xd = torch.tensor((x_dev - mean) / scale, dtype=torch.float32)
    yd = np.array([r["class_id"] for r in dev])
    torch.set_num_threads(2)
    torch.manual_seed(0)
    head = nn.Linear(4096, 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.1)
    positive_weight = torch.tensor(float((y == 0).sum() / (y == 1).sum()))
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    best_loss, best_state, best_step = float("inf"), None, None
    for step in range(1, 1001):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(head(x).squeeze(1), y)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            with torch.no_grad():
                dev_loss = float(nn.functional.binary_cross_entropy_with_logits(head(xd).squeeze(1), torch.tensor(yd, dtype=torch.float32)))
            if dev_loss < best_loss:
                best_loss = dev_loss
                best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
                best_step = step
    if best_state is None:
        raise RuntimeError("training selected no valid checkpoint")
    head.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        probabilities = torch.sigmoid(head(xd).squeeze(1)).numpy()
    predictions = probabilities >= 0.5
    tp, fp = int(((predictions == 1) & (yd == 1)).sum()), int(((predictions == 1) & (yd == 0)).sum())
    fn, tn = int(((predictions == 0) & (yd == 1)).sum()), int(((predictions == 0) & (yd == 0)).sum())
    model = {"architecture": "frozen XLSR layers 12/24 mean+std -> standardized 4096D -> linear binary head",
             "variety": args.variety, "class_ids": {"native_acceptable": 0, "hindi_like": 1},
             "weight": head.weight.detach().numpy()[0].tolist(), "bias": float(head.bias.detach()[0]),
             "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
             "encoder_revision": ENCODER_REVISION,
             "training_speaker_ids": sorted({r["speaker_id"] for r in train_meta}),
             "development_speaker_ids": sorted({r["speaker_id"] for r in dev_meta}),
             "training_text_sha256": sorted({r["text_sha256"] for r in train_meta}),
             "development_text_sha256": sorted({r["text_sha256"] for r in dev_meta}),
             "training_systems": sorted({source_items[r["clip_id"]]["system"] for r in train}),
             "development_systems": sorted({source_items[r["clip_id"]]["system"] for r in dev}),
             "status": "exploratory_pronunciation_head_not_calibrated", "minimum_raters": args.minimum_raters}
    report = {"train_counts": train_counts, "development_counts": dev_counts, "selected_step": best_step,
              "development_bce": best_loss, "development_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
              "development_precision": tp / (tp + fp) if tp + fp else None,
              "development_recall": tp / (tp + fn) if tp + fn else None,
              "held_out_test_performance": None,
              "limitations": ["Development data selects the checkpoint and is not an independent accuracy claim.",
                              "Pooled embeddings can retain speaker/synthesizer cues; system-held-out evaluation is required."]}
    write_group(args.out, {"model.json": model, "training_report.json": report},
                "Human-supervised pronunciation head; independent held-out evaluation still required.",
                [f"annotations {args.annotations.resolve()} sha256={hashlib.sha256(args.annotations.read_bytes()).hexdigest()}",
                 f"features {args.features.resolve()}", f"driver sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}"])
    print(json.dumps(report))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception("Training did not complete; no success or substitute labels may be reported")
        raise
