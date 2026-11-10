"""Cache compact frozen-encoder features, never source-derived leakage labels."""
from __future__ import annotations

import hashlib
import argparse
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModelForCTC

from hindi_detector.review_server import read_items

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "data/indicf5_audio8_v1"
MODEL = ROOT / "models/phoneme_probe"
OUT = ROOT / "reports/comparison_features_v1"
LOG = logging.getLogger(__name__)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    pack, out = args.pack.resolve(strict=True), args.out.resolve(strict=True)
    if not (out / "README.md").is_file():
        raise ValueError("feature cache retention README is required")
    items, _ = read_items(pack)
    revision = "2c733782da5604684829819a5eb744c193fe9398"
    inventory = json.loads((MODEL / "inventory.json").read_text())
    if inventory["model"] != "facebook/wav2vec2-xlsr-53-espeak-cv-ft" or inventory["revision"] != revision:
        raise ValueError("unexpected encoder inventory")
    for entry in inventory["files"]:
        path = MODEL / entry["path"]
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"encoder file verification failed: {path}")
    driver_hash = digest(Path(__file__))
    torch.set_num_threads(2)
    extractor = AutoFeatureExtractor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(MODEL, local_files_only=True).eval()
    if model.config.hidden_size != 1024 or model.config.num_hidden_layers != 24:
        raise ValueError("unexpected acoustic encoder structure")
    metadata = []
    for clip_id, row in sorted(items.items()):
        destination = out / f"{clip_id}.npz"
        waveform_path = pack / row["audio"]
        if digest(waveform_path) != row["sha256"]:
            raise ValueError("source audio hash mismatch")
        if not destination.exists():
            wave, sr = sf.read(waveform_path, dtype="float32", always_2d=True)
            if wave.shape[1] != 1 or not np.isfinite(wave).all() or len(wave) < sr // 5:
                raise ValueError(f"invalid feature input: {waveform_path}")
            # An explicit in-memory resampling step; source WAV is unchanged.
            if sr != 16000:
                divisor = int(np.gcd(sr, 16000))
                wave = resample_poly(wave[:, 0], 16000 // divisor, sr // divisor).astype("float32")
            else:
                wave = wave[:, 0]
            inputs = extractor(wave, sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                output = model(**inputs, output_hidden_states=True)
            pools = []
            for layer in [12, 24]:
                values = output.hidden_states[layer][0].cpu().numpy()
                pools.extend([values.mean(axis=0), values.std(axis=0)])
            feature = np.concatenate(pools).astype("float32")
            if feature.shape != (4096,) or not np.isfinite(feature).all():
                raise ValueError("invalid pooled speech features")
            np.savez_compressed(destination, speech=feature,
                                audio_sha256=np.array(row["sha256"]),
                                encoder_revision=np.array(revision),
                                driver_sha256=np.array(driver_hash))
        with np.load(destination, allow_pickle=False) as cached:
            if cached["speech"].shape != (4096,) or not np.isfinite(cached["speech"]).all():
                raise ValueError("corrupt feature cache")
            expected = {"audio_sha256": row["sha256"], "encoder_revision": revision, "driver_sha256": driver_hash}
            if any(key not in cached or str(cached[key].item()) != value for key, value in expected.items()):
                raise ValueError("feature cache does not match this audio/encoder/driver; preserve old manifests and use a separate output group")
        metadata.append({"clip_id": clip_id, "path": destination.name, "sha256": digest(destination),
                         "size_bytes": destination.stat().st_size, "audio_sha256": row["sha256"],
                         "speaker_id": row["speaker_id"], "sentence_id": row["sentence_id"],
                         "text_sha256": hashlib.sha256(re.sub(r"[\W_]+", "", unicodedata.normalize("NFC", row["text"])).encode()).hexdigest(),
                         "split": row["split"], "automatic_leakage_label": None})
        LOG.info("Encoded %s (%s/%s)", clip_id, len(metadata), len(items))
    record = {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
              "creator": "Codex GPT-6 /root", "model_revision": revision,
              "driver_sha256": driver_hash, "layers": [12, 24], "pooling": ["mean", "std"],
              "feature_dimension": 4096, "items": metadata, "status": "features_only_no_trained_detector"}
    (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception("Feature extraction failed; inspect any partial cache before reuse")
        raise
