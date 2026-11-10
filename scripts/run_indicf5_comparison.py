"""Strictly load the inspected IndicF5 bundle and synthesize the matched pack."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from vocos import Vocos

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/indicf5"
PACK = ROOT / "data/indicf5_audio8_v1"
LOG = logging.getLogger(__name__)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    pack = args.pack.resolve(strict=True)
    selection = json.loads((pack / "selection.json").read_text())
    jobs = selection["jobs"]
    if not isinstance(jobs, list) or not jobs or len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("selection must contain unique nonempty jobs")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", j["id"]) for j in jobs):
        raise ValueError("job identifiers cannot contain path separators")
    limit = len(jobs) if args.limit is None else args.limit
    if not 1 <= limit <= len(jobs):
        raise ValueError(f"limit must be between 1 and {len(jobs)}")
    output = pack / "indicf5"
    if not (output / "README.md").is_file():
        raise ValueError("synthesis provenance README is required before writing outputs")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS explicitly requested but unavailable")
    torch.set_num_threads(2)
    inventory = json.loads((MODEL / "inventory.json").read_text())
    if inventory["revision"] != "fb652a96e508091045890dae5bb99fe511facbcf":
        raise ValueError("unexpected model revision")
    for entry in inventory["files"]:
        path = MODEL / entry["path"]
        if path.stat().st_size != entry["size_bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"model file verification failed: {path}")
    sys.path.insert(0, str(MODEL))
    # Import only inspected architecture and inference modules. The HF wrapper
    # uses unpinned downloads/compile and is not executed by this adapter.
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer, seed_everything
    from f5_tts.infer.utils_infer import infer_batch_process

    vocab, size = get_tokenizer(str(MODEL / "checkpoints/vocab.txt"), "custom")
    model = CFM(
        transformer=DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
                        conv_layers=4, text_num_embeds=size, mel_dim=100),
        mel_spec_kwargs={"n_fft": 1024, "hop_length": 256, "win_length": 1024,
                         "n_mel_channels": 100, "target_sample_rate": 24000,
                         "mel_spec_type": "vocos"},
        odeint_kwargs={"method": "euler"}, vocab_char_map=vocab,
    )
    vocoder = Vocos.from_hparams(str(MODEL / "vocos/config.yaml"))
    state = load_file(str(MODEL / "model.safetensors"), device="cpu")
    generator_keys, vocoder_keys = "ema_model._orig_mod.", "vocoder._orig_mod."
    unknown = [k for k in state if not k.startswith((generator_keys, vocoder_keys))]
    if unknown:
        raise ValueError(f"unknown state prefixes: {unknown[:5]}")
    model.load_state_dict({k.removeprefix(generator_keys): v for k, v in state.items() if k.startswith(generator_keys)}, strict=True)
    vocoder.load_state_dict({k.removeprefix(vocoder_keys): v for k, v in state.items() if k.startswith(vocoder_keys)}, strict=True)
    del state
    model = model.eval().to(args.device)
    vocoder = vocoder.eval().to(args.device)
    receipt = output / "items.jsonl"
    existing = {}
    if receipt.exists():
        rows = [json.loads(line) for line in receipt.read_text().splitlines()]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate IndicF5 receipts")
        existing = {r["id"]: r for r in rows}
    config = {"schema_version": 1, "created_by": "Codex GPT-6 /root",
              "created_utc": datetime.now(timezone.utc).isoformat(), "device": args.device,
              "model_revision": inventory["revision"], "weights_sha256": digest(MODEL / "model.safetensors"),
              "driver_sha256": digest(Path(__file__)), "torch_version": torch.__version__,
              "nfe_step": 32, "cfg_strength": 2.0, "sway_sampling_coef": -1.0,
              "speed": 1.0, "target_rms": 0.1, "sample_rate": 24000, "seed": 0,
              "waveform_policy": "No generated-audio silence stripping, loudness normalization or ASR reranking",
              "reference_policy": "Exact audio8 reference file/transcript; native F5 RMS and resampling inside infer_batch_process",
              "model_load": "Strict generator/vocoder state loading with inspected compiled-prefix removal",
              "selection_sha256": digest(pack / "selection.json"), "status": "running"}
    config_path = output / "run.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in ["model_revision", "weights_sha256", "driver_sha256", "device", "selection_sha256"]:
            if old[key] != config[key]:
                raise ValueError(f"resume configuration changed: {key}")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    for job in jobs[:limit]:
        if job["id"] in existing:
            old = existing[job["id"]]
            if digest(pack / old["audio"]) != old["sha256"]:
                raise ValueError("existing output checksum mismatch")
            continue
        reference = (pack / job["reference_audio"]).resolve(strict=True)
        if digest(reference) != job["reference_sha256"]:
            raise ValueError(f"reference checksum mismatch: {reference}")
        wave, sr = sf.read(reference, dtype="float32", always_2d=True)
        if wave.shape[1] != 1 or not np.isfinite(wave).all() or float(np.max(np.abs(wave))) < 1e-5:
            raise ValueError("reference must be finite non-silent mono audio")
        seed_everything(0)
        start = time.monotonic()
        audio, rate, _ = infer_batch_process(
            (torch.from_numpy(wave.T), sr), job["reference_text"], [job["text"]],
            model, vocoder, mel_spec_type="vocos", target_rms=0.1, cross_fade_duration=0.15,
            nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0, speed=1.0,
            fix_duration=None, device=args.device,
        )
        if rate != 24000 or audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError("invalid IndicF5 waveform")
        path = output / f"{job['id']}.wav"
        if path.exists():
            raise FileExistsError(f"unreceipted output requires inspection: {path}")
        sf.write(path, audio, rate, subtype="PCM_16")
        row = {"id": job["id"], "speaker_id": job["speaker_id"], "text": job["text"],
               "audio": str(path.relative_to(pack)), "sha256": digest(path),
               "seconds": len(audio) / rate, "wall_seconds": time.monotonic() - start,
               "status": "ok_unreviewed", "reference_sha256": job["reference_sha256"],
               "seed": 0, "automatic_leakage_label": None}
        with receipt.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        existing[job["id"]] = row
        LOG.info("Generated %s (%0.2f s), total %s/%s", job["id"], row["seconds"], len(existing), len(jobs))
    config["status"] = "complete_unreviewed_audio" if len(existing) == len(jobs) else "partial_unreviewed_audio"
    config["completed"] = len(existing)
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception("IndicF5 comparison failed; inspect traceback and preserve incomplete state")
        raise
