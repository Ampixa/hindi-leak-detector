"""Optional pinned phoneme recognizer; accepts explicit reviewed token IDs only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_REVISION = "2c733782da5604684829819a5eb744c193fe9398"


def posterior(path: Path, model_directory: Path) -> tuple[np.ndarray, dict, int]:
    # Optional dependencies are loaded only for this command; errors propagate.
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForCTC
    import json

    directory = model_directory.resolve(strict=True)
    vocab = json.loads((directory / "vocab.json").read_text())
    waveform, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 16000 or waveform.shape[1] != 1:
        raise ValueError("phoneme recognizer requires explicitly prepared 16 kHz mono WAV")
    audio = waveform[:, 0]
    if not np.isfinite(audio).all() or len(audio) < 3200 or len(audio) > 30 * sr:
        raise ValueError("crop must be finite and between 0.2 and 30 seconds")
    if float(np.max(np.abs(audio))) < 1e-5 or float(np.max(np.abs(audio))) > 1.01:
        raise ValueError("silent or out-of-range audio")
    torch.set_num_threads(2)
    extractor = AutoFeatureExtractor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCTC.from_pretrained(directory, local_files_only=True, trust_remote_code=False).eval()
    if model.config.vocab_size != len(vocab) or set(vocab.values()) != set(range(len(vocab))):
        raise ValueError("model/vocabulary mismatch")
    inputs = extractor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
    return torch.log_softmax(logits, -1).cpu().numpy(), vocab, model.config.pad_token_id
