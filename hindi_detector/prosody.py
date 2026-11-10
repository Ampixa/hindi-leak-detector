"""Descriptive timing and pitch features. These do not identify an accent."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def analyse(path: Path) -> dict:
    audio, sr = sf.read(path, dtype="float64", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f"expected mono audio; explicitly downmix first: {path}")
    audio = audio[:, 0]
    if sr < 8000 or not np.isfinite(audio).all() or len(audio) < sr // 5:
        raise ValueError(f"invalid, nonfinite, or too short audio: {path}")
    if np.max(np.abs(audio)) > 1.01:
        raise ValueError(f"audio amplitude is out of range: {path}")
    size, hop = round(0.04 * sr), round(0.01 * sr)
    starts = np.arange(0, len(audio) - size + 1, hop)
    frames = np.stack([audio[s:s + size] for s in starts])
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    if np.max(rms) < 1e-5:
        raise ValueError(f"silent audio has no measurable speech prosody: {path}")
    active = rms > max(float(np.percentile(rms, 95)) * 0.18, 1e-5)
    bounds = np.flatnonzero(active)
    if len(bounds) < 4:
        raise ValueError(f"too few active frames: {path}")
    first, last = int(bounds[0]), int(bounds[-1])
    pause_lengths, start = [], None
    for i in range(first, last + 1):
        if not active[i] and start is None:
            start = i
        elif active[i] and start is not None:
            length = (i - start) * hop / sr
            if length >= 0.12:
                pause_lengths.append(length)
            start = None
    pitch = np.full(len(frames), np.nan)
    low, high = int(sr / 450), int(sr / 60)
    window = np.hanning(size)
    for i in bounds:
        frame = (frames[i] - frames[i].mean()) * window
        corr = np.correlate(frame, frame, mode="full")[size - 1:]
        if corr[0] <= 0:
            continue
        lag = low + int(np.argmax(corr[low:high]))
        if corr[lag] / corr[0] >= 0.45:
            pitch[i] = sr / lag
    voiced = np.isfinite(pitch)
    warnings = ["Energy threshold is a pause proxy, not a speech/phone aligner.",
                "Autocorrelation pitch is provisional and can contain octave errors."]
    result = {
        "audio_path": str(path.resolve()), "sample_rate": sr, "duration_seconds": len(audio) / sr,
        "active_fraction": float(active.mean()), "interior_pause_count": len(pause_lengths),
        "interior_pause_seconds": float(sum(pause_lengths)), "voiced_fraction": float(voiced.mean()),
        "f0_median_hz": None, "f0_range_semitones": None,
        "f0_slope_semitones_per_second": None, "tail_offset_semitones": None,
        "status": "descriptive_features_only", "warnings": warnings,
    }
    if voiced.sum() < 8:
        warnings.append("Too few periodic frames; pitch features unavailable.")
        return result
    reference = float(np.median(pitch[voiced]))
    semitones = 12 * np.log2(pitch / reference)
    times = starts / sr
    # Preserve original frame times instead of compressing unvoiced intervals.
    tail = voiced & (times >= times[last] - 0.25) & (times <= times[last])
    result.update({
        "f0_median_hz": reference,
        "f0_range_semitones": float(np.percentile(semitones[voiced], 95) - np.percentile(semitones[voiced], 5)),
        "f0_slope_semitones_per_second": float(np.polyfit(times[voiced], semitones[voiced], 1)[0]),
        "tail_offset_semitones": float(np.median(semitones[tail])) if tail.sum() >= 4 else None,
    })
    return result
