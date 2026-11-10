"""Check one actual source payload against the inspected native staging recipe."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import sphn

from hindi_detector.wave_identity import comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--staged', type=Path, required=True)
    args = parser.parse_args()
    original, rate = sf.read(args.original, dtype='float32', always_2d=True)
    if original.shape[1] not in {1, 2} or not np.isfinite(original).all():
        raise ValueError('Original channel structure differs from inspected mono/stereo payloads')
    channel_correlation = float(np.corrcoef(original.T)[0, 1]) if original.shape[1] == 2 else None
    if channel_correlation is not None and not np.isfinite(channel_correlation):
        raise ValueError('Stereo channel comparison is nonfinite')
    decoded = original.mean(axis=1)
    if rate != 24000:
        decoded = sphn.resample(decoded, rate, 24000)
    if decoded.ndim != 1 or not np.isfinite(decoded).all() or np.max(np.abs(decoded)) == 0:
        raise ValueError('Invalid actual staging-recipe output')
    predicted = decoded * (.95 / float(np.abs(decoded).max()))
    buffer = io.BytesIO()
    sf.write(buffer, predicted, 24000, format='WAV', subtype='PCM_16')
    expected = hashlib.sha256(args.staged.read_bytes()).hexdigest()
    reconstructed = hashlib.sha256(buffer.getvalue()).hexdigest()
    staged, staged_rate = sf.read(args.staged, dtype='float32')
    package = Path(sphn.__file__).parent
    resources = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(package.rglob('*')) if p.is_file() and p.suffix in {'.py', '.so', '.dylib', '.pyi'}]
    print(json.dumps({'source_channels': original.shape[1], 'source_sample_rate': rate,
                      'source_channel_correlation': channel_correlation,
                      'channel_review_flag': channel_correlation is not None and channel_correlation < .999,
                      'channel_flag_meaning': 'Low channel agreement needs inspection before using a downmix as a prosody reference. Forensic reconstruction still checks the documented historical mean-channel operation; original channels remain unchanged.',
                      'explicit_channel_operation': 'arithmetic mean, matching the inspected staging recipe',
                      'reproduced_staged_sha256': reconstructed, 'expected_staged_sha256': expected,
                      'encoded_staged_bytes_identical': reconstructed == expected,
                      'comparison': comparison(predicted, 24000, staged, staged_rate),
                      'sphn_version': importlib.metadata.version('sphn'), 'sphn_resources': resources,
                      'numpy_version': np.__version__, 'soundfile_version': sf.__version__,
                      'libsndfile_version': sf.__libsndfile_version__,
                      'historical_executed_revision_known': False}, allow_nan=False))


if __name__ == '__main__':
    main()
