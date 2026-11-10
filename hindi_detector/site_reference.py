"""Reference-relative site evidence with explicit leave-speaker/audio-out rules."""
from __future__ import annotations

import math


def rank_site(score: float, distribution: dict, speaker_id: str, audio_sha256: str) -> dict:
    if not math.isfinite(score):
        raise ValueError('Nonfinite candidate site score')
    references = distribution['references']
    if len({r['speaker_id'] for r in references}) != len(references):
        raise ValueError('Each reference speaker must contribute once')
    retained = [r for r in references if r['speaker_id'] != speaker_id and audio_sha256 not in r['audio_sha256s']]
    values = [r['score'] for r in retained]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite native reference score')
    if len(values) < 4:
        return {'status': 'insufficient_independent_native_references', 'n': len(values), 'native_reference_rank': None}
    rank = (.5 + sum(v < score for v in values) + .5 * sum(v == score for v in values)) / (len(values) + 1)
    return {'status': 'reference_relative_site_evidence', 'n': len(values), 'native_reference_rank': rank,
            'reference_speaker_ids': [r['speaker_id'] for r in retained],
            'meaning': 'Higher rank means stronger support for this candidate count alternative than same-word native reference readings. Not Hindi probability or correctness.'}
