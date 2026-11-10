"""Exact time-coordinate mapping through an explicitly recorded waveform edit."""
from __future__ import annotations

import math


def map_sample(position,start,end,replacement_samples):
    if not all(math.isfinite(v) for v in (position,start,end,replacement_samples)) or not 0<=start<end or replacement_samples<0:
        raise ValueError('Invalid waveform edit coordinates')
    if position<=start:return float(position)
    if position>=end:return float(position+replacement_samples-(end-start))
    return float(start+(position-start)*replacement_samples/(end-start))


def map_word(word,start,end,replacement_samples,rate=16000):
    if rate<=0:raise ValueError('Invalid sample rate')
    result={**word,'start':map_sample(word['start']*rate,start,end,replacement_samples)/rate,
        'end':map_sample(word['end']*rate,start,end,replacement_samples)/rate}
    if not 0<=result['start']<result['end']:raise ValueError('Edit removes the entire aligned word')
    return result
