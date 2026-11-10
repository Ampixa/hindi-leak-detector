"""Transcript quality measurements; these never assign pronunciation labels."""
from __future__ import annotations

import math
import unicodedata


def normalize(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError('ASR/reference transcript must be an actual string')
    return ' '.join(''.join(c if unicodedata.category(c)[0] in {'L', 'M', 'N'} else ' '
                           for c in unicodedata.normalize('NFC', text).casefold()).split())


def edit_distance(reference, hypothesis) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def transcript_metrics(reference: str, hypothesis: str) -> dict:
    r, h = normalize(reference), normalize(hypothesis)
    words = r.split(); target_chars = r.replace(' ', ''); observed_chars = h.replace(' ', '')
    if not words or not target_chars:
        raise ValueError('Nonempty reference is required; silence is not a zero-error transcript')
    word_errors = edit_distance(words, h.split()); character_errors = edit_distance(target_chars, observed_chars)
    result = {'normalized_reference': r, 'normalized_hypothesis': h, 'reference_words': len(words),
              'word_errors': word_errors, 'wer': word_errors / len(words), 'reference_characters': len(target_chars),
              'character_errors': character_errors, 'cer': character_errors / len(target_chars),
              'actual_hindi_error_label': None}
    if not math.isfinite(result['wer']) or not math.isfinite(result['cer']):
        raise ValueError('Invalid transcript quality measurement')
    return result
