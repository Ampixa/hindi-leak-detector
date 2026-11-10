"""Check the exact source context required by the physical control recipe."""
from __future__ import annotations


def boundary_issues(wave_lengths, native_words, donor_words, focal, sample_rate=16000):
    if len(wave_lengths) != 2 or not focal:
        raise ValueError('Two wave lengths and nonempty focal indices required')
    if [w['word'] for w in native_words] != [w['word'] for w in donor_words]:
        raise ValueError('Matched word sequences differ')
    issues = []
    for index in focal:
        n = native_words[index]
        start, end = int(n['start'] * sample_rate), int(n['end'] * sample_rate)
        for label, words in enumerate((native_words, donor_words)):
            h = words[index]
            left, right = int(h['start'] * sample_rate), int(h['end'] * sample_rate)
            if not 0 <= start < end <= wave_lengths[0] or not 0 <= left < right <= wave_lengths[label]:
                raise ValueError('Invalid measured word boundary')
            width = min(80, (right-left)//4, start, wave_lengths[0]-end)
            if width < 2:
                issues.append({'word_index': index, 'word': n['word'], 'source_label': label,
                               'fade_samples_available': width, 'native_start_sample': start,
                               'native_end_sample': end, 'native_samples': wave_lengths[0]})
    return issues
