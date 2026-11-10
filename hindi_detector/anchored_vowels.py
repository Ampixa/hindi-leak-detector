"""Reference-free internal-vowel evidence inside fixed consonant anchors.

The deletion path locates both anchors once per boundary recipe. Vowel and
blank-only hypotheses then use the IDENTICAL intervening frames; unrelated
vowels cannot improve that local likelihood. No word length, lexical identity,
speaker identity, source identity or reference-derived feature is emitted.
"""
from __future__ import annotations

import math

import numpy as np

from hindi_detector.exact_phone_ctc import path_score

FEATURE_NAMES = ['gap_deletion_margin_mean', 'gap_deletion_margin_min', 'gap_deletion_margin_max',
    'gap_vowel_mass_mean', 'gap_vowel_peak_mean', 'focal_vowel_log_emission_mean',
    'focal_ownership_mean', 'left_anchor_log_emission_mean', 'right_anchor_log_emission_mean',
    'gap_native_fit_mean', 'gap_blank_fit_mean', 'gap_available_fraction']
FLOOR = -30.


def _paths(compiled, variant, contrast, retained):
    at = retained.index(contrast['index'])
    native = [variant['units'][i] for i in retained]
    native[at] = compiled['any_vowel']
    return native, native[:at] + native[at + 1:], at


def measure(compiled, variant, contrast, crops):
    if contrast['kind'] != 'vowel_deletion' or len(crops) != 3:
        raise ValueError('Anchored evidence is only for deletion at three fixed word bounds')
    index, units, vowels = contrast['index'], variant['units'], compiled['any_vowel']
    if not 0 < index < len(units) - 1 or units[index - 1] & vowels or units[index + 1] & vowels:
        raise ValueError('Internal focal vowel requires two nonsyllabic text-level neighbors')
    retained = list(range(len(units)))
    removable = {i for i, phone in enumerate(variant['phones']) if i != index and phone in {'ax', 'h'}}
    def fit(indices):
        a, b, _ = _paths(compiled, variant, contrast, indices)
        return max(path_score(crops[1][2], a), path_score(crops[1][2], b))
    current = fit(retained)
    while removable:
        trials = []
        for at in sorted(removable):
            proposal = [i for i in retained if i != at]
            # Never remove either immediate consonant anchor around the focal
            # site, even when it happens to be optional /h/ in other positions.
            focal = proposal.index(index)
            if not 0 < focal < len(proposal) - 1 or proposal[focal - 1] != index - 1 or proposal[focal + 1] != index + 1:
                continue
            trials.append((fit(proposal), at, proposal))
        if not trials:
            break
        value, at, proposal = max(trials, key=lambda row: row[0])
        if value <= current + .01:
            break
        retained, current = proposal, value
        removable.remove(at)
    native, alternate, focal = _paths(compiled, variant, contrast, retained)
    measurements = []
    for padding, start_frame, values in crops:
        deletion = path_score(values, alternate, trace=True)
        left, right = deletion['spans'][focal - 1], deletion['spans'][focal]
        start, end = left['end_frame'], right['start_frame']
        if not 0 <= start <= end <= len(values):
            raise ValueError('Consonant anchors do not define an ordered local gap')
        gap = values[start:end]
        mass = peak = 0.
        if len(gap):
            native_score = path_score(gap, [vowels])
            blank_score = path_score(gap, [])
            if not math.isfinite(blank_score):
                raise ValueError('Nonfinite blank-only local hypothesis')
            if not math.isfinite(native_score):
                native_score = FLOOR * len(gap)
            p = np.exp(gap)
            vowel = p[:, sorted(vowels)].sum(1)
            nonblank = p[:, 4:].sum()
            mass = float(vowel.sum() / max(1e-12, nonblank))
            peak = float(vowel.max())
            native_fit, blank_fit = native_score / len(gap), blank_score / len(gap)
        else:
            # A zero-frame observed gap is evidence, not a measurement failure:
            # the two consonant peaks are adjacent. Represent the impossible
            # intervening-vowel hypothesis explicitly with the fixed censor.
            native_fit, blank_fit = FLOOR, 0.
        focal_log, ownership, span = FLOOR, 0., None
        try:
            intact = path_score(values, native, trace=True)
            span = intact['spans'][focal]
            focal_log = span['mean_log_emission']
            overlap = max(0, min(end, span['end_frame']) - max(start, span['start_frame']))
            ownership = overlap / max(1, span['end_frame'] - span['start_frame'])
        except ValueError as error:
            # No intact path is an explicit observed hypothesis outcome, not an
            # omitted positive. The deletion path above must still be feasible.
            if 'No feasible exact CTC path' not in str(error):
                raise
        measurements.append({'padding_seconds': padding, 'start_frame': start_frame,
            'gap_start_frame': start_frame + start, 'gap_end_frame': start_frame + end,
            'gap_deletion_margin': float(np.clip(blank_fit - native_fit, -10., 10.)), 'gap_vowel_mass': mass, 'gap_vowel_peak': peak,
            'native_focal_log_emission': focal_log, 'focal_ownership': ownership,
            'left_anchor': left, 'right_anchor': right, 'native_focal_span': span,
            'gap_native_fit': native_fit, 'gap_blank_fit': blank_fit, 'gap_available': bool(len(gap))})
    def mean(name):
        return float(np.mean([row[name] for row in measurements]))
    margins = [row['gap_deletion_margin'] for row in measurements]
    features = np.array([np.mean(margins), min(margins), max(margins), mean('gap_vowel_mass'), mean('gap_vowel_peak'),
        mean('native_focal_log_emission'), mean('focal_ownership'),
        np.mean([row['left_anchor']['mean_log_emission'] for row in measurements]),
        np.mean([row['right_anchor']['mean_log_emission'] for row in measurements]),
        mean('gap_native_fit'), mean('gap_blank_fit'), mean('gap_available')], np.float32)
    if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
        raise ValueError('Invalid anchored-vowel evidence features')
    return features, {'windows': measurements, 'retained_native_indices': retained,
                      'native_target_phones': variant['phones'], 'no_reference_audio_used': True,
                      'likelihood': 'Exact allowed-token CTC inside identical fixed deletion-anchor gaps',
                      'margin_clip': [-10., 10.]}
