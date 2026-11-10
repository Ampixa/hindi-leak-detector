"""Measure crop sensitivity of G2P evidence without selecting a favorable crop."""
from __future__ import annotations

import math

import numpy as np

from hindi_detector.ctc import validate_log_probs
from hindi_detector.g2p_evidence import word_path_evidence

# Fixed before observing the human-bank results. All recipes remain visible;
# there is no maximum-score selection or boundary optimization against a label.
WINDOWS = (
    ('baseline_context_30ms', -.03, .03),
    ('tight_word', 0., 0.),
    ('context_80ms', -.08, .08),
    ('translated_early_20ms', -.05, .01),
    ('translated_late_20ms', -.01, .05),
    ('start_early_20ms', -.05, .03),
    ('end_late_20ms', -.03, .05),
)


def audit_boundaries(log_probs: np.ndarray, hypothesis: dict, *, start: float, end: float, seconds: float) -> dict:
    values = validate_log_probs(log_probs, 0)
    if any(not math.isfinite(v) for v in (start, end, seconds)) or not 0 <= start < end <= seconds:
        raise ValueError('Word boundaries must lie within the actual positive audio duration')
    if hypothesis['status'] != 'supported_competing_g2p_paths':
        return {'status': 'unavailable_g2p_projection', 'projection_status': hypothesis['status'],
                'reason': hypothesis.get('reason'), 'windows': [], 'direction_changes': None,
                'all_windows_supported': False, 'hindi_error_probability': None}
    windows = []
    for name, left, right in WINDOWS:
        begin, finish = max(0., start + left), min(seconds, end + right)
        first = max(0, int(np.floor(begin / seconds * len(values))))
        last = min(len(values), int(np.ceil(finish / seconds * len(values))))
        evidence = word_path_evidence(values[first:last], hypothesis)
        windows.append({'name': name, 'start_seconds': begin, 'end_seconds': finish,
                        'first_frame': first, 'end_frame_exclusive': last,
                        'status': evidence['status'], 'reason': evidence.get('reason'),
                        'hindi_minus_nepali_log_likelihood': evidence['hindi_minus_nepali_log_likelihood'],
                        'nepali_minus_greedy_per_frame': evidence.get('nepali_minus_greedy_per_frame'),
                        'hindi_minus_greedy_per_frame': evidence.get('hindi_minus_greedy_per_frame')})
    scores = [w['hindi_minus_nepali_log_likelihood'] for w in windows
              if w['hindi_minus_nepali_log_likelihood'] is not None]
    changes = any(v < 0 for v in scores) and any(v > 0 for v in scores) if len(scores) >= 2 else None
    complete = len(scores) == len(WINDOWS)
    status = ('boundary_sensitive_g2p_evidence' if changes else
              'consistent_sign_in_tested_windows' if complete and all(v != 0 for v in scores) else
              'incomplete_or_tied_boundary_evidence')
    return {'status': status, 'windows': windows, 'direction_changes': changes,
            'all_windows_supported': complete, 'supported_windows': len(scores),
            'tested_windows': len(WINDOWS), 'score_min': min(scores) if scores else None,
            'score_max': max(scores) if scores else None,
            'hindi_error_probability': None,
            'meaning': 'Agreement across fixed crop perturbations is a sensitivity diagnostic, not an alignment-correctness or native-pronunciation judgment. Wider crops can include neighboring speech. No crop is chosen to maximize Hindi evidence.'}
