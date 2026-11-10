"""Boundary-aware component inference and descriptive reference comparisons."""
from __future__ import annotations

import math

import numpy as np

from hindi_detector.boundary_audit import WINDOWS
from hindi_detector.evaluation import predict_linear
from hindi_detector.interventions import intervention_features
from hindi_detector.schwa_probe import compare_site, site_paths


def crop_windows(frame_count: int, rate: int, start: float, end: float) -> list[dict]:
    if type(frame_count) is not int or type(rate) is not int or frame_count < 1 or rate < 1:
        raise ValueError('Need positive actual waveform frame count and integer rate')
    seconds = frame_count / rate
    if any(not math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= seconds + 1e-6:
        raise ValueError('Word times lie outside the actual waveform')
    result = []
    for name, left, right in WINDOWS:
        first = max(0, int(np.floor(max(0., start + left) * rate)))
        last = min(frame_count, int(np.ceil(min(seconds, end + right) * rate)))
        duration = (last - first) / rate
        result.append({'name': name, 'source_first_frame': first, 'source_end_frame_exclusive': last,
                       'start_seconds': first / rate, 'end_seconds': last / rate,
                       'within_component_duration_contract': .2 <= duration <= 15})
    return result


def summarize_component_windows(windows: list[dict], branch: str) -> dict:
    valid = [w['request_scores'][branch] for w in windows if w['request_scores'] is not None]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in valid):
        raise ValueError('Component produced an invalid request score')
    complete = len(valid) == len(WINDOWS)
    changing = any(v < .5 for v in valid) and any(v >= .5 for v in valid) if len(valid) >= 2 else None
    baseline = windows[0]['request_scores'][branch] if windows and windows[0]['request_scores'] is not None else None
    return {'baseline_request_score': baseline, 'supported_windows': len(valid), 'planned_windows': len(WINDOWS),
            'all_windows_supported': complete, 'request_class_changes': changing,
            'minimum_request_score': min(valid) if valid else None, 'maximum_request_score': max(valid) if valid else None,
            'status': 'boundary_sensitive_request_class' if changing else
                      'same_request_class_all_windows' if complete else 'partial_or_unavailable_boundary_coverage',
            'actual_hindi_error_probability': None}


def score_real_word(engine, models: dict, wave: np.ndarray, rate: int, word: dict, candidate: dict, ipa: dict) -> dict:
    from scipy.signal import resample_poly

    if wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError('Real-word scoring needs the verified finite mono waveform')
    recipes = crop_windows(len(wave), rate, word['start'], word['end'])
    try:
        paths = site_paths(candidate, ipa, engine.vocab)
        if len(paths) != 1 or paths[0]['lexicon_schwa_count'] != 1 or paths[0]['alternative_schwa_count'] != 0:
            raise ValueError('Frozen components support exactly one one-to-zero site')
    except ValueError as error:
        return {'status': 'unsupported_lexical_hypothesis', 'reason': str(error), 'windows': [],
                'branches': {name: summarize_component_windows([], name) for name in models},
                'actual_hindi_error_probability': None}
    windows = []
    for recipe in recipes:
        result = {**recipe, 'request_scores': None, 'site_log_likelihood_ratio': None, 'reason': None}
        if not recipe['within_component_duration_contract']:
            result.update(status='crop_outside_component_duration_contract', reason='Independent crop is shorter than .2 or longer than15 seconds; no padding or replacement score.')
            windows.append(result)
            continue
        crop = wave[recipe['source_first_frame']:recipe['source_end_frame_exclusive']]
        if np.max(np.abs(crop)) < 1e-5:
            result.update(status='silent_crop_unavailable', reason='No non-silent word audio in this crop.')
            windows.append(result)
            continue
        divisor = int(np.gcd(rate, 16000))
        samples = resample_poly(crop, 16000 // divisor, rate // divisor).astype('float32') if rate != 16000 else crop
        with engine.torch.inference_mode():
            logits = engine.phone(**engine.extractor(samples, sampling_rate=16000, return_tensors='pt')).logits[0]
            if not engine.torch.isfinite(logits).all():
                raise RuntimeError('Nonfinite phone encoder output; no phonetic abstention substituted')
            log_probs = logits.log_softmax(-1).cpu().numpy()
        try:
            site = compare_site(log_probs, paths[0])
        except ValueError as error:
            result.update(status='impossible_lexical_path_for_crop', reason=str(error))
            windows.append(result)
            continue
        # Model/feature errors propagate instead of becoming phonetic abstentions.
        vectors = intervention_features(log_probs, site, candidate, engine.vocab, engine.special)
        scores = {name: float(predict_linear(model, vectors[name][None])[0]) for name, model in models.items()}
        result.update(status='unreviewed_independently_encoded_word_evidence', request_scores=scores,
                      site_log_likelihood_ratio=site['marginal_alternative_minus_lexicon'], posterior_frames=len(log_probs))
        windows.append(result)
    return {'status': 'independent_word_windows_measured_not_hindi_labels', 'windows': windows,
            'branches': {name: summarize_component_windows(windows, name) for name in models},
            'inference_recipe': 'Crop the original mono waveform at its native sample rate, resample each crop, and run the phone encoder independently, matching isolated-word component training.',
            'actual_hindi_error_probability': None}


def prosody_difference(query: dict, reference: dict, *, minimum_voiced_fraction: float = .5) -> dict:
    fractions = [query['voiced_fraction'], reference['voiced_fraction']]
    if not 0 < minimum_voiced_fraction <= 1 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in fractions):
        raise ValueError('Prosody comparison needs valid measured voicing coverage')
    result = {'relative_word_duration_delta': query['duration_relative'] - reference['duration_relative'],
              'relative_pause_after_delta': query['pause_after_relative'] - reference['pause_after_relative'],
              'pitch_shape_rmse_st': None, 'relative_pitch_mean_delta_st': None,
              'pitch_change_delta_st': None, 'actual_hindi_intonation_error': None,
              'query_voiced_fraction': fractions[0], 'reference_voiced_fraction': fractions[1],
              'minimum_voiced_fraction': minimum_voiced_fraction,
              'pitch_comparison_status': 'insufficient_pitch_or_voiced_coverage'}
    if query['pitch_available'] and reference['pitch_available'] and min(fractions) >= minimum_voiced_fraction:
        q = np.array([query[f'pitch_bin_{i}'] for i in range(8)], dtype=float)
        r = np.array([reference[f'pitch_bin_{i}'] for i in range(8)], dtype=float)
        result.update(pitch_shape_rmse_st=float(np.sqrt(np.mean(((q - q.mean()) - (r - r.mean())) ** 2))),
                      relative_pitch_mean_delta_st=query['pitch_mean_st'] - reference['pitch_mean_st'],
                      pitch_change_delta_st=query['pitch_change_st'] - reference['pitch_change_st'],
                      pitch_comparison_status='measured_shape_not_error_label')
    if any(not math.isfinite(v) for v in result.values() if isinstance(v, (int, float))):
        raise ValueError('Nonfinite query/reference prosody')
    return result


def human_reference_rank(query: dict, references: list[dict], *, exclude_speaker_id: str | None, minimum_groups: int = 6) -> dict:
    by_speaker = {}
    for row in references:
        if row['speaker_id'] == exclude_speaker_id:
            continue
        if row['all_windows_supported'] and row['baseline_request_score'] is not None:
            by_speaker.setdefault(row['speaker_id'], []).append(row)
    count = len(by_speaker)
    if not query['all_windows_supported'] or query['baseline_request_score'] is None or count < minimum_groups:
        return {'status': 'insufficient_human_or_boundary_coverage', 'source_groups': count,
                'required_groups': minimum_groups, 'native_acceptability': None, 'hindi_error_probability': None}
    medians = [float(np.median([v['baseline_request_score'] for v in values])) for values in by_speaker.values()]
    lows = [float(np.median([v['minimum_request_score'] for v in values])) for values in by_speaker.values()]
    highs = [float(np.median([v['maximum_request_score'] for v in values])) for values in by_speaker.values()]
    return {'status': 'descriptive_human_reference_rank_only', 'source_groups': count,
            'query_baseline_request_score': query['baseline_request_score'],
            'human_baseline_median': float(np.median(medians)),
            'human_window_envelope_min': min(lows), 'human_window_envelope_max': max(highs),
            'smoothed_fraction_human_baselines_at_or_above_query': (1 + sum(v >= query['baseline_request_score'] for v in medians)) / (count + 1),
            'query_all_window_scores_above_observed_human_envelope': query['minimum_request_score'] > max(highs),
            'native_acceptability': None, 'hindi_error_probability': None,
            'meaning': 'Speaker-weighted observed score rank, not a calibrated conformal p-value or native-acceptability judgment. Lexical sense, discourse and regional coverage remain limited.'}
