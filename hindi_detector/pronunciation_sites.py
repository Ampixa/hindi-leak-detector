"""Local competing vowel hypotheses compiled from actual dual-G2P output."""
from __future__ import annotations

import math

import numpy as np

from hindi_detector.phone_lattice import VOWELS, path_score, represent, units_for_project


def align_units(native, hindi):
    """Minimum edits with shared acoustic representations treated as matches."""
    cost = np.full((len(native)+1, len(hindi)+1), np.inf)
    parent = {}
    cost[0, 0] = 0
    for i in range(len(native)+1):
        for j in range(len(hindi)+1):
            options = []
            if i and j:
                options.append((cost[i-1, j-1] + (0 if native[i-1] & hindi[j-1] else 1.25), i-1, j-1, (i-1, j-1)))
            if i:
                options.append((cost[i-1, j]+1, i-1, j, (i-1, None)))
            if j:
                options.append((cost[i, j-1]+1, i, j-1, (None, j-1)))
            if options:
                value, a, b, pair = min(options, key=lambda v: v[0])
                cost[i, j] = value
                parent[i, j] = (a, b, pair)
    out = []
    i, j = len(native), len(hindi)
    while i or j:
        i, j, pair = parent[i, j]
        out.append(pair)
    return list(reversed(out))


def compile_word(row, ipa, vocabulary):
    all_vowels = frozenset(vocabulary[t] for t in VOWELS if t in vocabulary)
    hindi = [represent(t, vocabulary) for t in row['hindi_units']]
    variants = []
    for variant_id, phones in enumerate(row['nepali_project_variants']):
        phones = [p for p in phones if p != '.']
        units, projections = units_for_project(phones, ipa, vocabulary)
        contrasts = []
        for native_index, hindi_index in align_units(units, hindi):
            if native_index is None:
                # Insertion needs a separate between-phone anchor recipe.
                continue
            if hindi_index is None:
                if phones[native_index] == 'ax':
                    contrasts.append({'kind': 'vowel_deletion', 'index': native_index,
                        'expected': projections[native_index]['ipa'], 'alternative': None})
                continue
            a, b = units[native_index], hindi[hindi_index]
            if not a & b and a <= all_vowels and b <= all_vowels:
                contrasts.append({'kind': 'vowel_quality', 'index': native_index,
                    'expected': projections[native_index]['ipa'], 'alternative': row['hindi_units'][hindi_index],
                    'alternative_unit': b})
        variants.append({'variant_id': variant_id, 'phones': phones, 'units': units,
                         'projections': projections, 'contrasts': contrasts})
    return {'word': row['word'], 'variants': variants, 'any_vowel': all_vowels,
            'hindi_units': row['hindi_units'], 'hypotheses_are_realization_gold': False}


def windows(item, ordinal, paddings=(0., .03, .06)):
    word = item['detail']['words'][ordinal]
    frame_seconds = item['detail']['phone_frame_seconds']
    values = item['phone_log_probs']
    result = []
    for pad in paddings:
        start = max(0, int(np.floor((word['start']-pad)/frame_seconds)))
        end = min(len(values), int(np.ceil((word['end']+pad)/frame_seconds)))
        if not start < end:
            raise ValueError('Invalid measured phone crop')
        result.append((pad, start, values[start:end]))
    return result


def select_reference_structure(compiled, reference_windows):
    candidates = []
    for variant in compiled['variants']:
        values = [path_score(x, variant['units'])/len(x) for _, _, x in reference_windows]
        candidates.append((float(np.median(values)), variant))
    _, chosen = max(candidates, key=lambda v: v[0])
    if not math.isfinite(max(v[0] for v in candidates)):
        raise ValueError('No representable native variant in reference frames')
    # Reference-supported reduction is background context, not an error label.
    broad = [compiled['any_vowel'] if p == 'ax' else u for p, u in zip(chosen['phones'], chosen['units'])]
    reduced = []
    for index, phone in enumerate(chosen['phones']):
        if phone != 'ax':
            continue
        alternative = broad[:index] + broad[index+1:]
        if not alternative:
            continue
        margins = [(path_score(x, alternative)-path_score(x, broad))/len(x) for _, _, x in reference_windows]
        if np.median(margins) > 0:
            reduced.append(index)
    return chosen, reduced


def evidence_for_site(compiled, variant, reduced_context, contrast, crops):
    index = contrast['index']
    retained = [i for i in range(len(variant['units'])) if i not in reduced_context or i == index]
    target = [variant['units'][i] for i in retained]
    at = retained.index(index)
    if contrast['kind'] == 'vowel_deletion':
        target[at] = compiled['any_vowel']
        alternate = target[:at] + target[at+1:]
    else:
        alternate = target.copy()
        alternate[at] = contrast['alternative_unit']
    if not alternate:
        raise ValueError('Empty site alternative')
    evidence = []
    for pad, start, values in crops:
        a, b = path_score(values, target), path_score(values, alternate)
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Site hypotheses cannot both fit measured word frames')
        evidence.append({'padding_seconds': pad, 'frames': len(values), 'start_frame': start,
            'target_score_per_frame': a/len(values), 'alternative_score_per_frame': b/len(values),
            'alternative_minus_target_per_frame': (b-a)/len(values)})
    trace = path_score(crops[1][2], target, trace=True)
    span = trace['spans'][at]
    return {'windows': evidence, 'target_phone_span': {**span,
        'absolute_start_frame': crops[1][1]+span['start_frame'],
        'absolute_end_frame': crops[1][1]+span['end_frame']},
        'reference_supported_context_reductions': reduced_context,
        'retained_native_indices': retained}


def paired_site_features(query_evidence, reference_evidence, contrast):
    q = np.array([r['alternative_minus_target_per_frame'] for r in query_evidence['windows']])
    r = np.array([r['alternative_minus_target_per_frame'] for r in reference_evidence['windows']])
    a = np.array([r['target_score_per_frame'] for r in query_evidence['windows']])
    b = np.array([r['alternative_score_per_frame'] for r in query_evidence['windows']])
    values = [q.mean(), np.median(q), q.min(), q.max(), q.std(), r.mean(), r.min(), r.max(),
              (q-r).mean(), (q-r).min(), (q-r).max(), (q-r).std(), a.mean(), b.mean(),
              float(np.mean(q>0)), float(np.mean(q>r)),
              float(contrast['kind']=='vowel_deletion'), float(contrast['kind']=='vowel_quality')]
    result = np.array(values, np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite measured site features')
    return result


FEATURE_NAMES = ['query_margin_mean', 'query_margin_median', 'query_margin_min', 'query_margin_max',
    'query_margin_std', 'reference_margin_mean', 'reference_margin_min', 'reference_margin_max',
    'relative_margin_mean', 'relative_margin_min', 'relative_margin_max', 'relative_margin_std',
    'query_target_fit', 'query_alternative_fit', 'positive_margin_fraction', 'above_reference_fraction',
    'is_vowel_deletion', 'is_vowel_quality']
