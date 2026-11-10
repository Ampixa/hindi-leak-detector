"""Check that local acoustic anchors actually support a vowel diagnosis."""
from __future__ import annotations

from hindi_detector.phone_lattice import path_score

POLICY = {
    'version': 'local_vowel_support_v21',
    'minimum_mean_log_emission': -1.5,
    'origin': 'Same predeclared nucleus-anchor floor used by the physical control selection; no word-specific threshold',
    'meaning': 'Expected vowel observed in reference; alternative vowel or surviving consonant anchors observed in query',
}


def validate_site(compiled, variant, contrast, reference_evidence, query_crops, reference_crops):
    retained = reference_evidence['retained_native_indices']
    at = retained.index(contrast['index'])
    target = [variant['units'][i] for i in retained]
    if contrast['kind'] == 'vowel_deletion':
        target[at] = compiled['any_vowel']
        alternate = target[:at] + target[at+1:]
    else:
        alternate = target.copy()
        alternate[at] = contrast['alternative_unit']
    reference = path_score(reference_crops[1][2], target, trace=True)
    query = path_score(query_crops[1][2], alternate, trace=True)
    floor = POLICY['minimum_mean_log_emission']
    checks = [{'role': 'reference_expected_vowel', 'index': at,
               'mean_log_emission': reference['spans'][at]['mean_log_emission']}]
    if contrast['kind'] == 'vowel_quality':
        checks.append({'role': 'query_alternative_vowel', 'index': at,
                       'mean_log_emission': query['spans'][at]['mean_log_emission']})
        location = query['spans'][at]
        start, end = location['start_frame'], location['end_frame']
    else:
        anchors = [i for i in (at-1, at+1) if 0 <= i < len(target) and not target[i] <= compiled['any_vowel']]
        if not anchors:
            return {'status':'unsupported_local_anchors','checks':checks,'reason':'No adjacent consonant anchor for deletion'}
        for i in anchors:
            qi = i if i < at else i-1
            checks.extend([
                {'role':'reference_adjacent_consonant','index':i,
                 'mean_log_emission':reference['spans'][i]['mean_log_emission']},
                {'role':'query_surviving_consonant','index':qi,
                 'mean_log_emission':query['spans'][qi]['mean_log_emission']}])
        left = query['spans'][at-1]['end_frame'] if at else 0
        right = query['spans'][at]['start_frame'] if at < len(alternate) else len(query_crops[1][2])
        start, end = min(left,right), max(left,right)
    failures = [r for r in checks if r['mean_log_emission'] < floor]
    return {'status':'supported_local_vowel_diagnosis' if not failures else 'weak_local_phonetic_support',
            'checks':checks,'failed_checks':failures,'threshold':floor,
            'query_contrast_start_frame':query_crops[1][1]+start,
            'query_contrast_end_frame':query_crops[1][1]+end,
            'shared_word_alignment_not_reverified':True,
            'not_a_calibrated_error_probability':True}
