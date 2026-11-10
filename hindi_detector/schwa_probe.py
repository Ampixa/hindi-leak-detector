"""Site counterfactuals from unreviewed lexicons, never Hindi/accent gold."""
from __future__ import annotations

import math

import numpy as np

from hindi_detector.candidates import schwa_slots
from hindi_detector.ctc import log_likelihood

CENTRAL = ['ʌ', 'ə', 'ɐ']
# Equivalent tie-bar notation only; no dental/retroflex/aspiration collapse.
NOTATION = {'t͡s': 'ts', 't͡sʰ': 'tsʰ', 'd͡z': 'dz', 'd͡zʱ': 'dzʱ'}


def site_paths(candidate: dict, ipa: dict[str, str], vocab: dict[str, int]) -> list[dict]:
    target = candidate['target_phones']
    skeleton, slots = schwa_slots(target)
    other_skeleton, other_slots = schwa_slots(candidate['hindi_schwa_hypothesis'])
    if skeleton != other_skeleton:
        raise ValueError('Schwa comparison changes non-schwa lexical phones')
    actual = [{'gap_index': i, 'target_count': a, 'hindi_count': b} for i, (a, b) in enumerate(zip(slots, other_slots, strict=True)) if a != b]
    if actual != candidate['sites']:
        raise ValueError('Candidate site metadata disagrees with actual phone sequences')

    def encode(phones, central):
        tokens = []
        for phone in phones:
            if phone not in ipa or not ipa[phone]:
                raise ValueError('No explicit lexical IPA mapping for ' + phone)
            token = central if phone == 'ax' else ipa[phone]
            if token not in vocab:
                token = NOTATION.get(token, token)
            if token not in vocab or vocab[token] in {0, 1, 2, 3}:
                raise ValueError('Probe vocabulary cannot represent lexical IPA: ' + phone + ' -> ' + token)
            tokens.append(vocab[token])
        return tokens

    paths = []
    for site in actual:
        alternate = []
        for gap in range(len(slots)):
            count = site['hindi_count'] if gap == site['gap_index'] else slots[gap]
            alternate.extend(['ax'] * count)
            if gap < len(skeleton):
                alternate.append(skeleton[gap])
        paths.append({'gap_index': site['gap_index'], 'lexicon_schwa_count': site['target_count'],
                      'alternative_schwa_count': site['hindi_count'], 'lexicon_project_phones': target,
                      'alternative_project_phones': alternate,
                      'vowel_quality_nuisance_paths': [{'central_vowel': vowel, 'lexicon_ids': encode(target, vowel),
                                                       'alternative_ids': encode(alternate, vowel)} for vowel in CENTRAL]})
    return paths


def compare_site(log_probs: np.ndarray, paths: dict) -> dict:
    values = []
    for variant in paths['vowel_quality_nuisance_paths']:
        a = log_likelihood(log_probs, variant['lexicon_ids'])
        b = log_likelihood(log_probs, variant['alternative_ids'])
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Word frames cannot represent both site hypotheses')
        values.append({'central_vowel': variant['central_vowel'], 'lexicon_log_likelihood': a,
                       'alternative_log_likelihood': b, 'alternative_minus_lexicon': b - a})
    a = float(np.logaddexp.reduce([v['lexicon_log_likelihood'] for v in values]) - math.log(len(values)))
    b = float(np.logaddexp.reduce([v['alternative_log_likelihood'] for v in values]) - math.log(len(values)))
    return {'gap_index': paths['gap_index'], 'lexicon_schwa_count': paths['lexicon_schwa_count'],
            'alternative_schwa_count': paths['alternative_schwa_count'],
            'lexicon_project_phones': paths['lexicon_project_phones'],
            'alternative_project_phones': paths['alternative_project_phones'], 'vowel_conditioned_results': values,
            'marginal_alternative_minus_lexicon': b - a,
            'status': 'unreviewed_site_counterfactual_not_hindi_label',
            'meaning': 'Raw exact CTC evidence for a site-count alternative versus the lexicon path, marginalized equally over three central-vowel qualities. Not a correctness probability.'}
