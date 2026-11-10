"""Compare explicit G2P phone priors; never turn a path preference into accent gold."""
from __future__ import annotations

import re

import numpy as np

from hindi_detector.ctc import compare, log_likelihood


def compile_word_hypotheses(rows: list[dict], vocab: dict[str, int], special: set[int]) -> dict[str, dict]:
    result = {}
    for row in rows:
        word = row['word']
        if not isinstance(word, str) or not word or word in result:
            raise ValueError('Need unique nonempty human-bank words')
        raw_native = row['nepali_g2p']['ipa_candidate_units']
        raw_hindi = row['hindi_g2p']['units_with_stress_preserved']
        if not isinstance(raw_native, list) or not isinstance(raw_hindi, list):
            raise ValueError('G2P hypotheses must be explicit unit lists')
        annotation = {'nepali_candidate_ipa_units': raw_native, 'hindi_candidate_ipa_units': raw_hindi,
                      'probe_special_ids': sorted(special),
                      'score_meaning': 'Raw exact CTC comparison of two predicted phone paths. Native speech can favor either; no Hindi-error label follows.'}
        changes = []

        def encode(values, language):
            output = []
            for original in values:
                if original == '.':
                    changes.append({'language': language, 'unit': '.', 'operation': 'remove syllable-boundary annotation'})
                    continue
                if not isinstance(original, str) or not original.strip():
                    raise ValueError('Missing explicit IPA candidate unit')
                token = original.strip()
                unstressed = re.sub(r'^[ˈˌ]+', '', token)
                if unstressed != token:
                    changes.append({'language': language, 'unit': token, 'projected_unit': unstressed, 'operation': 'remove explicit stress annotation for segmental CTC'})
                token = unstressed
                if token not in vocab:
                    tied = token.replace('\u0361', '').replace('\u035c', '')
                    if tied != token and tied in vocab:
                        changes.append({'language': language, 'unit': token, 'projected_unit': tied, 'operation': 'equivalent affricate tie-bar notation'})
                        token = tied
                if token not in vocab or vocab[token] in special:
                    raise ValueError('Probe cannot represent ' + language + ' IPA unit ' + repr(token))
                output.append(vocab[token])
            if not output:
                raise ValueError('Empty projected G2P path')
            return output

        try:
            a, b = encode(raw_native, 'nepali'), encode(raw_hindi, 'hindi')
            annotation.update(status='supported_competing_g2p_paths' if a != b else 'no_resolved_segmental_contrast',
                              nepali_ids=a, hindi_ids=b)
        except ValueError as error:
            annotation.update(status='unsupported_g2p_projection', reason=str(error))
        annotation['explicit_annotation_projections'] = changes
        result[word] = annotation
    return result


def word_path_evidence(log_probs: np.ndarray, hypothesis: dict) -> dict:
    output = {k: v for k, v in hypothesis.items() if k not in {'nepali_ids', 'hindi_ids'}}
    output['hindi_minus_nepali_log_likelihood'] = None
    if hypothesis['status'] != 'supported_competing_g2p_paths':
        return output
    try:
        score = compare(log_probs, hypothesis['nepali_ids'], hypothesis['hindi_ids'])
        path = np.asarray(log_probs).argmax(axis=1)
        greedy = [int(token) for i, token in enumerate(path) if token != 0 and (i == 0 or token != path[i - 1])]
        greedy_score = log_likelihood(log_probs, greedy)
        output.update(status='unreviewed_two_g2p_acoustic_evidence',
                      nepali_log_likelihood=score.target_log_likelihood,
                      hindi_log_likelihood=score.hindi_log_likelihood,
                      hindi_minus_nepali_log_likelihood=score.hindi_minus_target_log_likelihood,
                      probe_greedy_sequence_ids=greedy, probe_greedy_sequence_log_likelihood=greedy_score,
                      probe_greedy_has_special_labels=any(token in hypothesis['probe_special_ids'] for token in greedy),
                      nepali_minus_greedy_per_frame=(score.target_log_likelihood - greedy_score) / len(path),
                      hindi_minus_greedy_per_frame=(score.hindi_log_likelihood - greedy_score) / len(path),
                      hypotheses_exhaustive=False,
                      adequacy_note='Both G2P paths can fit poorly; their likelihood ratio alone cannot establish a valid pronunciation or error.')
    except ValueError as error:
        output.update(status='impossible_g2p_path_for_word_alignment', reason=str(error))
    return output
