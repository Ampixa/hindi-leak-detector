"""Summarize observed human path evidence without manufacturing Hindi/native labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bank = json.loads((args.bank / 'word_bank.json').read_text())
    selection = json.loads((args.bank / 'selection.json').read_text())
    clips = {r['clip_id']: r for r in selection['clips']}
    metadata = json.loads((args.evidence / 'manifest.json').read_text())
    if metadata['g2p_bank_sha256'] != sha(args.bank / 'word_bank.json'):
        raise ValueError('Human G2P evidence and source bank differ')
    allow = {r['word']: {e['audio']: e for e in r['human_examples']} for r in bank}
    observations = defaultdict(list)
    for item in metadata['items']:
        if item['status'] != 'ok_unreviewed_word_evidence':
            continue
        clip = clips[item['clip_id']]
        if clip['system'] != 'human_nepali_unlabeled' or clip['sha256'] != item['audio_sha256']:
            raise ValueError('Human realization analysis must retain unlabelled source identity')
        path = (args.evidence / item['detail']).resolve(strict=True)
        if not path.is_relative_to(args.evidence.resolve()) or sha(path) != item['detail_sha256']:
            raise ValueError('Human word evidence identity mismatch')
        detail = json.loads(path.read_text())
        for index, word in enumerate(detail['words']):
            if word['word'] not in allow or clip['audio'] not in allow[word['word']]:
                continue
            example = allow[word['word']][clip['audio']]
            evidence = word.get('g2p_path_evidence', {'status': 'missing_g2p_evidence', 'hindi_minus_nepali_log_likelihood': None})
            observations[word['word']].append({'clip_id': item['clip_id'], 'speaker_id': item['speaker_id'],
                                              'state': example['state'], 'district': example['district'], 'word_index': index,
                                              'start': word['start'], 'end': word['end'], 'evidence': evidence})
    summaries = []
    for word in sorted(allow):
        rows = observations[word]; statuses = Counter(r['evidence']['status'] for r in rows)
        speakers = defaultdict(list)
        for row in rows:
            score = row['evidence'].get('hindi_minus_nepali_log_likelihood')
            if score is not None:
                speakers[row['speaker_id']].append(score)
        medians = [float(np.median(v)) for v in speakers.values()]
        summaries.append({'word': word, 'planned_recordings': len(allow[word]), 'observed_word_occurrences': len(rows),
                          'source_groups': len({r['speaker_id'] for r in rows}), 'statuses': dict(statuses),
                          'supported_source_groups': len(medians),
                          'source_groups_favoring_hindi_g2p_path': sum(v > 0 for v in medians),
                          'source_groups_favoring_nepali_g2p_path': sum(v < 0 for v in medians),
                          'median_hindi_minus_nepali_log_likelihood': float(np.median(medians)) if medians else None,
                          'interpretation': 'Within-speaker median prevents repeated words inflating independent evidence. Path preference can reflect valid variation, G2P error, or probe error; it is not a Hindi-leakage label.'})
    report = {'status': 'real_human_dual_g2p_evidence_collected_unlabelled', 'word_families': len(summaries),
              'families_with_supported_comparisons': sum(r['supported_source_groups'] > 0 for r in summaries),
              'observed_planned_word_occurrences': sum(r['observed_word_occurrences'] for r in summaries),
              'extraction_successes': metadata['successes'], 'extraction_failures': metadata['failures'],
              'native_acceptability_accuracy': None, 'preferred_style_accuracy': None, 'hindi_error_accuracy': None,
              'geography_used_as_label': False, 'g2p_outputs_used_as_gold': False,
              'limits': ['This is a speaker-weighted observation bank, not a trained/validated human-word error classifier.',
                         'Full predicted path preference mixes phone quality, phone count and model path-length bias.',
                         'Both G2P paths may fit poorly; per-frame comparisons with the probe greedy sequence are retained.',
                         'Unsupported IPA and impossible paths remain explicit, never native support or zero risk.',
                         'Current regional coverage is reported West Bengal districts; Nepal-region coverage is not established.',
                         'Single-original-utterance candidacy uses text/duration metadata, not verified original audio identity.',
                         'Native variation, preferred TTS style and Hindi-phone evidence are separate requested outputs.']}
    write_group(args.out, {'report.json': report, 'word_summary.json': summaries, 'observations.json': dict(observations)},
                'Ground competing G2P hypotheses in selected real human word recordings; do not equate Hindi-path preference with an error.',
                [str(args.bank.resolve()), str(args.evidence.resolve()), 'Evidence SHA ' + sha(args.evidence / 'manifest.json'),
                 'Bank SHA ' + sha(args.bank / 'word_bank.json'), 'Driver SHA ' + sha(Path(__file__)),
                 'python -m scripts.summarize_human_g2p --bank data/human_g2p_bank_v5 --evidence reports/human_g2p_evidence_v5 --out NEW'])
    print(json.dumps(report))


if __name__ == '__main__':
    main()
