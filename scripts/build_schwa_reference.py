"""Native per-word/site reference distributions, without accent labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sites', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.sites / 'word_sites.json').read_text())
    selection = json.loads(args.selection.read_text())
    jobs = {j['id']: j for j in selection['jobs']}
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['status'] != 'ok_unreviewed_counterfactual' or row['sentence_id'] not in jobs:
            continue
        job = jobs[row['sentence_id']]
        if row['system'] != 'native_recording' or row['split'] != 'train' or row['speaker_id'] != job['speaker_id']:
            raise ValueError('Native bank contains another source/split/group')
        if row['word'] not in job['matched_anchor_words']:
            continue
        for site in row['sites']:
            key = (row['word'], row['candidate_id'], site['gap_index'])
            groups[key][row['speaker_id']].append({'score': site['marginal_alternative_minus_lexicon'],
                                                 'audio_sha256': row['audio_sha256'], 'clip_id': row['clip_id']})
    distributions = []
    for (word, candidate, gap), speakers in sorted(groups.items()):
        references = [{'speaker_id': s, 'score': float(np.mean([v['score'] for v in values])),
                       'audio_sha256s': sorted({v['audio_sha256'] for v in values}),
                       'clip_ids': sorted({v['clip_id'] for v in values})} for s, values in sorted(speakers.items())]
        scores = np.array([r['score'] for r in references])
        if len(scores) < 4:
            continue
        distributions.append({'word': word, 'candidate_id': candidate, 'gap_index': gap,
                              'source_group_count': len(references), 'references': references,
                              'median_score': float(np.median(scores)),
                              'median_absolute_deviation': float(np.median(np.abs(scores - np.median(scores)))),
                              'minimum': float(scores.min()), 'maximum': float(scores.max())})
    result = {'status': 'native_site_reference_distribution_not_hindi_gold', 'distributions': distributions,
              'supported_anchor_words': len({d['word'] for d in distributions}),
              'minimum_independent_source_groups': 4,
              'comparison_rule': 'For a new word/site, compare its raw counterfactual score with same-word native references; exclude the query audio and source speaker group before calculating an empirical rank. Never treat rank as Hindi probability.',
              'no_reference_rule': 'Return unsupported/insufficient references; do not substitute zero risk or a different word.',
              'binary_threshold': None, 'validated_schwa_accuracy': None,
              'limits': ['Six native source groups per selected lexeme are a small reference sample.',
                         'Native lexicon alternatives may both be legitimate; reference rank is a diagnostic, not correctness.',
                         'Native phrase contexts differ. This bank cannot determine intonation appropriateness.',
                         'Each speaker contributes one averaged score, so repeated tokens do not inflate independent reference count.']}
    write_group(args.out, {'reference.json': result}, 'Center schwa-site evidence on actual same-word native variation instead of treating deletion as an error.',
                [str(args.sites.resolve()), str(args.selection.resolve()),
                 'Sites SHA ' + hashlib.sha256((args.sites / 'word_sites.json').read_bytes()).hexdigest(),
                 'Selection SHA ' + hashlib.sha256(args.selection.read_bytes()).hexdigest(),
                 'Driver SHA ' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'python -m scripts.build_schwa_reference --sites reports/schwa_sites_native_benchmark_v3 --selection data/native_wordbank_v3/selection.json --out NEW'])
    print(json.dumps({'supported_anchor_words': result['supported_anchor_words'], 'site_distributions': len(distributions)}))


if __name__ == '__main__':
    main()
