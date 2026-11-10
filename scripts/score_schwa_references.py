"""Score real site hypotheses against native variation; no binary Hindi threshold."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.site_reference import rank_site


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--sites', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bank = json.loads(args.reference.read_text())
    lookup = {(r['word'], r['candidate_id'], r['gap_index']): r for r in bank['distributions']}
    if len(lookup) != len(bank['distributions']):
        raise ValueError('Duplicate native site distribution')
    rows = json.loads((args.sites / 'word_sites.json').read_text())
    result = []
    for row in rows:
        for site in row.get('sites', []):
            key = (row['word'], row['candidate_id'], site['gap_index'])
            comparison = rank_site(site['marginal_alternative_minus_lexicon'], lookup[key], row['speaker_id'], row['audio_sha256']) if key in lookup else {'status': 'no_same_word_native_reference', 'native_reference_rank': None, 'n': 0}
            result.append({k: row[k] for k in ['clip_id', 'system', 'speaker_id', 'word', 'word_index', 'candidate_id', 'start', 'end']} |
                          {'gap_index': site['gap_index'], 'raw_site_counterfactual': site['marginal_alternative_minus_lexicon'], **comparison})
    report = {'status': 'native_relative_schwa_evidence_not_validated_hindi_detector', 'site_comparisons': len(result),
              'supported': sum(r['native_reference_rank'] is not None for r in result),
              'unsupported': sum(r['native_reference_rank'] is None for r in result),
              'validated_schwa_accuracy': None, 'binary_threshold': None,
              'self_reference_policy': 'Both query source speaker and exact audio excluded before ranking; each remaining speaker contributes once.'}
    write_group(args.out, {'report.json': report, 'scores.json': result},
                'Reference-relative vowel-site evidence with source-speaker/audio exclusion and explicit insufficient-coverage output.',
                [str(args.reference.resolve()), str(args.sites.resolve()),
                 'Reference SHA ' + hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                 'Input sites SHA ' + hashlib.sha256((args.sites / 'word_sites.json').read_bytes()).hexdigest(),
                 'python -m scripts.score_schwa_references --reference reports/native_schwa_reference_v3/reference.json --sites SITE_GROUP --out NEW'])
    print(json.dumps(report))


if __name__ == '__main__':
    main()
