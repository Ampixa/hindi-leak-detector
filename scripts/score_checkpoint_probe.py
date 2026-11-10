"""Checkpoint-order diagnostics on a different architecture, without accent gold."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.word_model import load_bags, predict_bag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bags, manifest = load_bags(args.evidence)
    models = json.loads(args.models.read_text())
    if set(models) != {'phonetics', 'prosody', 'combined'} or any(m.get('reference_recipe') for m in models.values()):
        raise ValueError('Need independent word heads for this transfer diagnostic')
    if any(m['phone_encoder_revision'] != manifest['phone_encoder_revision'] for m in models.values()):
        raise ValueError('Probe/head phone recipe mismatch')
    groups = defaultdict(dict)
    for bag in bags:
        group = groups[bag['sentence_id']]
        if bag['system'] in group or bag['split'] != 'reserved_evaluation':
            raise ValueError('Duplicate or non-probe checkpoint clip')
        group[bag['system']] = bag
    results = []
    for sentence, group in sorted(groups.items()):
        if set(group) != {'pocket_hindi_parent', 'pocket_nepali_v1c'}:
            raise ValueError('Need the actual matched parent/fine-tune pair')
        parent, tuned = group['pocket_hindi_parent'], group['pocket_nepali_v1c']
        if [w['word'] for w in parent['words']] != [w['word'] for w in tuned['words']]:
            raise ValueError('Checkpoint word sequence mismatch')
        branches = {}
        for name, model in models.items():
            a, wa = predict_bag(model, parent); b, wb = predict_bag(model, tuned)
            branches[name] = {'parent_score': a, 'fine_tune_score': b, 'parent_minus_fine_tune': a - b,
                              'words': [{'word': word['word'], 'parent_source_evidence': x, 'fine_tune_source_evidence': y}
                                        for word, x, y in zip(parent['words'], wa, wb, strict=True)]}
        results.append({'sentence_id': sentence, 'parent_audio_sha256': parent['audio_sha256'],
                        'fine_tune_audio_sha256': tuned['audio_sha256'], 'branches': branches})
    summary = {name: {'pairs': len(results),
                      'parent_above_fine_tune': sum(r['branches'][name]['parent_minus_fine_tune'] > 0 for r in results),
                      'mean_parent_minus_fine_tune': float(np.mean([r['branches'][name]['parent_minus_fine_tune'] for r in results]))}
               for name in models}
    report = {'status': 'same_architecture_external_checkpoint_diagnostic_not_accent_accuracy',
              'branches': summary, 'validated_hindi_accuracy': None,
              'limits': ['Six seed/text pairs cover only three authored texts, not six independent linguistic samples.',
                         'Parent versus fine-tune role is a diagnostic direction, not human accent gold.',
                         'The historical exact parent weight hash used by the fine-tune remains unverified.',
                         'Successful ordering would not establish localized schwa/prosody correctness.']}
    source = Path(__file__)
    write_group(args.out, {'report.json': report, 'predictions.json': results,
                          'source_snapshot.json': {'path': 'scripts/' + source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'source_utf8': source.read_text()}},
                'Check whether local source evidence transfers to Pocket checkpoint pairs sharing architecture and synthesis recipe.',
                [str(args.evidence.resolve()), 'Evidence manifest SHA ' + hashlib.sha256((args.evidence / 'manifest.json').read_bytes()).hexdigest(),
                 str(args.models.resolve()), 'Model SHA ' + hashlib.sha256(args.models.read_bytes()).hexdigest(),
                 'python -m scripts.score_checkpoint_probe --evidence reports/wordlab_pocket_probe_v3 --models reports/wordlab_models_v3/models.json --out NEW'])
    print(json.dumps(report))


if __name__ == '__main__':
    main()
