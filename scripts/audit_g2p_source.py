"""Coverage and source contrast of exact competing G2P paths, without relabeling."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import binary_metrics
from hindi_detector.paired_detector import sha

LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-group', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    inputs = json.loads((args.model_group / 'inputs.json').read_text()); directory = Path(inputs['features'])
    rows = {}; coverage = {}; reasons = Counter()
    for part, receipts in inputs['frozen_receipts'].items():
        groups = defaultdict(dict)
        for receipt in receipts:
            path = directory / receipt['detail']
            if sha(path) != receipt['detail_sha256']:
                raise ValueError('G2P detail identity differs')
            detail = json.loads(path.read_text())
            groups[receipt['text_id']][receipt['row']['system'], receipt['row']['speaker_id']] = (receipt, detail)
        selected = []; supported_pairs = 0; unsupported_pairs = 0
        for tid, group in groups.items():
            voices = sorted({key[1] for key in group})
            for j, voice in enumerate(voices):
                native = group['pocket_nepali_v1c', voice]
                hindi = group['pocket_hindi_parent', voice]
                reference = group['pocket_nepali_v1c', voices[1-j]]
                for i in native[0]['focal_word_indices']:
                    gs = [item[1]['words'][i].get('g2p_path_evidence', {}) for item in (native, hindi, reference)]
                    if any(g.get('status') != 'unreviewed_two_g2p_acoustic_evidence' for g in gs):
                        unsupported_pairs += 1
                        reasons.update(g.get('reason', g.get('status', 'absent')) for g in gs if g.get('status') != 'unreviewed_two_g2p_acoustic_evidence')
                        continue
                    supported_pairs += 1
                    delta = lambda g: g['hindi_minus_greedy_per_frame'] - g['nepali_minus_greedy_per_frame']
                    baseline = delta(gs[2])
                    for label, g in enumerate(gs[:2]):
                        selected.append({'text_id': tid, 'word': native[1]['words'][i]['word'], 'label': label,
                            'raw_path_margin_per_frame': delta(g), 'reference_adjusted_margin': delta(g) - baseline,
                            'nepali_minus_greedy_per_frame': g['nepali_minus_greedy_per_frame'],
                            'hindi_minus_greedy_per_frame': g['hindi_minus_greedy_per_frame']})
        rows[part] = selected; coverage[part] = {'supported_pairs': supported_pairs, 'unsupported_pairs': unsupported_pairs,
            'supported_focal_lexemes': len({r['word'] for r in selected})}
    train = rows['train']; dev = rows['development']; test = rows['reserved_evaluation']
    if not train or not dev or not test:
        raise ValueError('No supported competing paths in one or more partitions')
    summary = {'coverage': coverage, 'unsupported_reasons': dict(reasons), 'margins': {}}
    for name in ('raw_path_margin_per_frame', 'reference_adjusted_margin'):
        # Choose direction solely from train distributions, threshold from development.
        positive = np.array([r[name] for r in train if r['label']]); negative = np.array([r[name] for r in train if not r['label']])
        direction = 1 if np.median(positive) >= np.median(negative) else -1
        dv = np.array([direction * r[name] for r in dev]); dl = np.array([r['label'] for r in dev])
        choices = np.unique(np.r_[dv, np.nextafter(dv.max(), np.inf)])
        threshold = max(choices, key=lambda t: (np.mean(dv[dl == 1] >= t) + np.mean(dv[dl == 0] < t), -abs(t)))
        tv = np.array([direction * r[name] for r in test]); tl = np.array([r['label'] for r in test])
        # Binary metric scores encode the frozen margin decision, not probabilities.
        summary['margins'][name] = {'direction_from_training': direction, 'threshold_from_development': float(threshold),
            'test_source_metrics': binary_metrics(tl, (tv >= threshold).astype(float)),
            'paired_hindi_above_native': sum(direction*(b[name]-a[name]) > 0 for a,b in zip(test[::2],test[1::2])),
            'test_pairs': len(test)//2}
    report = {'summary': summary, 'actual_hindi_error_accuracy': None,
        'limits': ['Only exactly representable phone paths are scored; dental/aspiration distinctions are not collapsed.',
                   'This audit uses the first native lexicon variant as a diagnostic path; alternatives remain valid and are not error labels.',
                   'Both predicted paths can fit poorly. Source separation is not native pronunciation correctness.',
                   'Direction selected on train and threshold on development; no held-out tuning.']}
    write_group(args.out, {'report.json': report, 'margins.json': rows,
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Exact G2P path coverage and source-order diagnostic; compact native JSON metrics and per-pair actual margins. No new labels, phone projections, model fit or waveform copied.',
        [str(args.model_group.resolve()), 'python -m scripts.audit_g2p_source --model-group GROUP --out NEW'])
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Exact G2P source audit failed')
        raise
