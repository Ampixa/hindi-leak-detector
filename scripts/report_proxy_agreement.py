"""Report coverage explicitly for the post-v2 agreement/abstention policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import agreement_decisions, selective_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.revision / 'predictions.json').read_text())
    probe = json.loads((args.revision / 'probe_predictions.json').read_text())

    def decisions(items):
        return agreement_decisions(np.array([r['v1_score'] for r in items]), np.array([r['v2_score'] for r in items]))

    d = decisions(rows); p = decisions(probe)
    source_i = [i for i, r in enumerate(rows) if r['system'] in {'indicf5_stock', 'audio8_r3'}]
    labels = np.array([int(rows[i]['system'] == 'indicf5_stock') for i in source_i])
    groups = {}
    for system in sorted({r['system'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['system'] == system]
        groups[system] = {'n': len(indices), 'positive_flags': int(np.sum(d[indices] == 1)),
                          'negative_decisions': int(np.sum(d[indices] == 0)), 'uncertain': int(np.sum(d[indices] == -1))}
    probe_groups = {}
    for variant in sorted({r['variant'] for r in probe}):
        indices = [i for i, r in enumerate(probe) if r['variant'] == variant]
        probe_groups[variant] = selective_metrics(np.zeros(len(indices), dtype=int), p[indices])
    report = {'status': 'exploratory_two_head_agreement_policy', 'rule': 'Both fixed-0.5 heads must agree; otherwise uncertain.',
              'source_proxy_metrics': selective_metrics(labels, d[source_i]),
              'benchmark_controls': groups, 'native_probe_by_view': probe_groups,
              'provenance_limit': 'Rule introduced after binary v2 results. Reused benchmark/probe measurements are exploratory; no fresh accuracy claim.',
              'limits': ['Heads are correlated and agreement is not a calibrated confidence guarantee.',
                         'Coverage and positive recall include abstentions explicitly; covered accuracy alone is insufficient.',
                         'All reference labels remain weak source/native assumptions, not human accent gold.']}
    output = [{**r, 'decision': {-1: 'uncertain', 0: 'nepali_like_proxy', 1: 'hindi_like_proxy'}[int(value)]}
              for r, value in zip(rows, d, strict=True)]
    write_group(args.out, {'report.json': report, 'predictions.json': output},
                'Conservative proxy agreement with explicit abstention coverage; exploratory, not validated Hindi detection.',
                [f'{args.revision.resolve()} predictions SHA256 {hashlib.sha256((args.revision / "predictions.json").read_bytes()).hexdigest()}',
                 f'probe predictions SHA256 {hashlib.sha256((args.revision / "probe_predictions.json").read_bytes()).hexdigest()}',
                 'python -m scripts.report_proxy_agreement --revision reports/source_proxy_v2 --out NEW'])
    print(json.dumps(report))


if __name__ == '__main__':
    main()
