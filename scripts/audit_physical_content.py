"""Recheck physical held-out edits for content loss, without changing the model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.asr_gate import CpuContentGate, RECIPE
from hindi_detector.evaluation import binary_metrics
from scripts.calibrate_asr_gate import audio, sha

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    while True:
        state = json.loads((args.group / 'status.json').read_text())
        if state['phase'] == 'complete_physical_detector_evaluation':
            break
        if datetime.now(timezone.utc) >= datetime.fromisoformat('2026-11-08T02:10:00+00:00'):
            raise TimeoutError('Physical evaluation did not finish before bounded ASR audit deadline')
        os.kill(state['pid'], 0); time.sleep(10)
    predictions = json.loads((args.group / 'predictions.json').read_text())
    model_report = json.loads((args.group / 'report.json').read_text())
    write_group(args.out, {'inputs.json': {'group': str(args.group.resolve()), 'predictions_sha256': sha(args.group / 'predictions.json'),
        'model_sha256': sha(args.group / 'model.pt'), 'recipe': RECIPE}, 'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Post-edit content audit of held-out physical controls with the already-fixed recipe and model. items.jsonl stores decoder outputs/decisions, runtime.json exact ASR. report.json retains all decisions and paired-pass-only source metrics. No retraining or waveform deletion. ASR is never an accent label.',
        ['.venv-asr-gate/bin/python -m scripts.audit_physical_content --group reports/physical_detector_v12 --out NEW'])
    gate = CpuContentGate(ROOT, threads=2)
    (args.out / 'runtime.json').write_text(json.dumps(gate.runtime, indent=2)+'\n')
    by_text = defaultdict(list); counts = defaultdict(Counter)
    for row in predictions:
        samples = audio(args.group, row)
        result = gate.score(samples, row['text'], 'pocket_hindi_parent' if row['condition'] == 'hindi_word_insert' else 'pocket_nepali_v1c')
        receipt = {'text_id':row['text_id'],'condition':row['condition'],'audio':row['audio'],'sha256':row['sha256'],'content_quality':result}
        with (args.out/'items.jsonl').open('a') as f:f.write(json.dumps(receipt,ensure_ascii=False,allow_nan=False)+'\n')
        by_text[row['text_id']].append((row,result)); counts[row['condition']][result['decision']] += 1
    passed = [row for pairs in by_text.values() if len(pairs)==2 and all(r['decision']=='pass' for _,r in pairs) for row,_ in pairs]
    metrics = {}
    if passed:
        for branch, m in model_report['summary'].items():
            pairs = [(w['source_label'],w['scores'][branch]) for r in passed for i,w in enumerate(r['words']) if i in r['focal']]
            labels=np.array([p[0] for p in pairs]);scores=np.array([p[1] for p in pairs])
            metrics[branch]={'development_threshold':binary_metrics(labels,scores,m['threshold']),
                'conservative_threshold':binary_metrics(labels,scores,m['conservative_threshold'])}
    report={'decisions':{k:dict(v) for k,v in counts.items()},'tested_text_pairs':len(by_text),'both_pass_text_pairs':len(passed)//2,
        'paired_pass_focal_source_metrics':metrics,'actual_hindi_error_accuracy':None,
        'warning':'Content-conditioned internal source metrics; rejection can bias coverage. Original unfiltered metrics remain in model report.'}
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        LOG.exception('Physical content audit failed explicitly');raise
