"""Refresh the development-only checkpoint with all currently complete pairs."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime,timezone
import json
import logging
from pathlib import Path

from scripts.extract_corrected_training import read_rows
from scripts.train_corrected_detector import complete_groups,fit

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'reports/structured_evidence_v15';out=ROOT/'reports/corrected_training_v16'
    if (out/'preview').exists() or (out/'final').exists():raise FileExistsError('Fresh preview required; never replace final model')
    partitions=complete_groups(base);groups=defaultdict(list)
    for r in read_rows(base/'cpu_recovery/items.jsonl'):groups[r['text_id']].append(r)
    for tid,rows in groups.items():
        if len(rows)==6 and sum(r['kind']=='physical' for r in rows)==2:
            partitions[rows[0]['row']['split']][tid]=rows
    contrast={r['word'] for r in json.loads((ROOT/'data/pocket_bulk_plan_v11/g2p_priors.json').read_text())}
    report=fit(base,out/'preview',partitions,contrast,final=False,pairwise=True)
    if not (out/'final/report.json').exists():
        (ROOT/'active_training.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
            'creator':'Codex /root','owner':'Ampixa/user','current':'reports/corrected_training_v16',
            'model':'reports/corrected_training_v16/preview/model.pt','model_sha256':report['model_sha256'],
            'status':'development_preview_final_pipeline_active','next_review':'2026-11-09','actual_hindi_error_accuracy':None},indent=2)+'\n')
    print(json.dumps({'partition_texts':report['partition_texts'],'model_sha256':report['model_sha256'],'reserved_test':report['reserved_test']}))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Corrected preview refresh failed explicitly');raise
