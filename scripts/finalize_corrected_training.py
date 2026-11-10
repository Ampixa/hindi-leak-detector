"""Write a compact final review after fitting and recorded-speaker auditing finish."""
from __future__ import annotations

from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np

from hindi_detector.corrected_synthesis import sha

ROOT=Path(__file__).resolve().parents[1]


def main(*, resume_failed=False):
    training=ROOT/'reports/corrected_training_v16';native=ROOT/'reports/native_variation_audit_v16'
    target=training/'final_summary.json';progress=training/'finalizer_status.json'
    if target.exists():raise FileExistsError('Final summary already exists')
    if progress.exists():
        previous=json.loads(progress.read_text())
        if not resume_failed or previous['phase']!='failed_explicitly':
            raise FileExistsError('Only an explicitly failed finalizer may resume')
    def status(phase,**extra):
        p=progress.with_suffix('.partial');p.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n');p.replace(progress)
    try:
        status('waiting_for_final_training_and_recorded_variation')
        while True:
            states=[json.loads((p/'status.json').read_text()) for p in (training,native)]
            if any(s['phase']=='failed_explicitly' for s in states):raise RuntimeError('An owned final fitting/audit stage failed; inspect its explicit log')
            if states[0]['phase']=='complete_corrected_training_and_reserved_evaluation' and states[1]['phase']=='complete_native_variation_audit':break
            for s in states:
                if not s['phase'].startswith('complete_'):os.kill(s['pid'],0)
            time.sleep(10)
        report=json.loads((training/'final/report.json').read_text());variation=json.loads((native/'report.json').read_text())
        digest=sha(training/'final/model.pt')
        if digest!=report['model_sha256'] or digest!=variation['model_sha256']:raise ValueError('Final model/audit identities differ')
        predictions=json.loads((native/'predictions.json').read_text());sensitivity={}
        for branch in ('phonetics','prosody','combined'):
            groups=defaultdict(list)
            for row in predictions:
                for w in row['words']:
                    groups[row['human_clip_id'],w['word'],w['start'],w['end']].append(w['scores'][branch])
            high=report['reserved_test'][branch]['operating_points']['high']
            matched=[v for v in groups.values() if len(v)==2]
            sensitivity[branch]={'word_occurrences_with_two_references':len(matched),
                'flag_flip_fraction':float(np.mean([(v[0]>=high)!=(v[1]>=high) for v in matched])) if matched else None,
                'median_score_delta':float(np.median([abs(v[0]-v[1]) for v in matched])) if matched else None}
        # Group by actual aligned occurrence, so repeated words remain separate.
        (native/'reference_sensitivity_by_occurrence.json').write_text(json.dumps({'model_sha256':digest,'summary':sensitivity,
            'key':'human_clip_id, lexical_word, actual_start, actual_end','actual_hindi_error_accuracy':None},indent=2)+'\n')
        content=defaultdict(Counter)
        for line in (ROOT/'reports/corrected_pair_v14/content_items.jsonl').read_text().splitlines():
            row=json.loads(line);content[row['system']+'/'+row['split']+'/'+row['text_kind']][row['content_quality']['decision']]+=1
        recovery=json.loads((training/'final/data_quality_addendum.json').read_text())
        summary={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'corrected_training_and_audits_complete',
            'model_path':str(training/'final/model.pt'),'model_sha256':digest,'systems':['pocket_teacher_250000_24l','indicf5_stock'],
            'partition_texts':report['partition_texts'],'partition_focal_lexemes':report['partition_focal_lexemes'],
            'conditioning_groups':report['conditioning_groups'],'physical_reserved_source_metrics':report['reserved_test'],
            'original_tts_transfer_source_metrics':report['original_tts_transfer'],'recorded_variation':variation,
            'reference_sensitivity_by_actual_occurrence':sensitivity,'content_coverage_by_source_split_kind':{k:dict(v) for k,v in content.items()},
            'cpu_recovery':recovery,'actual_hindi_error_accuracy':None,'actual_prosody_error_accuracy':None,
            'scope':'Known source changes and score distributions. No independently labeled real Hindi pronunciation/prosody error benchmark.',
            'review_url':'http://100.80.123.8:8770/comparison','native_report_sha256':sha(native/'report.json'),'model_report_sha256':sha(training/'final/report.json')}
        target.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        (ROOT/'active_training.json').write_text(json.dumps({'created_utc':summary['created_utc'],'creator':'Codex /root','owner':'Ampixa/user',
            'current':'reports/corrected_training_v16','model':'reports/corrected_training_v16/final/model.pt','model_sha256':digest,
            'summary':'reports/corrected_training_v16/final_summary.json','status':'complete_research_candidate','actual_hindi_error_accuracy':None,
            'next_review':'2026-11-09'},indent=2)+'\n')
        status('complete_final_review',model_sha256=digest)
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Final corrected review assembly failed explicitly');raise
