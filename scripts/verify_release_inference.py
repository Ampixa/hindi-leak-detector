"""Cold CLI parity, automatic-reference smoke and unknown-word execution checks."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import traceback

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from scripts.extract_corrected_training import read_rows

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'reports/release_cold_verification_v31');args=parser.parse_args()
    out=args.out;model=ROOT/'reports/task_vowel_release_v31'
    if out.exists():raise FileExistsError(out)
    actual=json.loads((model/'actual_swar/cases.json').read_text());cases={r['id']:r for r in actual};requests=[]
    for key,word,expected in [('real_yt_yt01_005','हैन',True),('real_yt_yt01_008','किनकि',True),('real_yt_yt04_030','किनकि',False)]:
        case=cases[key]
        for mode in ('explicit','teacher_cache'):
            requests.append({'name':key+'_'+mode,'query':str(ROOT/'data/swar_speech_v18'/case['analysis']['swar']['audio']),
                'reference':str(ROOT/'data/swar_speech_v18'/case['analysis']['real']['audio']) if mode=='explicit' else None,
                'text':case['text'],'word':word,'expected_flag':expected,'actual_case':key,'mode':mode})
    plan=ROOT/'data/minimal_audit_v29';selection=json.loads((plan/'selection.json').read_text());jobs={j['id']:j for j in selection['jobs']}
    rows=read_rows(ROOT/'reports/intrinsic_audit_v29/items.jsonl');lookup={(r['word_id'],jobs[r['job_id']]['voice_ordinal'],r['label']):r for r in rows}
    cached={r['word'] for r in json.loads((ROOT/'data/swar_phone_plan_v18/words.json').read_text())};receipts={r['id']:r for r in read_rows(ROOT/'reports/minimal_audit_features_v29/items.jsonl')}
    for kind in ('vowel_deletion','vowel_quality'):
        candidates=[]
        for row in rows:
            if row['kind']!=kind or row['label']!=1 or not row['secondary_realization_qualified'] or row['word'] in cached:continue
            other=lookup[row['word_id'],1-jobs[row['job_id']]['voice_ordinal'],0]
            if any(f[2]>=0 or f[8]<=-1.5 for f in (other['features'],other['secondary_features'])):continue
            candidates.append((row,other))
        if not candidates:raise ValueError('No pre-score realized unknown-word API example for '+kind)
        q,r=sorted(candidates,key=lambda pair:pair[0]['job_id'])[0]
        requests.append({'name':'unknown_'+kind,'query':str(plan/'teacher'/receipts[q['job_id']]['audio']),
            'reference':str(plan/'teacher'/receipts[r['job_id']]['audio']),'text':jobs[q['job_id']]['canonical_text'],
            'word':q['word'],'expected_flag':None,'actual_case':None,'mode':'unknown_word_explicit',
            'selection':'First job-ID realized request pair, independent of classifier scores; target absent from799-word runtime cache'})
    write_group(out,{'plan.json':requests,'protocol.json':{'model_sha256':sha(model/'model.pt'),
        'parity_tolerance_features':1e-5,'parity_tolerance_word_seconds':1e-6,
        'known_cases':'Development regressions, not independent error accuracy',
        'unknown_cases':'API/dynamic-G2P execution only; requested phonetic changes, not real Hindi-error gold'},
        'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Cold CLI checks. plan.json exact sources/text/target expectations before execution; each named '
        'subdirectory contains original CLI result/provenance and matching.log contains subprocess output. '
        'report.json expected development regression checks, batch/cold feature/time parity and unknown '
        'word processing. No raw audio/weight copies. Current consumer release promotion, review2026-11-09; '
        'keep compact conclusions, diagnostic logs disposable24h after consumer closes. NOT on HF/inherited rights.',
        ['.venv-indicf5/bin/python -m scripts.verify_release_inference'])
    def status(phase,**kw):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');t.replace(p)
    results=[]
    try:
        for request in requests:
            status('running_cold_inference_checks',completed=len(results),planned=len(requests),current=request['name'])
            directory=out/request['name'];cmd=[str(ROOT/'.venv-indicf5/bin/python'),'-m','scripts.detect_nepali_leakage',
                '--query',request['query'],'--text',request['text'],'--out',str(directory)]
            if request['reference']:cmd+=['--reference',request['reference']]
            log=out/(request['name']+'.log')
            with log.open('xb') as stream:subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=180)
            value=json.loads((directory/'result.json').read_text());words=[w for w in value['words'] if w['word']==request['word']]
            flagged=any(w['decision']=='strong_vowel_change' for w in words);processed=any(w['sites'] for w in words)
            result={'name':request['name'],'word':request['word'],'processed':processed,'flagged':flagged,'expected_flag':request['expected_flag'],
                'check_passed':processed and (request['expected_flag'] is None or flagged==request['expected_flag']),
                'result_sha256':sha(directory/'result.json'),'model_sha256':value['model_sha256']}
            if request['mode']=='explicit':
                batch=cases[request['actual_case']]['analysis']['swar']['words'];cold=value['per_reference'][0]['words'];delta=0.;time_delta=0.
                if len(batch)!=len(cold):raise ValueError('Cold/batch canonical word count differs')
                for a,b in zip(batch,cold,strict=True):
                    if a['word']!=b['word']:raise ValueError('Cold/batch word identity differs')
                    time_delta=max(time_delta,abs(a['start']-b['start']),abs(a['end']-b['end']))
                    def key(site):return site['site']['kind'],site['site']['index'],tuple(site['native_phones'])
                    av={key(s):s for s in a['sites']};bv={key(s):s for s in b['sites']}
                    if set(av)!=set(bv):raise ValueError('Cold/batch tested site coverage differs')
                    for k in av:delta=max(delta,float(np.max(np.abs(np.array(av[k]['features'])-np.array(bv[k]['features'])))))
                result.update(max_feature_difference=delta,max_word_time_difference_seconds=time_delta)
                if delta>1e-5 or time_delta>1e-6:raise ValueError('Cold CLI feature/time parity failed')
            results.append(result);(out/'partial_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        report={'status':'complete_cold_release_verification','checks':results,'checks_passed':sum(r['check_passed'] for r in results),
            'checks_total':len(results),'model_sha256':sha(model/'model.pt'),'actual_hindi_error_accuracy':None}
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');status(report['status'],passed=report['checks_passed'],total=len(results));print(json.dumps(report,ensure_ascii=False))
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Cold release verification failed explicitly');raise
