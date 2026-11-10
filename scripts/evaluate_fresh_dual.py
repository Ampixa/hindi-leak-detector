"""Fresh unseen word/voice/carrier request audit after dual-head freeze."""
from __future__ import annotations

from collections import Counter
from datetime import datetime,timezone
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_paired_detector import load,score,decisions
from hindi_detector.evaluation import binary_metrics
from hindi_detector.paired_local_detector import load as local_load,score as local_score,decisions as local_decisions
from hindi_detector.paired_local_vowels import vector
from scripts.extract_corrected_training import read_rows

ROOT=Path(__file__).resolve().parents[1]


def metrics(y,rows,probabilities,ds):
    result={}
    for kind in ('vowel_deletion','vowel_quality'):
        mask=np.array([r['kind']==kind for r in rows])
        if mask.any():result[kind]={'final_decision':binary_metrics(y[mask],(np.array(ds)[mask]=='strong_vowel_change').astype(float),.5),
            'paired_ordering':float(np.mean(probabilities[mask][1::2]>probabilities[mask][::2]))}
    return result


def main():
    base=ROOT/'reports/intrinsic_audit_v29';out=ROOT/'reports/fresh_dual_audit_v29';model=ROOT/'reports/dual_paired_detector_v28'
    if out.exists():raise FileExistsError(out)
    if json.loads((model/'status.json').read_text())['phase']!='complete_frozen_paired_local_candidate':raise ValueError('Head is not frozen')
    if json.loads((base/'status.json').read_text())['phase']!='complete_natural_intrinsic_matrices':raise ValueError('Fresh data incomplete')
    network,state=load(model/'model.pt');local,local_state=local_load(ROOT/'reports/paired_local_detector_v27/model.pt')
    if sha(model/'model.pt')!=json.loads((model/'status.json').read_text())['model_sha256']:
        raise ValueError('Frozen dual checkpoint changed before fresh evaluation')
    plan=ROOT/'data/minimal_audit_v29';selection=json.loads((plan/'selection.json').read_text());jobs={j['id']:j for j in selection['jobs']}
    rows=read_rows(base/'items.jsonl');lookup={(r['word_id'],jobs[r['job_id']]['voice_ordinal'],r['label']):r for r in rows}
    used=[r for r in json.loads((ROOT/'reports/dual_paired_features_v28/index.json').read_text()) if r['split'] in {'train','development'}]
    if {r['word'] for r in rows}&{r['word'] for r in used}:raise ValueError('Fresh word entered fitting/selection')
    if {r['voice'] for r in rows}&{r['voice'] for r in used}:raise ValueError('Fresh conditioning group entered fitting/selection')
    vectors=[];metadata=[];qualifications=[]
    for q in rows:
        r=lookup[q['word_id'],1-jobs[q['job_id']]['voice_ordinal'],0]
        a=vector(q['features'],r['features'],q['evidence'],r['evidence'],q['kind'])
        b=vector(q['secondary_features'],r['secondary_features'],q['secondary_evidence'],r['secondary_evidence'],q['kind'])
        # Same query-pair criterion as the producer plus native other-voice ref.
        rf=np.array(r['secondary_features']);valid=q['secondary_realization_qualified'] and rf[2]<0 and rf[8]>-1.5 and rf[5]>-1.5 and rf[30]==1
        metadata.append({k:q[k] for k in ('job_id','word_id','word','voice','label','kind')});qualifications.append(bool(valid));vectors.append(np.r_[a,b])
    x=np.array(vectors,np.float32);y=np.array([r['label'] for r in metadata]);q=np.array(qualifications,bool)
    if len(rows)%2 or any(a!=b for a,b in zip(q[::2],q[1::2],strict=True)):raise ValueError('Fresh audit requires whole paired qualification')
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete_fresh_request_audit',
        'source_plan_sha256':sha(plan/'selection.json'),'source_items_sha256':sha(base/'items.jsonl'),
        'pairs':len(rows)//2,'secondary_qualified_pairs':int(q.sum())//2,'word_families':len({r['word'] for r in metadata}),
        'conditioning_groups':sorted({r['voice'] for r in metadata}),'all_words_voices_carrier_excluded_from_fitting':True,
        'models':{},'actual_hindi_error_accuracy':None,'limits':[
            'All-row labels are requested spellings; canonical teacher requests can themselves realize the competing vowel.',
            'Qualified subset labels are acoustic-derived and share the LV60 probe used by the dual head; they are a circular diagnostic, not independent accuracy.',
            'Only actual waveform interventions provide known controlled change labels; real Hindi-error accuracy remains unknown.']}
    predictions=[]
    for name,net,st,values,scorer,decider in [('paired57',local,local_state,x[:,:57],local_score,local_decisions),('dual114',network,state,x,score,decisions)]:
        p=scorer(net,st,values);ds=decider(st,values,p)
        report['models'][name]={'model_sha256':sha((ROOT/'reports/paired_local_detector_v27/model.pt') if name=='paired57' else model/'model.pt'),
            'all_requested':metrics(y,metadata,p,ds),'secondary_qualified_circular_diagnostic':metrics(y[q],[r for r,m in zip(metadata,q) if m],p[q],[d for d,m in zip(ds,q) if m]) if q.any() else None}
        predictions.extend({**r,'model':name,'score':float(v),'decision':d,'secondary_qualified':bool(m)} for r,v,d,m in zip(metadata,p,ds,q,strict=True))
    write_group(out,{'report.json':report,'predictions.json':predictions,'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Fresh64-word/two-unseen-conditioning-voice/new-carrier request evaluation after model freeze. '
        'report.json compares prior57 and dual114 without tuning; predictions.json every raw/qualified '
        'request decision. Qualified diagnostic is explicitly circular because LV60 supplied its labels. '
        'No raw audio/model copy, NOT on HF/inherited rights. Current consumer release evidence; review '
        '2026-11-09, retain compact conclusions through selected model review.',
        ['.venv-indicf5/bin/python -m scripts.evaluate_fresh_dual'])
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Fresh dual request audit failed explicitly');raise
