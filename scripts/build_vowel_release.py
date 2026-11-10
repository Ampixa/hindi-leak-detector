"""Package per-cause regression fallback without word-specific inference rules."""
from __future__ import annotations

import copy
from datetime import datetime,timezone
import json
import logging
from pathlib import Path

import numpy as np
import torch

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_paired_detector import DualPairedNetwork
from hindi_detector.dual_paired_vowels import supported
from hindi_detector.task_vowel_detector import decisions,load,score
from scripts.train_paired_local import partition
from scripts.train_task_vowels import head_prob,metrics

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'reports/task_vowel_detector_v30';out=ROOT/'reports/task_vowel_release_v31';base=ROOT/'reports/dual_geometry_features_v30'
    if out.exists():raise FileExistsError(out)
    _,state=load(source/'model.pt');state=copy.deepcopy(state);actual=json.loads((source/'actual_swar/report.json').read_text())
    if sha(source/'model.pt')!=actual['model_sha256']:raise ValueError('Task regression/model identity differs')
    required=[{'id':'real_yt_yt01_005','word':'हैन','kind':'vowel_quality'},
              {'id':'real_yt_yt01_008','word':'किनकि','kind':'vowel_deletion'}]
    fallbacks=[]
    for rule in required:
        observed=next(r for r in actual['regressions'] if r['id']==rule['id'] and r['word']==rule['word'])
        if any(s['kind']==rule['kind'] and s['decision']=='strong_vowel_change' for s in observed['site_decisions']):continue
        name='paired_local_detector_v27' if rule['kind']=='vowel_quality' else 'dual_paired_detector_v28'
        path=ROOT/'reports'/name/'model.pt';old=torch.load(path,map_location='cpu',weights_only=False)
        if sha(path)!=json.loads((path.parent/'report.json').read_text())['model_sha256']:raise ValueError('Fallback checkpoint changed')
        mode=old['feature_mode']
        if name=='paired_local_detector_v27':
            if mode!='local57':raise ValueError('Frozen quality fallback shape changed')
            mode='primary57'
        head={'feature_mode':mode,'architecture':old['architecture'],'weights':old['weights'],'mean':old['mean'],'scale':old['scale'],
            'best_epoch':old['best_epoch'],'origin':'development_regression_fallback/'+name,'source_model_sha256':sha(path)}
        network=DualPairedNetwork(head['architecture'],mode);network.load_state_dict(head['weights'],strict=True);network.eval()
        xs=[];ys=[]
        for domain in ('physical','natural','indic'):
            x,y,rows,q=partition(base,domain,'development',True);mask=np.array([r['kind']==rule['kind'] for r in rows])
            if mask.sum()>=8 and len({r['word'] for r,m in zip(rows,mask) if m})>=2:xs.append(x[mask]);ys.append(y[mask])
        x=np.concatenate(xs);y=np.concatenate(ys);p=head_prob(network,head,x);negative=p[y==0].copy();negative[~supported(x)[y==0]]=-1
        high=max(.5,float(np.nextafter(np.sort(negative)[max(0,len(negative)-int(.05*len(negative))-1)],np.float32(np.inf))))
        head['operating_points']={'high':high,'low':min(.2,high),'development_negatives':len(negative)}
        state['heads'][rule['kind']]=head;fallbacks.append({'kind':rule['kind'],'reason':'Mandatory development pronunciation regression failed',
            'regression':rule,'fallback_model':name,'fallback_sha256':sha(path),'no_word_specific_runtime_rule':True})
    state['release_id']='task_vowel_release_v31';state['development_regressions_used_for_release_selection']=required
    write_group(out,{'release_policy.json':{'parent_model_sha256':sha(source/'model.pt'),'development_regressions':required,
        'fallbacks':fallbacks,'policy':'Per-cause frozen-head rollback on mandatory development regressions; operating points recalibrated on existing qualified development data, never a word-specific threshold.',
        'regressions_are_not_independent_accuracy':True},'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Morning release candidate: one checkpoint packages deletion and vowel-quality heads over shared '
        'dual phone evidence. release_policy.json explicitly records use of the two development regression '
        'labels for release rollback, not gradient training. No word-specific inference branch. model.pt '
        'selected weights/normalization/dev thresholds/pins; report.json retrospective test evidence. '
        'No encoder/audio copies or HF upload. Current consumer actual Swar scoring/cold/causal validation; '
        'owner Ampixa/user, review2026-11-09. Keep current selected model and compact evidence; inherited '
        'source/MMS research rights. Broad real Hindi-error and intonation accuracy remain unestablished.',
        ['.venv-indicf5/bin/python -m scripts.build_vowel_release'])
    torch.save(state,out/'model.pt');digest=sha(out/'model.pt');networks,loaded=load(out/'model.pt')
    report={'status':'frozen_release_pending_validation','created_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':digest,
        'model_kind':'task_vowel_v30','heads':{k:{n:h[n] for n in ('feature_mode','architecture','origin','operating_points')} for k,h in state['heads'].items()},
        'fallbacks':fallbacks,'parent_training_counts':json.loads((source/'report.json').read_text())['counts'],
        'test':{},'actual_hindi_error_accuracy':None,'development_regressions_used_for_release_selection':True}
    for domain in ('physical','natural','indic'):
        x,y,rows,q=partition(base,domain,'test',False);p=score(networks,loaded,x)
        report['test'][domain]={'retrospective_all_requests_or_physical':metrics(loaded,x,y,rows,p),
            'secondary_qualified_circular_diagnostic':metrics(loaded,x[q],y[q],[r for r,m in zip(rows,q) if m],p[q]) if q.any() else None}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if sha(out/'model.pt')!=digest:raise ValueError('Release checkpoint changed during packaging audit')
    print(json.dumps({'model_sha256':digest,'fallbacks':fallbacks,'heads':report['heads']}))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Regression-gated release packaging failed explicitly');raise
