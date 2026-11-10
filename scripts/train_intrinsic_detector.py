"""Train/dev-only query-only fitting; freeze before fresh minimal-pair test."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.evaluation import binary_metrics
from hindi_detector.intrinsic_detector import IntrinsicNetwork,decisions,score,supported
from hindi_detector.intrinsic_vowels import FEATURE_NAMES

ROOT=Path(__file__).resolve().parents[1]


def partition(base,part,only_qualified):
    report=json.loads((base/'report.json').read_text());path=base/('arrays_'+part+'.npz')
    pin=next(r for r in report['inventory'] if r['path']==path.name)
    if sha(path)!=pin['sha256'] or sha(base/'index.json')!=report['index_sha256']:
        raise ValueError('Frozen intrinsic data changed')
    rows=[r for r in json.loads((base/'index.json').read_text()) if r['split']==part]
    with np.load(path,allow_pickle=False) as values:
        x,y,q=values['features'],values['labels'],values['qualified']
    if len(x)!=len(rows) or not np.array_equal(y,np.array([r['label'] for r in rows])) or not np.isfinite(x).all():
        raise ValueError('Intrinsic matrix/index identity differs')
    if len(rows)%2 or any(a['word']!=b['word'] or (a['label'],b['label'])!=(0,1) or qa!=qb
            for a,b,qa,qb in zip(rows[::2],rows[1::2],q[::2],q[1::2],strict=True)):
        raise ValueError('Incomplete intrinsic pairs')
    for r in rows:r['dataset']=base.name
    if only_qualified:return x[q],y[q],[r for r,m in zip(rows,q) if m],q[q]
    return x,y,rows,q


def metrics(state,x,y,rows,p):
    output={};ds=np.array(decisions(state,x,p))
    for kind in ('vowel_deletion','vowel_quality'):
        mask=np.array([r['kind']==kind for r in rows])
        if not mask.any():continue
        output[kind]={'rows':int(mask.sum()),'raw_score':binary_metrics(y[mask],p[mask],state['operating_points'][kind]['high']),
            'final_gated':binary_metrics(y[mask],(ds[mask]=='strong_vowel_change').astype(float),.5),
            'paired_ordering':float(np.mean(p[mask][1::2]>p[mask][::2])),
            'uncertain_fraction':float(np.mean(ds[mask]=='uncertain'))}
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'reports/intrinsic_detector_v24')
    parser.add_argument('--additional-source',type=Path)
    args=parser.parse_args()
    out=args.out
    if out.exists():raise FileExistsError(out)
    bases=[ROOT/'reports/intrinsic_physical_v24',ROOT/'reports/intrinsic_minimal_v24']
    if args.additional_source is not None:bases.append(args.additional_source.resolve())
    write_group(out,{'protocol.json':{'feature_names':FEATURE_NAMES,'architectures':['linear','mlp16'],'seed':44,
        'selection':'Minimum equal-domain mean development BCE; physical, qualified-natural and optional qualified Indic source domains',
        'data_groups':[str(b) for b in bases],
        'thresholds':'Predeclared direction/fit gate; development-only per-kind threshold targets <=5% sham flags, minimum score0.5',
        'labels':'Physical intervention truth and secondary-model-qualified natural requests, not real Hindi-error gold',
        'test_policy':'Fresh natural test arrays first loaded after model/thresholds frozen; physical test previously evaluated'},
        'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
            for p in (Path(__file__),ROOT/'hindi_detector/intrinsic_detector.py',ROOT/'hindi_detector/intrinsic_vowels.py')]},
        'Reference-free candidate. model.pt selected head/train-only normalization/dev thresholds/dependency '
        'pins; history.json two small optimization traces; report.json physical/natural request metrics '
        'separately with realization-qualified test subset. predictions.json reserved decisions, not '
        'Hindi-error annotations. No epoch checkpoints or raw audio copy. Current consumer audio+text '
        'word detector; review2026-11-09, keep selected model and compact evidence only if consumer remains. '
        'NOT on HF; inherited acoustic research/source licences apply.',
        ['.venv-indicf5/bin/python -m scripts.train_intrinsic_detector'])
    def status(phase,**kw):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');t.replace(p)
    try:
        status('waiting_for_intrinsic_training_partitions')
        for base in bases:
            while True:
                if not (base/'status.json').exists():time.sleep(5);continue
                previous=json.loads((base/'status.json').read_text())
                if previous['phase']=='failed_explicitly':raise RuntimeError('Data preparation failed: '+base.name)
                if previous['phase'].startswith('complete_'):break
                os.kill(previous['pid'],0);time.sleep(5)
        train_parts=[partition(b,'train',True) for b in bases];dev_parts=[partition(b,'development',True) for b in bases]
        if any(len(p[0])<16 for p in train_parts+dev_parts):raise ValueError('Insufficient qualified pairs in one source/domain')
        train=np.concatenate([p[0] for p in train_parts]);y=np.concatenate([p[1] for p in train_parts]);tr=[r for p in train_parts for r in p[2]]
        dev=np.concatenate([p[0] for p in dev_parts]);dy=np.concatenate([p[1] for p in dev_parts]);dr=[r for p in dev_parts for r in p[2]]
        if any(r['word'] in {'हैन','किनकि'} for r in tr+dr):raise ValueError('Known user lexeme entered fitting')
        mean=train.mean(0);scale=np.maximum(train.std(0),.05)
        norm=lambda x:torch.from_numpy(np.clip((x-mean)/scale,-8,8).astype(np.float32))
        tx=norm(train);ty=torch.from_numpy(y.astype(np.float32));dyt=torch.from_numpy(dy.astype(np.float32));dx=norm(dev)
        counts=Counter((r['dataset'],r['word']) for r in tr);words=Counter(r['dataset'] for r in { (r['dataset'],r['word']):r for r in tr}.values())
        w=torch.tensor([1/(counts[r['dataset'],r['word']]*words[r['dataset']]) for r in tr],dtype=torch.float32);w/=w.mean()
        dev_counts=Counter(r['dataset'] for r in dr);dw=torch.tensor([1/dev_counts[r['dataset']] for r in dr],dtype=torch.float32);dw/=dw.mean()
        torch.set_num_threads(2);candidates=[];histories={}
        for architecture in ('linear','mlp16'):
            torch.manual_seed(44);network=IntrinsicNetwork(architecture);optimizer=torch.optim.AdamW(network.parameters(),lr=.01,weight_decay=.02)
            best=float('inf');stale=0;snapshot=None;history=[];best_epoch=0
            for epoch in range(500):
                network.train();optimizer.zero_grad(set_to_none=True);logits=network(tx)
                bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,ty,reduction='none')
                loss=(bce*w).mean()+.25*(torch.nn.functional.softplus(-(logits[1::2]-logits[::2]))*w[::2]).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite intrinsic objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5,error_if_nonfinite=True);optimizer.step();network.eval()
                with torch.inference_mode():value=float((torch.nn.functional.binary_cross_entropy_with_logits(network(dx),dyt,reduction='none')*dw).mean())
                history.append({'epoch':epoch+1,'train_loss':float(loss.detach()),'development_bce':value})
                if value<best-.0001:
                    best=value;stale=0;best_epoch=epoch+1;buffer=io.BytesIO();torch.save(network.state_dict(),buffer);snapshot=buffer.getvalue()
                else:stale+=1
                if stale>=40:break
            network.load_state_dict(torch.load(io.BytesIO(snapshot),weights_only=True));candidates.append((best,architecture,network.eval(),best_epoch));histories[architecture]=history
        best,architecture,network,best_epoch=min(candidates,key=lambda x:x[0])
        state={'schema':'intrinsic_vowel_site_v24','architecture':architecture,'weights':network.state_dict(),
            'feature_names':FEATURE_NAMES,'mean':mean,'scale':scale,'best_epoch':best_epoch,'operating_points':{},
            'intrinsic_code_sha256':sha(ROOT/'hindi_detector/intrinsic_vowels.py'),
            'projection_code_sha256':sha(ROOT/'hindi_detector/phone_lattice.py'),
            'site_code_sha256':sha(ROOT/'hindi_detector/pronunciation_sites.py'),
            'phone_inventory':json.loads((bases[0]/'report.json').read_text())['phone_inventory'],
            'data_report_sha256':{b.name:sha(b/'report.json') for b in bases},'actual_hindi_error_accuracy':None,
            'requires_reference_audio':False}
        dp=score(network,state,dev);available=supported(dev)
        for kind in ('vowel_deletion','vowel_quality'):
            mask=np.array([r['kind']==kind for r in dr]);negative=dp[mask&(dy==0)].copy();ng=available[mask&(dy==0)]
            if len(negative)<8:raise ValueError('Insufficient development negatives for '+kind)
            negative[~ng]=-1
            high=max(.5,float(np.nextafter(np.sort(negative)[max(0,len(negative)-int(.05*len(negative))-1)],np.float32(np.inf))))
            state['operating_points'][kind]={'high':high,'low':min(.2,high),'development_negative_count':len(negative)}
        torch.save(state,out/'model.pt');digest=sha(out/'model.pt');(out/'history.json').write_text(json.dumps(histories)+'\n')
        status('model_frozen_before_fresh_natural_test',model_sha256=digest)
        report={'status':'complete_intrinsic_candidate','model_sha256':digest,'architecture':architecture,'best_epoch':best_epoch,
            'development_bce':best,'requires_reference_audio':False,'operating_points':state['operating_points'],
            'training_rows':dict(Counter(r['dataset'] for r in tr)),'development':metrics(state,dev,dy,dr,dp),
            'test':{},'actual_hindi_error_accuracy':None,'actual_user_regressions':None,
            'limits':['Requested natural edits and acoustic realization qualification are not independently labeled real Hindi errors.',
                'Optional Indic corpus all-row labels denote source requests; only the secondary-qualified subset diagnoses realized vowel directions.',
                'Secondary acoustic model shares training-family biases; natural request test and qualified subset must both be reported.',
                'Physical test is retrospective; fresh natural test has disjoint words/voices/carriers.',
                'Legitimate Nepali vowel reductions may be flagged; no validated Hindi-specific intonation classifier.']}
        predictions=[]
        for base in bases:
            x,labels,rows,q=partition(base,'test',False);p=score(network,state,x)
            report['test'][base.name]={'all_requested_or_physical':metrics(state,x,labels,rows,p),
                'secondary_qualified':metrics(state,x[q],labels[q],[r for r,m in zip(rows,q) if m],p[q]) if q.any() else None,
                'rows':len(rows),'qualified_rows':int(q.sum())}
            for row,probability,decision in zip(rows,p,decisions(state,x,p),strict=True):
                predictions.append({**row,'controlled_change_score':float(probability),'decision':decision})
        (out/'predictions.json').write_text(json.dumps(predictions,ensure_ascii=False)+'\n');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if sha(out/'model.pt')!=digest:raise ValueError('Model changed after reserved evaluation')
        status('complete_intrinsic_candidate',model_sha256=digest)
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Intrinsic fitting failed explicitly');raise
