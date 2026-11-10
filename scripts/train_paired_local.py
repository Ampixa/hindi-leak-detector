"""Freeze a stronger paired head before consuming the new internal-vowel test."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
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
from hindi_detector.dual_paired_detector import (DualPairedNetwork,decisions as dual_decisions,
    score as dual_score,select_inputs)
from hindi_detector.dual_paired_vowels import FEATURE_NAMES as DUAL_NAMES,supported as dual_supported
from hindi_detector.evaluation import binary_metrics
from hindi_detector.paired_local_detector import PairedLocalNetwork,decisions,score
from hindi_detector.paired_local_vowels import FEATURE_NAMES,supported

ROOT=Path(__file__).resolve().parents[1]


def partition(base,domain,part,qualified_only):
    report=json.loads((base/'report.json').read_text());path=base/('arrays_'+domain+'_'+part+'.npz')
    pin=next(r for r in report['inventory'] if r['path']==path.name)
    if sha(path)!=pin['sha256'] or sha(base/'index.json')!=report['index_sha256']:raise ValueError('Local training partition changed')
    rows=[r for r in json.loads((base/'index.json').read_text()) if r['domain']==domain and r['split']==part]
    with np.load(path,allow_pickle=False) as values:x,y,q=values['features'],values['labels'],values['qualified']
    if len(rows)!=len(x) or not np.array_equal(y,np.array([r['label'] for r in rows])) or not np.isfinite(x).all():raise ValueError('Local data/index identity differs')
    if len(rows)%2 or any((a['label'],b['label'])!=(0,1) or a['word']!=b['word'] or qa!=qb
        for a,b,qa,qb in zip(rows[::2],rows[1::2],q[::2],q[1::2],strict=True)):raise ValueError('Local partition has incomplete pairs')
    if qualified_only:return x[q],y[q],[r for r,m in zip(rows,q) if m],q[q]
    return x,y,rows,q


def metrics(state,x,y,rows,p):
    decide=dual_decisions if state['schema']=='dual_paired_vowel_v28' else decisions
    ds=np.array(decide(state,x,p));result={}
    for kind in ('vowel_deletion','vowel_quality'):
        mask=np.array([r['kind']==kind for r in rows])
        if not mask.any():continue
        result[kind]={'raw_score':binary_metrics(y[mask],p[mask],state['operating_points'][kind]['high']),
            'final_gated_decision':binary_metrics(y[mask],(ds[mask]=='strong_vowel_change').astype(float),.5),
            'paired_ordering':float(np.mean(p[mask][1::2]>p[mask][::2])),
            'uncertain_fraction':float(np.mean(ds[mask]=='uncertain'))}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dual',action='store_true');args=parser.parse_args()
    out=ROOT/('reports/dual_paired_detector_v28' if args.dual else 'reports/paired_local_detector_v27')
    base=ROOT/('reports/dual_paired_features_v28' if args.dual else 'reports/paired_local_features_v27')
    modes=('primary57','secondary57','dual114') if args.dual else ('margins18','local57')
    names=DUAL_NAMES if args.dual else FEATURE_NAMES
    network_class=DualPairedNetwork if args.dual else PairedLocalNetwork
    score_fn=dual_score if args.dual else score;decide=dual_decisions if args.dual else decisions
    support_fn=dual_supported if args.dual else supported
    if out.exists():raise FileExistsError(out)
    write_group(out,{'protocol.json':{'feature_modes':modes,'architectures':['linear','mlp16'],'seed':44,
        'feature_names':names,'dual_acoustic_measurements':args.dual,
        'selection':'Minimum equal-domain qualified development BCE. Domains with fewer than8 qualified dev pairs or4 dev words excluded from selection/thresholds; their qualified training rows may still train.',
        'normalization':'Train only; no speaker/generator/word ID features',
        'gates':('Both references favor/observe expected vowel, both relative margins positive at all3 pads, both alternative fits>-1.5; quality alternative vowel observed by both and wins in at least one; deletion local gap anchored in both and expected-vowel emission falls in at least one. No absolute deletion-path preference required.' if args.dual else 'Expected vowel observed in reference, alternative query path consistently favored relative to reference, fit>-1.5, quality alternative vowel actually observed; no all-consonant veto'),
        'thresholds':'Kind-specific development-only <=5% gated-negative target; high score minimum0.5',
        'fresh_test':('All parent tests retrospective. Separate minimal_audit_v29 excluded from fitting and this script; evaluated only after freeze.' if args.dual else 'min26 internal deletion/AI-quality words are untouched by fitting and tested only after freeze'),
        'labels':'Known physical changes plus secondary-probe-qualified natural/source vowel pairs, not independent real Hindi-error gold'},
        'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
            for p in (Path(__file__),ROOT/'hindi_detector/paired_local_detector.py',ROOT/'hindi_detector/paired_local_vowels.py')]},
        'Stronger paired vowel candidate. model.pt chosen weights/train-only normalization/dev thresholds '
        'and dependency pins, history.json four optimizer histories without epoch checkpoints, report.json '
        'separates physical truth/natural requests/source requests and secondary-qualified subsets; fresh '
        'internal test separately named. predictions.json raw/final reserved decisions. No new raw audio '
        'or encoder-weight copy. Current consumer improved actual-word detector; review2026-11-09, retain '
        'selected model/compact provenance, retire unselected diagnostic state24h after consumer closes. '
        'NOT on HF; inherited source/MMS research licences. Prosody correctness is not established.',
        ['.venv-indicf5/bin/python -m scripts.train_paired_local'])
    def status(phase,**kw):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');t.replace(p)
    try:
        status('waiting_for_paired_local_data')
        while True:
            previous=json.loads((base/'status.json').read_text())
            if previous['phase']=='failed_explicitly':raise RuntimeError('Paired local producer failed')
            if previous['phase']==('complete_dual_paired_matrices' if args.dual else 'complete_paired_local_matrices'):break
            os.kill(previous['pid'],0);time.sleep(5)
        train_parts=[];dev_parts=[];domain_status={}
        for domain in ('physical','natural','indic'):
            train=partition(base,domain,'train',True);dev=partition(base,domain,'development',True)
            if len(train[0])>=16:train_parts.append(train)
            eligible=len(dev[0])>=16 and len({r['word'] for r in dev[2]})>=4
            if eligible:dev_parts.append(dev)
            domain_status[domain]={'training_rows':len(train[0]),'development_rows':len(dev[0]),'development_in_selection':eligible}
        if len(train_parts)<2 or len(dev_parts)<2:raise ValueError('Insufficient independently qualified training/dev domains')
        train=np.concatenate([p[0] for p in train_parts]);y=np.concatenate([p[1] for p in train_parts]);tr=[r for p in train_parts for r in p[2]]
        dev=np.concatenate([p[0] for p in dev_parts]);dy=np.concatenate([p[1] for p in dev_parts]);dr=[r for p in dev_parts for r in p[2]]
        if any(r['word'] in {'हैन','किनकि'} for r in tr+dr):raise ValueError('Known user lexeme entered optimization')
        mean=train.mean(0);scale=np.maximum(train.std(0),.05)
        norm=lambda x:np.clip((x-mean)/scale,-8,8).astype(np.float32)
        counts=Counter((r['domain'],r['word']) for r in tr);nwords=Counter(k[0] for k in counts)
        w=torch.tensor([1/(counts[r['domain'],r['word']]*nwords[r['domain']]) for r in tr],dtype=torch.float32);w/=w.mean()
        dcounts=Counter(r['domain'] for r in dr);dw=torch.tensor([1/dcounts[r['domain']] for r in dr],dtype=torch.float32);dw/=dw.mean()
        ty=torch.from_numpy(y.astype(np.float32));dyt=torch.from_numpy(dy.astype(np.float32));torch.set_num_threads(2)
        candidates=[];histories={}
        for mode in modes:
            tx=norm(train);dx=norm(dev)
            if args.dual:tx=select_inputs(tx,mode);dx=select_inputs(dx,mode)
            elif mode=='margins18':tx=tx[:,:18];dx=dx[:,:18]
            tx=torch.from_numpy(tx.copy());dx=torch.from_numpy(dx.copy())
            for architecture in ('linear','mlp16'):
                torch.manual_seed(44);network=network_class(architecture,mode);optimizer=torch.optim.AdamW(network.parameters(),lr=.01,weight_decay=.02)
                best=float('inf');stale=0;snapshot=None;best_epoch=0;history=[]
                for epoch in range(500):
                    network.train();optimizer.zero_grad(set_to_none=True);logits=network(tx)
                    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,ty,reduction='none')
                    loss=(bce*w).mean()+.25*(torch.nn.functional.softplus(-(logits[1::2]-logits[::2]))*w[::2]).mean()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite paired local loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5,error_if_nonfinite=True);optimizer.step();network.eval()
                    with torch.inference_mode():value=float((torch.nn.functional.binary_cross_entropy_with_logits(network(dx),dyt,reduction='none')*dw).mean())
                    history.append({'epoch':epoch+1,'train_loss':float(loss.detach()),'development_bce':value})
                    if value<best-.0001:
                        best=value;stale=0;best_epoch=epoch+1;buffer=io.BytesIO();torch.save(network.state_dict(),buffer);snapshot=buffer.getvalue()
                    else:stale+=1
                    if stale>=40:break
                network.load_state_dict(torch.load(io.BytesIO(snapshot),weights_only=True));candidates.append((best,mode,architecture,network.eval(),best_epoch));histories[mode+'/'+architecture]=history
        best,mode,architecture,network,best_epoch=min(candidates,key=lambda x:x[0])
        state={'schema':'dual_paired_vowel_v28' if args.dual else 'paired_local_vowel_v27','feature_mode':mode,'architecture':architecture,'weights':network.state_dict(),
            'feature_names':names,'mean':mean,'scale':scale,'best_epoch':best_epoch,'operating_points':{},
            'data_report_sha256':sha(base/'report.json'),'actual_hindi_error_accuracy':None,
            'phone_inventory':json.loads((ROOT/'reports/swar_corpus_features_v18/runtime.json').read_text())['phone'],
            'code_sha256':{p:sha(ROOT/p) for p in ['hindi_detector/paired_local_vowels.py','hindi_detector/intrinsic_vowels.py','hindi_detector/phone_lattice.py','hindi_detector/pronunciation_sites.py']}}
        if args.dual:
            state['secondary_phone_inventory']=json.loads((ROOT/'reports/swar_phone_crosscheck_v20/runtime.json').read_text())['phone']
            state['code_sha256']['hindi_detector/dual_paired_vowels.py']=sha(ROOT/'hindi_detector/dual_paired_vowels.py')
        dp=score_fn(network,state,dev);available=support_fn(dev)
        for kind in ('vowel_deletion','vowel_quality'):
            mask=np.array([r['kind']==kind for r in dr]);negative=dp[mask&(dy==0)].copy();negative[~available[mask&(dy==0)]]=-1
            if len(negative)<8:raise ValueError('Insufficient dev negatives for '+kind)
            high=max(.5,float(np.nextafter(np.sort(negative)[max(0,len(negative)-int(.05*len(negative))-1)],np.float32(np.inf))))
            state['operating_points'][kind]={'high':high,'low':min(.2,high),'development_negatives':len(negative)}
        torch.save(state,out/'model.pt');digest=sha(out/'model.pt');(out/'history.json').write_text(json.dumps(histories)+'\n')
        status('model_frozen_before_new_internal_test',model_sha256=digest)
        report={'status':'complete_frozen_paired_local_candidate','model_sha256':digest,'feature_mode':mode,'architecture':architecture,
            'best_epoch':best_epoch,'development_bce':best,'domains':domain_status,'operating_points':state['operating_points'],
            'training_pairs':len(tr)//2,'development_pairs':len(dr)//2,'development':metrics(state,dev,dy,dr,dp),'test':{},
            'actual_hindi_error_accuracy':None,'actual_user_regression_accuracy':None,'limits':[
                'Controlled physical and acoustic-qualified natural/source labels do not establish independent real Hindi-error accuracy.',
                'Previous physical/min23/source tests are retrospective for this development cycle; min26 test is fresh.',
                'Other-voice native references help separate vowel changes from identity but do not guarantee native acceptability.',
                'Hindi-specific intonation correctness remains unvalidated.']}
        fresh={j['id'] for j in json.loads((ROOT/'data/minimal_internal_v26/selection.json').read_text())['jobs']};predictions=[]
        for domain in ('physical','natural','indic'):
            x,labels,rows,q=partition(base,domain,'test',False);p=score_fn(network,state,x)
            report['test'][domain]={'all_requests_or_physical':metrics(state,x,labels,rows,p),
                'secondary_qualified':metrics(state,x[q],labels[q],[r for r,m in zip(rows,q) if m],p[q]) if q.any() else None,
                'rows':len(rows),'qualified_rows':int(q.sum())}
            mask=np.array([r['key'] in fresh for r in rows]);qualified_mask=mask&q
            if mask.any():report['fresh_internal_test']={'all_requests':metrics(state,x[mask],labels[mask],[r for r,m in zip(rows,mask) if m],p[mask]),
                'secondary_qualified':metrics(state,x[qualified_mask],labels[qualified_mask],[r for r,m in zip(rows,qualified_mask) if m],p[qualified_mask]) if qualified_mask.any() else None,
                'pairs':int(mask.sum())//2,'qualified_pairs':int(qualified_mask.sum())//2}
            for row,probability,decision in zip(rows,p,decide(state,x,p),strict=True):predictions.append({**row,'controlled_change_score':float(probability),'decision':decision})
        (out/'predictions.json').write_text(json.dumps(predictions,ensure_ascii=False)+'\n');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if sha(out/'model.pt')!=digest:raise ValueError('Paired local model changed after held-out use')
        status('complete_frozen_paired_local_candidate',model_sha256=digest)
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Paired local fitting failed explicitly');raise
