"""Per-cause dev-only selection on repaired features, with frozen baselines."""
from __future__ import annotations

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
from hindi_detector.dual_paired_detector import DualPairedNetwork,select_inputs
from hindi_detector.dual_paired_vowels import FEATURE_NAMES,supported
from hindi_detector.evaluation import binary_metrics
from hindi_detector.task_vowel_detector import decisions,score
from scripts.train_paired_local import partition

ROOT=Path(__file__).resolve().parents[1]


def head_prob(network,head,x):
    values=select_inputs(x,head['feature_mode']);values=np.clip((values-head['mean'])/head['scale'],-8,8).astype(np.float32)
    with torch.inference_mode():return torch.sigmoid(network(torch.from_numpy(values.copy()))).cpu().numpy()


def metrics(state,x,y,rows,p):
    ds=np.array(decisions(state,x,p));result={}
    for kind in ('vowel_deletion','vowel_quality'):
        mask=np.array([r['kind']==kind for r in rows])
        if mask.any():result[kind]={'final_gated_decision':binary_metrics(y[mask],(ds[mask]=='strong_vowel_change').astype(float),.5),
            'paired_ordering':float(np.mean(p[mask][1::2]>p[mask][::2])),'uncertain_fraction':float(np.mean(ds[mask]=='uncertain'))}
    return result


def main():
    out=ROOT/'reports/task_vowel_detector_v30';base=ROOT/'reports/dual_geometry_features_v30'
    if out.exists():raise FileExistsError(out)
    write_group(out,{'protocol.json':{'tasks':['vowel_deletion','vowel_quality'],'modes':['primary57','secondary57','dual114'],
        'architectures':['linear','mlp16'],'frozen_candidates':['paired_local_detector_v27','dual_paired_detector_v28'],
        'selection':'Per kind minimum equal-domain qualified development BCE, no actual Swar predictions or held-out scores used in ranking.',
        'domain_floor':'At least4 qualified development pairs and2 development words per kind/domain; training domains need at least4 pairs.',
        'threshold':'Per-kind selected head, development-only5% gated-negative target, high score minimum0.5',
        'labels':'Known physical changes and acoustic-qualified natural/source proxies; latter are not independent of model features.',
        'test_scope':'Parent control/request tests retrospective after development iterations. Fresh waveform interventions evaluated only after freeze.'},
        'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
            for p in (Path(__file__),ROOT/'hindi_detector/task_vowel_detector.py',ROOT/'hindi_detector/dual_paired_vowels.py')]},
        'Task-specific two-head candidate on corrected control coordinates. model.pt selected per-cause '
        'weights/train-only normalization/development thresholds/source pins; selection.json all dev BCEs; '
        'history.json optimizer traces only, no epoch checkpoints. report.json separates retrospective '
        'physical truth and request/proxy diagnostics. No encoder/audio copy, NOT on HF. Current consumer '
        'morning word detector, review2026-11-09; retain selected artifact/compact evidence while active. '
        'Inherited source/MMS research rights; real Hindi-error and intonation accuracy remain unknown.',
        ['.venv-indicf5/bin/python -m scripts.train_task_vowels'])
    def status(phase,**kw):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');t.replace(p)
    try:
        status('waiting_for_corrected_control_coordinates')
        while True:
            previous=json.loads((base/'status.json').read_text())
            if previous['phase']=='failed_explicitly':raise RuntimeError('Coordinate repair failed')
            if previous['phase']=='complete_corrected_dual_geometry':break
            os.kill(previous['pid'],0);time.sleep(5)
        train_parts=[partition(base,d,'train',True) for d in ('physical','natural','indic')]
        dev_parts=[partition(base,d,'development',True) for d in ('physical','natural','indic')]
        histories={};selections={};heads={};networks={};counts={};torch.set_num_threads(2)
        for kind in ('vowel_deletion','vowel_quality'):
            def select(parts,minimum):
                xs=[];ys=[];rows=[];domains=[]
                for part in parts:
                    mask=np.array([r['kind']==kind for r in part[2]]);subset=[r for r,m in zip(part[2],mask) if m]
                    if mask.sum()<minimum or len({r['word'] for r in subset})<2:continue
                    xs.append(part[0][mask]);ys.append(part[1][mask]);rows.extend(subset);domains.append(subset[0]['domain'])
                if len(xs)<2:raise ValueError('Insufficient physical/natural domain support for '+kind)
                return np.concatenate(xs),np.concatenate(ys),rows,domains
            x,y,rows,domains=select(train_parts,8);dx,dy,dr,ddomains=select(dev_parts,8)
            if any(r['word'] in {'हैन','किनकि'} for r in rows+dr):raise ValueError('Known user word entered per-task optimization')
            wc=Counter((r['domain'],r['word']) for r in rows);nw=Counter(k[0] for k in wc)
            w=torch.tensor([1/(wc[r['domain'],r['word']]*nw[r['domain']]) for r in rows],dtype=torch.float32);w/=w.mean()
            dc=Counter(r['domain'] for r in dr);dw=torch.tensor([1/dc[r['domain']] for r in dr],dtype=torch.float32);dw/=dw.mean()
            ty=torch.from_numpy(y.astype(np.float32));dyt=torch.from_numpy(dy.astype(np.float32));candidates=[]
            for mode in ('primary57','secondary57','dual114'):
                values=select_inputs(x,mode);dvalues=select_inputs(dx,mode);mean=values.mean(0);scale=np.maximum(values.std(0),.05)
                tx=torch.from_numpy(np.clip((values-mean)/scale,-8,8).astype(np.float32).copy());dt=torch.from_numpy(np.clip((dvalues-mean)/scale,-8,8).astype(np.float32).copy())
                for architecture in ('linear','mlp16'):
                    torch.manual_seed(44);network=DualPairedNetwork(architecture,mode);optimizer=torch.optim.AdamW(network.parameters(),lr=.01,weight_decay=.02)
                    best=float('inf');stale=0;snapshot=None;history=[];best_epoch=0
                    for epoch in range(500):
                        network.train();optimizer.zero_grad(set_to_none=True);logits=network(tx)
                        loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,ty,reduction='none')*w).mean()+.25*(torch.nn.functional.softplus(-(logits[1::2]-logits[::2]))*w[::2]).mean()
                        if not torch.isfinite(loss):raise ValueError('Nonfinite per-task objective')
                        loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5,error_if_nonfinite=True);optimizer.step();network.eval()
                        with torch.inference_mode():value=float((torch.nn.functional.binary_cross_entropy_with_logits(network(dt),dyt,reduction='none')*dw).mean())
                        history.append({'epoch':epoch+1,'loss':float(loss.detach()),'development_bce':value})
                        if value<best-.0001:
                            best=value;stale=0;best_epoch=epoch+1;buffer=io.BytesIO();torch.save(network.state_dict(),buffer);snapshot=buffer.getvalue()
                        else:stale+=1
                        if stale>=40:break
                    network.load_state_dict(torch.load(io.BytesIO(snapshot),weights_only=True));head={'feature_mode':mode,'architecture':architecture,
                        'weights':network.state_dict(),'mean':mean,'scale':scale,'best_epoch':best_epoch,'origin':'task_specific_corrected_fit'}
                    candidates.append((best,mode+'/'+architecture,network.eval(),head));histories[kind+'/'+mode+'/'+architecture]=history
            for name in ('paired_local_detector_v27','dual_paired_detector_v28'):
                path=ROOT/'reports'/name/'model.pt';old=torch.load(path,map_location='cpu',weights_only=False)
                if sha(path)!=json.loads((path.parent/'report.json').read_text())['model_sha256']:raise ValueError('Frozen baseline checkpoint changed')
                mode=old['feature_mode']
                if name=='paired_local_detector_v27':
                    if mode!='local57':raise ValueError('Expected verified frozen57 baseline')
                    mode='primary57'
                head={'feature_mode':mode,'architecture':old['architecture'],'weights':old['weights'],'mean':old['mean'],'scale':old['scale'],
                    'best_epoch':old['best_epoch'],'origin':name,'source_model_sha256':sha(path)}
                network=DualPairedNetwork(head['architecture'],mode);network.load_state_dict(head['weights'],strict=True);network.eval()
                p=np.clip(head_prob(network,head,dx).astype(np.float64),1e-7,1-1e-7)
                loss=float(np.mean((-dy*np.log(p)-(1-dy)*np.log1p(-p))*dw.numpy()))
                candidates.append((loss,'frozen/'+name,network,head))
            best,name,network,head=min(candidates,key=lambda c:(c[0],c[1]));p=head_prob(network,head,dx);eligible=supported(dx)
            negative=p[dy==0].copy();negative[~eligible[dy==0]]=-1
            high=max(.5,float(np.nextafter(np.sort(negative)[max(0,len(negative)-int(.05*len(negative))-1)],np.float32(np.inf))))
            head['operating_points']={'high':high,'low':min(.2,high),'development_negatives':len(negative)}
            heads[kind]=head;networks[kind]=network;selections[kind]={'selected':name,'development_bce':best,
                'candidates':[{'name':n,'development_bce':v} for v,n,_,_ in candidates]};counts[kind]={'training_pairs':len(rows)//2,'development_pairs':len(dr)//2,'training_domains':domains,'development_domains':ddomains}
        state={'schema':'task_vowel_v30','heads':heads,'feature_names':FEATURE_NAMES,'data_report_sha256':sha(base/'report.json'),
            'phone_inventory':json.loads((base/'report.json').read_text())['phone_inventory'],
            'secondary_phone_inventory':json.loads((base/'report.json').read_text())['secondary_phone_inventory'],
            'code_sha256':{p:sha(ROOT/p) for p in ['hindi_detector/task_vowel_detector.py','hindi_detector/dual_paired_vowels.py','hindi_detector/paired_local_vowels.py','hindi_detector/intrinsic_vowels.py','hindi_detector/phone_lattice.py','hindi_detector/pronunciation_sites.py']},'actual_hindi_error_accuracy':None}
        torch.save(state,out/'model.pt');digest=sha(out/'model.pt');(out/'selection.json').write_text(json.dumps(selections,indent=2)+'\n');(out/'history.json').write_text(json.dumps(histories)+'\n')
        status('model_frozen_before_waveform_intervention_audit',model_sha256=digest)
        report={'status':'complete_frozen_task_vowels','model_sha256':digest,'counts':counts,'selected_heads':{k:{n:h[n] for n in ('feature_mode','architecture','origin','operating_points')} for k,h in heads.items()},
            'test':{},'actual_hindi_error_accuracy':None,'limits':['Parent tests are retrospective; natural/source acoustic-qualified labels are not independent of model features.',
                'Fresh actual-waveform edit audit is separate and run after freeze.','No independently labeled general Hindi-error or Hindi-intonation accuracy is established.']}
        for domain in ('physical','natural','indic'):
            x,y,rows,q=partition(base,domain,'test',False);p=score(networks,state,x)
            report['test'][domain]={'all_requests_or_physical':metrics(state,x,y,rows,p),'secondary_qualified_diagnostic':metrics(state,x[q],y[q],[r for r,m in zip(rows,q) if m],p[q]) if q.any() else None}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if sha(out/'model.pt')!=digest:raise ValueError('Task checkpoint changed after retrospective evaluation')
        status('complete_frozen_task_vowels',model_sha256=digest)
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Task-specific fitting failed explicitly');raise
