"""Fit temporal phone/site and prosody candidates on the corrected physical pairs."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
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
from hindi_detector.corrected_synthesis import SYSTEMS,require_pair,sha
from hindi_detector.evaluation import binary_metrics
from hindi_detector.paired_detector import PairedWordNetwork,score_arrays
from hindi_detector.structured_detector import BRANCHES,load_feature,paired_inputs,site_vector
from scripts.extract_corrected_training import read_rows

ROOT=Path(__file__).resolve().parents[1];LOG=logging.getLogger(__name__)
PARTS=('train','development','reserved_evaluation')


def complete_groups(directory):
    grouped=defaultdict(list)
    for r in read_rows(directory/'items.jsonl'):grouped[r['text_id']].append(r)
    kept={}
    for tid,rows in grouped.items():
        original=[r for r in rows if r['kind']=='original'];physical=[r for r in rows if r['kind']=='physical']
        if len(original)==4 and len(physical)==2 and all(r['status']=='ok_structured_features' for r in rows):
            if not all(r['row']['content_quality']['decision']=='pass' for r in physical):raise ValueError('Quarantined physical audio reached fit set')
            if {r['row']['condition'] for r in physical}!={'teacher_sham','f5_word_insert'}:raise ValueError('Physical pair conditions differ')
            kept[tid]=rows
    return {p:{tid:rows for tid,rows in kept.items() if rows[0]['row']['split']==p} for p in PARTS}


def samples(groups,directory,contrast_words,*,original_audit=False):
    output=[]
    for tid,receipts in groups.items():
        lookup={r['clip_id']:r for r in receipts};physical=[r for r in receipts if r['kind']=='physical']
        originals={(r['row']['system'],r['row']['speaker_id']):r for r in receipts if r['kind']=='original'}
        for receipt in physical:
            row=receipt['row'];reference_receipt=lookup[row['reference_clip_id']]
            if reference_receipt['row']['system']!=SYSTEMS[0] or reference_receipt['row']['speaker_id']==row['speaker_id']:
                raise ValueError('Reference must be teacher in the other conditioning group')
            reference=load_feature(directory,reference_receipt)
            chosen=[(receipt,row['condition'])]
            if original_audit:
                system=SYSTEMS[1] if row['source_label'] else SYSTEMS[0]
                chosen=[(originals[system,row['speaker_id']],'original_f5' if row['source_label'] else 'original_teacher')]
            for query_receipt,condition in chosen:
                query=load_feature(directory,query_receipt);p,s=paired_inputs(query,reference)
                words=query['detail']['words'];focal=query_receipt['focal_word_indices'];labels=np.zeros(len(words),np.float32)
                if row['source_label']:labels[focal]=1
                mask=np.array([i in focal or w['word'] not in contrast_words for i,w in enumerate(words)])
                if original_audit:mask[:]=False;mask[focal]=True
                output.append({'text_id':tid,'text':row['text'],'text_kind':row['text_kind'],'condition':condition,
                    'speaker_id':row['speaker_id'],'reference_speaker_id':reference_receipt['row']['speaker_id'],
                    'query_clip_id':query_receipt['clip_id'],'reference_clip_id':reference_receipt['clip_id'],
                    'phonetics':p,'prosody':s,'labels':labels,'mask':mask,'focal':focal,'words':words,
                    'site_supported':[bool(site_vector(w)[0]) for w in words]})
    return output


def flatten(rows,focal=False):
    masks=[]
    for r in rows:
        mask=r['mask'].copy()
        if focal:mask[:]=False;mask[r['focal']]=True
        masks.append(mask)
    return tuple(np.concatenate([r[k][m] for r,m in zip(rows,masks)]) for k in ('phonetics','prosody','labels'))


def matched_layout(rows):
    """Equal focal-pair weights; background controls cannot determine class prior."""
    weights=[];groups=defaultdict(lambda:{'indices':[],'focal':defaultdict(dict)});cursor=0
    for row in rows:
        indices=np.flatnonzero(row['mask']);focal=set(row['focal']);background=sum(i not in focal for i in indices)
        for i in indices:
            groups[row['text_id']]['indices'].append(cursor)
            if i in focal:
                groups[row['text_id']]['focal'][int(i)][int(row['labels'][i])]=cursor
                weights.append(1/len(focal))
            else:weights.append(.125/max(1,background))
            cursor+=1
    for group in groups.values():
        if any(set(v)!={0,1} for v in group['focal'].values()):raise ValueError('Matched focal ranking requires both source versions of each word')
    return torch.tensor(weights,dtype=torch.float32),list(groups.values())


def evaluate(network,state,rows):
    output=[]
    for r in rows:
        scores=score_arrays(network,state,r['phonetics'],r['prosody'])
        record={k:r[k] for k in ('text_id','text','text_kind','condition','speaker_id','reference_speaker_id','query_clip_id','reference_clip_id','focal')}
        record['words']=[{**{k:w[k] for k in ('word','start','end')},'source_label':int(r['labels'][i]) if r['mask'][i] else None,
            'site_supported':r['site_supported'][i],'scores':dict(zip(BRANCHES,map(float,scores[i]))),'actual_hindi_error_probability':None}
            for i,w in enumerate(r['words'])]
        output.append(record)
    metrics={}
    for branch in BRANCHES:
        focal=[(w['source_label'],w['scores'][branch],w['word']) for r in output for i,w in enumerate(r['words']) if i in r['focal']]
        y=np.array([r[0] for r in focal]);values=np.array([r[1] for r in focal]);ops=state['operating_points'][branch]
        neighbors=[w['scores'][branch] for r in output if r['condition']=='f5_word_insert' for i,w in enumerate(r['words']) if i not in r['focal'] and w['source_label'] is not None]
        groups=defaultdict(list)
        for i,row in enumerate(focal):groups[row[2]].append(i)
        keys=list(groups);rng=np.random.default_rng(45);draws=[]
        for _ in range(500):
            indices=[i for key in rng.choice(keys,len(keys),replace=True) for i in groups[key]]
            value=binary_metrics(y[indices],values[indices],ops['balanced'])['balanced_accuracy']
            if value is not None:draws.append(value)
        metrics[branch]={'focal_source':binary_metrics(y,values,ops['balanced']),
            'conservative_focal_source':binary_metrics(y,values,ops['high']),
            'operating_points':ops,'focal_lexemes':len(keys),'lexeme_cluster_balanced_accuracy_ci95':np.quantile(draws,[.025,.975]).tolist() if draws else None,
            'unchanged_neighbor_flag_rate':float(np.mean(np.array(neighbors)>=ops['balanced'])) if neighbors else None,
            'conservative_unchanged_neighbor_flag_rate':float(np.mean(np.array(neighbors)>=ops['high'])) if neighbors else None,
            'uncertain_fraction':float(np.mean((values>ops['low'])&(values<ops['high'])))}
        paired=defaultdict(dict)
        for r in output:
            for i,w in enumerate(r['words']):
                if i in r['focal']:paired[r['text_id'],i][w['source_label']]=w['scores'][branch]
        if any(set(v)!={0,1} for v in paired.values()):raise ValueError('Evaluation focal source pairs incomplete')
        deltas=[v[1]-v[0] for v in paired.values()]
        metrics[branch]['paired_source_ordering']={'pairs':len(deltas),'f5_above_teacher':sum(v>0 for v in deltas),
            'fraction':float(np.mean(np.array(deltas)>0)),'median_score_delta':float(np.median(deltas))}
    coverage={'focal_occurrences':sum(len(r['focal']) for r in output),
        'supported_site_occurrences':sum(w['site_supported'] for r in output for i,w in enumerate(r['words']) if i in r['focal'])}
    return metrics,output,coverage


def fit(directory,out,partitions,contrast_words,*,final=False,pairwise=False):
    selected={p:partitions[p] for p in (PARTS if final else PARTS[:2])}
    sets={}
    for part,groups in selected.items():
        rows=[r['row'] for g in groups.values() for r in g if r['kind']=='original']
        sets[part]={'texts':{r['text'] for r in rows},'voices':{r['speaker_id'] for r in rows},'focal_words':{w for r in rows for w in r['target_words']}}
    for a in sets:
        for b in sets:
            if a<b and any(sets[a][key]&sets[b][key] for key in sets[a]):raise ValueError('Training/development/test partition overlap')
    source_paths=[Path(__file__),ROOT/'hindi_detector/structured_detector.py',ROOT/'hindi_detector/paired_detector.py']
    write_group(out,{'inputs.json':{'features':str(directory.resolve()),'runtime_sha256':sha(directory/'runtime.json'),
        'frozen_receipts':{p:[r for g in groups.values() for r in g] for p,groups in selected.items()},
        'partition_texts':{p:len(g) for p,g in selected.items()},'partition_focal_lexemes':{p:len(s['focal_words']) for p,s in sets.items()},
        'test_policy':'reserved arrays loaded only after selected model and development thresholds written' if final else 'no reserved arrays or scores used',
        'loss_recipe':'equal matched-focal BCE,0.25 total background weight/text,0.5 within-word ranking' if pairwise else 'legacy globally class-weighted BCE'},
        'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()} for p in source_paths]},
        'Selected structured candidate. model.pt stores fresh neural weights, train-only normalization and development operating points; history.json compact optimizer trace. report.json metrics distinguish physical known-source targets from original-TTS transfer; predictions.json actual word scores/times and site support. Source labels are not pronunciation correctness. No epoch checkpoints retained. Pilot is an incremental development diagnostic; final is selected after full corrected evidence. Current consumer corrected word inference/review; keep selected models and compact metrics, review2026-11-09.',
        ['Fresh seed44; all complete corrected physical pairs in each selected partition; no old model weights.','Inputs pin every query/reference/physical waveform and structured feature identity.'])
    train=samples(partitions['train'],directory,contrast_words);dev=samples(partitions['development'],directory,contrast_words)
    p,s,y=flatten(train);dp,ds,dy=flatten(dev,True)
    if not np.isin(y,[0,1]).all() or not np.any(y==1) or not np.any(y==0):raise ValueError('Invalid corrected training labels')
    normalization={name:{'mean':v.mean(0),'scale':np.maximum(v.std(0),.01)} for name,v in [('phonetics',p),('prosody',s)]}
    convert=lambda name,v:torch.from_numpy(np.clip((v-normalization[name]['mean'])/normalization[name]['scale'],-8,8).astype(np.float32))
    tx,ts,ty=convert('phonetics',p),convert('prosody',s),torch.from_numpy(y)
    dx,ds_t,dy_t=convert('phonetics',dp),convert('prosody',ds),torch.from_numpy(dy)
    torch.manual_seed(44);torch.set_num_threads(2)
    dimensions={'phonetic_dimension':p.shape[1],'prosody_dimension':s.shape[1]};network=PairedWordNetwork(**dimensions)
    optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.02);weight=torch.tensor(float((y==0).sum()/(y==1).sum()))
    if pairwise:sample_weights,text_groups=matched_layout(train)
    best=float('inf');stale=0;history=[];checkpoint=None
    for epoch in range(120):
        network.train();losses=[]
        batches=torch.randperm(len(text_groups)).split(16) if pairwise else torch.randperm(len(y)).split(256)
        for batch in batches:
            if pairwise:
                groups=[text_groups[int(i)] for i in batch]
                indices=torch.tensor([j for g in groups for j in g['indices']],dtype=torch.long)
                reverse={int(g):i for i,g in enumerate(indices)}
                rank_pairs=[(reverse[v[0]],reverse[v[1]]) for g in groups for v in g['focal'].values()]
            else:indices=batch
            optimizer.zero_grad(set_to_none=True);logits=network(tx[indices],ts[indices])
            if pairwise:
                per_word=torch.nn.functional.binary_cross_entropy_with_logits(logits,ty[indices,None].expand_as(logits),reduction='none')
                loss=(per_word*sample_weights[indices,None]).sum()/(sample_weights[indices].sum()*len(BRANCHES))
                lo=torch.tensor([p[0] for p in rank_pairs]);hi=torch.tensor([p[1] for p in rank_pairs])
                loss=loss+.5*torch.nn.functional.softplus(-(logits[hi]-logits[lo])).mean()
            else:loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,ty[indices,None].expand_as(logits),pos_weight=weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite corrected training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        network.eval()
        with torch.inference_mode():
            logits=network(dx,ds_t);value=float(torch.nn.functional.binary_cross_entropy_with_logits(logits,dy_t[:,None].expand_as(logits)))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'development_focal_bce':value})
        if value<best-.0001:
            best=value;stale=0;best_epoch=epoch+1;buffer=io.BytesIO();torch.save(network.state_dict(),buffer);checkpoint=buffer.getvalue()
        else:stale+=1
        if stale>=15:break
    network.load_state_dict(torch.load(io.BytesIO(checkpoint),weights_only=True))
    state={'schema':'structured_word_source_v15','state_dict':network.state_dict(),'dimensions':dimensions,'normalization':normalization,
        'thresholds':dict.fromkeys(BRANCHES,.5),'operating_points':{},'best_epoch':best_epoch,
        'feature_runtime_sha256':sha(directory/'runtime.json'),'model_pair_sha256':json.loads((directory/'inputs.json').read_text())['model_pair_sha256'],
        'labels':'physically inserted IndicF5 source focal word versus24-layer teacher sham, not actual Hindi-error gold',
        'recipe':{'temporal_bins':20,'site_padding_seconds':[0.,.03,.06],'prosody_neighbor_offsets':[-2,-1,0,1,2],'hidden_embeddings':False},
        'actual_hindi_accuracy':None,'partition_sets':{p:{k:sorted(v) for k,v in s.items()} for p,s in sets.items()}}
    state['fitting_recipe']='matched_focal_and_pairwise_v16' if pairwise else 'global_class_weighting_v15'
    development=score_arrays(network,state,dp,ds)
    for i,branch in enumerate(BRANCHES):
        balanced=max((binary_metrics(dy,development[:,i],float(t))['balanced_accuracy'],-abs(t-.5),float(t)) for t in np.linspace(.05,.95,91))[2]
        negative=np.sort(development[dy==0,i]);positive=np.sort(development[dy==1,i])
        high=float(np.nextafter(negative[max(0,len(negative)-int(.1*len(negative))-1)],np.float32(np.inf)))
        low=float(np.nextafter(positive[min(len(positive)-1,int(.1*len(positive)))],np.float32(-np.inf)))
        state['operating_points'][branch]={'balanced':balanced,'high':high,'low':min(low,high)};state['thresholds'][branch]=high
    torch.save(state,out/'model.pt');model_hash=sha(out/'model.pt')
    (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    dev_metrics,_,dev_coverage=evaluate(network,state,dev)
    report={'status':'trained_corrected_structured_source_candidate','model_sha256':model_hash,'final':final,'best_epoch':best_epoch,
        'fitting_recipe':state['fitting_recipe'],
        'epochs_run':len(history),'training_loss_word_rows':len(y),'partition_texts':{p:len(g) for p,g in selected.items()},
        'partition_focal_lexemes':{p:len(s['focal_words']) for p,s in sets.items()},'conditioning_groups':{p:len(s['voices']) for p,s in sets.items()},
        'development':dev_metrics,'development_site_coverage':dev_coverage,'reserved_test':None,
        'actual_hindi_error_accuracy':None,'actual_prosody_error_accuracy':None,
        'limits':['Physical source provenance is known; actual realized phone correctness is not supplied by source identity.',
                  'No hidden speech embeddings used; phone posteriors can still retain voice/codec cues.',
                  'Exact unsupported dental/aspirated paths remain unavailable; temporal central-vowel evidence remains measured separately.',
                  'Prosody branch learns source-related local delivery shifts, not validated Hindi-specific intonation.',
                  'Shared carrier templates and source family; no unseen-generator or real-human Hindi-error accuracy claim.']}
    if final:
        test=samples(partitions['reserved_evaluation'],directory,contrast_words)
        metrics,predictions,coverage=evaluate(network,state,test)
        original=samples(partitions['reserved_evaluation'],directory,contrast_words,original_audit=True)
        original_metrics,original_predictions,original_coverage=evaluate(network,state,original)
        report.update(reserved_test=metrics,reserved_site_coverage=coverage,original_tts_transfer=original_metrics,original_site_coverage=original_coverage)
        (out/'predictions.json').write_text(json.dumps({'physical':predictions,'original':original_predictions},ensure_ascii=False,allow_nan=False)+'\n')
    if sha(out/'model.pt')!=model_hash:raise ValueError('Frozen model changed during reserved evaluation')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--features',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pairwise',action='store_true');args=parser.parse_args()
    if not (args.out/'protocol.json').is_file() or (args.out/'status.json').exists():raise ValueError('Need fresh documented training controller')
    require_pair(json.loads((ROOT/'data/teacher_indicf5_v14/model_pair.json').read_text()))
    contrast={r['word'] for r in json.loads((ROOT/'data/pocket_bulk_plan_v11/g2p_priors.json').read_text())}
    def status(phase,**extra):
        p=args.out/'status.json';temp=p.with_suffix('.partial');temp.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n');temp.replace(p)
    try:
        pilot=False
        while True:
            state=json.loads((args.features/'status.json').read_text());parts=complete_groups(args.features)
            counts={p:len(g) for p,g in parts.items()}
            if state['phase']=='failed_explicitly':raise RuntimeError('Corrected extraction failed; no old data fallback')
            if not pilot and counts['train']>=96 and counts['development']>=24:
                status('fitting_corrected_pilot_without_reserved_test',partition_texts=counts)
                report=fit(args.features,args.out/'pilot',parts,contrast,final=False,pairwise=args.pairwise);pilot=True
                status('pilot_trained_waiting_for_full_corrected_evidence',partition_texts=counts,pilot_model_sha256=report['model_sha256'])
            if state['phase']=='complete_corrected_structured_evidence':
                if min(counts.values())<16:raise RuntimeError('Insufficient complete corrected controls in a partition')
                status('fitting_final_corrected_candidate',partition_texts=counts)
                report=fit(args.features,args.out/'final',parts,contrast,final=True,pairwise=args.pairwise)
                status('complete_corrected_training_and_reserved_evaluation',partition_texts=counts,model_sha256=report['model_sha256']);break
            status('pilot_trained_waiting_for_full_corrected_evidence' if pilot else 'waiting_for_corrected_training_controls',partition_texts=counts)
            os.kill(state['pid'],0);time.sleep(10)
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        LOG.exception('Corrected training stopped explicitly');raise
