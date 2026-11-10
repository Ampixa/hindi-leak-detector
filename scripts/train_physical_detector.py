"""Follow-up training on actual waveform word inserts and matching sham joins."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import binary_metrics
from hindi_detector.paired_detector import BRANCHES, PairedWordNetwork, load_feature, load_model, paired_features, score_arrays, sha
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2); torch.manual_seed(44)
    inputs = json.loads((args.base / 'inputs.json').read_text()); directory = Path(inputs['features'])
    _, base = load_model(args.base / 'model.pt'); projection = base['projection']
    grouped = {}
    for part, receipts in inputs['frozen_receipts'].items():
        groups = defaultdict(list)
        for r in receipts:
            groups[r['text_id']].append(r)
        values = list(groups.items())
        grouped[part] = values[:192] if part == 'train' else values[:32] if part == 'development' else values[20:84]
    contrast = {r['word'] for r in json.loads((ROOT / 'data/pocket_bulk_plan_v11/g2p_priors.json').read_text())}
    write_group(args.out, {'inputs.json': {'base': str(args.base.resolve()), 'base_inputs_sha256': sha(args.base / 'inputs.json'),
        'fixed_text_ids': {p: [tid for tid, _ in g] for p, g in grouped.items()},
        'test_excludes_first_20_physical_audit_texts': True, 'training_recipe': 'same branch architecture and optimization; actual5ms-faded word edits replace feature-only training'},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in (Path(__file__), ROOT / 'hindi_detector/paired_detector.py', ROOT / 'hindi_detector/word_evidence.py')]},
        'Physical-word follow-up candidate. Generated controls/*.wav are active training/evaluation inputs with hashes/bytes in journals; features/*.npz retains compact paired word inputs; items.jsonl records per-wave source, labels, timing. model.pt selected final checkpoint, history.json train/dev trace, report.json/predictions.json held-out internal outcomes. No human accent gold. Keep selected model/compact metrics; review waves/features within24h after consumer completes and remove if superseded.',
        ['.venv-indicf5/bin/python -m scripts.train_physical_detector --base reports/paired_detector_pilot_v11 --out NEW', 'Fixed existing split; one query conditioning group/text and different native reference; first20 prior physical-test texts excluded. No new model weights downloaded.'])
    (args.out / 'controls').mkdir(); (args.out / 'features').mkdir()
    engine = WordEvidenceEngine(ROOT, device='mps', pitch_method='dio', hidden_layers=(12, 24))
    total = 0
    def status(phase, **extra):
        p = args.out / 'status.json'; temp = p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(), 'encoded_controls': total, **extra}, indent=2)); temp.replace(p)
    def extract(part):
        nonlocal total
        result = []
        for tid, receipts in grouped[part]:
            loaded = {(r['row']['system'], r['row']['speaker_id']): load_feature(directory, r) for r in receipts}
            voices = sorted({r['row']['speaker_id'] for r in receipts}); voice = voices[0]
            native = loaded['pocket_nepali_v1c', voice]; hindi = loaded['pocket_hindi_parent', voice]; reference = loaded['pocket_nepali_v1c', voices[1]]
            focal = native['receipt']['focal_word_indices']; text = native['receipt']['row']['text']
            waves = {}
            for name, item in [('native', native), ('hindi', hindi)]:
                row = item['receipt']['row']; waves[name] = audio(Path(row['source_pack']), row)
            for label, donor in enumerate((native, hindi)):
                samples = waves['native'].copy()
                for i in sorted(focal, reverse=True):
                    n, h = native['detail']['words'][i], donor['detail']['words'][i]
                    start, end = int(n['start']*16000), int(n['end']*16000)
                    chunk = waves['hindi' if label else 'native'][int(h['start']*16000):int(h['end']*16000)].copy()
                    width = min(80, len(chunk)//4, start, len(samples)-end)
                    if width < 2:
                        raise ValueError('Insufficient physical boundary support')
                    samples[start-width:start] *= np.linspace(1,0,width); samples[end:end+width] *= np.linspace(0,1,width)
                    chunk[:width] *= np.linspace(0,1,width); chunk[-width:] *= np.linspace(1,0,width)
                    samples = np.concatenate([samples[:start],chunk,samples[end:]])
                wav = args.out/'controls'/f'{tid}_{label}.wav'; sf.write(wav,samples,16000,subtype='PCM_16')
                samples, rate = sf.read(wav,dtype='float32')
                if rate != 16000:
                    raise ValueError('Physical PCM rate differs')
                detail, arrays = engine.analyze(samples,text); p,s = paired_features({'detail':detail,**arrays},reference,projection)
                labels = np.zeros(len(p),np.float32); labels[focal] = label
                mask = np.array([i in focal or w['word'] not in contrast for i,w in enumerate(detail['words'])])
                fp = args.out/'features'/f'{tid}_{label}.npz'; np.savez_compressed(fp,phonetics=p,prosody=s,labels=labels,mask=mask)
                record = {'text_id':tid,'split':part,'text':text,'text_kind':native['receipt']['row']['text_kind'], 'condition':'hindi_word_insert' if label else 'native_sham_rejoin',
                    'speaker_id':voice,'reference_speaker_id':voices[1], 'audio':str(wav.relative_to(args.out)), 'sha256':sha(wav),'bytes':wav.stat().st_size,
                    'features':str(fp.relative_to(args.out)),'features_sha256':sha(fp),'feature_bytes':fp.stat().st_size,
                    'native_clip_id':native['receipt']['clip_id'],'hindi_clip_id':hindi['receipt']['clip_id'],'reference_clip_id':reference['receipt']['clip_id'],
                    'words':detail['words'],'focal':focal}
                with (args.out/'items.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                result.append({**record,'phonetics':p,'prosody':s,'labels':labels,'mask':mask}); total += 1
                status('encoding_physical_'+part,current_text=tid); LOG.info('Physical training controls %s',total)
        return result
    train=extract('train'); dev=extract('development')
    def flatten(rows,focal=False):
        masks=[]
        for r in rows:
            m=r['mask'].copy()
            if focal:m[:]=False;m[r['focal']]=True
            masks.append(m)
        return tuple(np.concatenate([r[k][m] for r,m in zip(rows,masks)]) for k in ('phonetics','prosody','labels'))
    p,s,y=flatten(train); dp,ds,dy=flatten(dev,True)
    norm={k:{'mean':x.mean(0),'scale':np.maximum(x.std(0),.01)} for k,x in [('phonetics',p),('prosody',s)]}
    convert=lambda k,x:torch.from_numpy(np.clip((x-norm[k]['mean'])/norm[k]['scale'],-8,8).astype(np.float32))
    tx,ts,ty=convert('phonetics',p),convert('prosody',s),torch.from_numpy(y)
    dx,ds_t,dy_t=convert('phonetics',dp),convert('prosody',ds),torch.from_numpy(dy)
    dims={'phonetic_dimension':p.shape[1],'prosody_dimension':s.shape[1]}; network=PairedWordNetwork(**dims)
    optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.02); weight=torch.tensor(float((y==0).sum()/(y==1).sum()))
    best=float('inf'); stale=0; history=[]; checkpoint=None
    for epoch in range(120):
        network.train(); losses=[]
        for indices in torch.randperm(len(y)).split(256):
            optimizer.zero_grad(set_to_none=True); logits=network(tx[indices],ts[indices])
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,ty[indices,None].expand_as(logits),pos_weight=weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite physical fit loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(),5.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        network.eval()
        with torch.inference_mode():
            logits=network(dx,ds_t); value=float(torch.nn.functional.binary_cross_entropy_with_logits(logits,dy_t[:,None].expand_as(logits)))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'development_focal_bce':value})
        if value < best-.0001:
            best=value;stale=0;best_epoch=epoch+1;buffer=io.BytesIO();torch.save(network.state_dict(),buffer);checkpoint=buffer.getvalue()
        else:stale+=1
        if stale>=15:break
    network.load_state_dict(torch.load(io.BytesIO(checkpoint),weights_only=True))
    state={'schema':'paired_word_source_candidate_v11','state_dict':network.state_dict(),'dimensions':dims,'normalization':norm,'projection':projection,
        'thresholds':dict.fromkeys(BRANCHES,.5),'branches':BRANCHES,'labels':'physically inserted Hindi-source focal word vs matching native sham; not pronunciation gold',
        'actual_hindi_accuracy':None,'best_epoch':best_epoch,'feature_runtime_sha256':base['feature_runtime_sha256']}
    dev_scores=score_arrays(network,state,dp,ds)
    conservative={}
    for i,b in enumerate(BRANCHES):
        choices=[(binary_metrics(dy,dev_scores[:,i],float(t))['balanced_accuracy'],-abs(t-.5),float(t)) for t in np.linspace(.05,.95,91)]
        state['thresholds'][b]=max(choices)[2]
        native=np.sort(dev_scores[dy==0,i]);conservative[b]=float(np.nextafter(native[max(0,len(native)-int(.1*len(native))-1)],np.float32(np.inf)))
    torch.save(state,args.out/'model.pt');model_hash=sha(args.out/'model.pt')
    (args.out/'history.json').write_text(json.dumps(history,indent=2)); status('frozen_before_physical_test_generation',model_sha256=model_hash)
    test=extract('reserved_evaluation');tp,ss,yy=flatten(test,True);scores=score_arrays(network,state,tp,ss);metrics={}
    predictions=[]
    for r in test:
        values=score_arrays(network,state,r['phonetics'],r['prosody'])
        predictions.append({k:r[k] for k in ('text_id','condition','text','audio','sha256','bytes','speaker_id','reference_speaker_id','focal')})
        predictions[-1]['words']=[{'word':w['word'],'start':w['start'],'end':w['end'],'source_label':int(r['labels'][i]) if r['mask'][i] else None,'scores':dict(zip(BRANCHES,map(float,values[i])))} for i,w in enumerate(r['words'])]
    for i,b in enumerate(BRANCHES):
        neighbors=[w['scores'][b] for r in predictions if r['condition']=='hindi_word_insert' for j,w in enumerate(r['words']) if j not in r['focal'] and w['source_label'] is not None]
        metrics[b]={'focal':binary_metrics(yy,scores[:,i],state['thresholds'][b]),'threshold':state['thresholds'][b],
            'conservative':binary_metrics(yy,scores[:,i],conservative[b]),'conservative_threshold':conservative[b],
            'unchanged_neighbor_flag_rate':float(np.mean(np.array(neighbors)>=state['thresholds'][b]))}
    report={'summary':metrics,'partition_texts':{p:len(g) for p,g in grouped.items()},'waveforms':total,'model_sha256':model_hash,'best_epoch':best_epoch,
        'actual_hindi_error_accuracy':None,'actual_prosody_error_accuracy':None,
        'limits':['Follow-up internal experiment motivated by failure of feature-only test; no untouched external benchmark.',
        'First20 earlier physical audit texts excluded; focal lexeme and conditioning partitions inherited and remain disjoint.',
        'Physical edits have known source provenance, not known pronunciation errors. Fade seams and channel mismatch remain possible cues.',
        'One conditioning query/text; only192 training texts. Same architecture/optimizer; epoch and thresholds selected only on32 development texts.']}
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');(args.out/'predictions.json').write_text(json.dumps(predictions,ensure_ascii=False,allow_nan=False)+'\n')
    if sha(args.out/'model.pt')!=model_hash:raise ValueError('Frozen physical model changed')
    status('complete_physical_detector_evaluation',model_sha256=model_hash); print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        LOG.exception('Physical detector follow-up failed explicitly');raise
