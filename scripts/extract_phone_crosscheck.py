"""A second acoustic backbone on identical verified waveforms/shared word times."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForCTC

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=ROOT/'reports/swar_corpus_features_v18')
    parser.add_argument('--source',type=Path,default=ROOT/'data/swar_speech_v18')
    parser.add_argument('--out',type=Path,default=ROOT/'reports/swar_phone_crosscheck_v20')
    parser.add_argument('--rendered-reference',action='store_true',help='Use verified rendered-reference receipt schema')
    args=parser.parse_args()
    model = ROOT/'models/phoneme_probe_lv60_v20'
    base = args.base
    source = args.source
    out = args.out
    if out.exists():
        raise FileExistsError(out)
    write_group(out, {'inputs.json': {'primary_features': str(base),
        'source':str(source),'rendered_reference':args.rendered_reference,
        'policy': 'Same verified audio and original MMS word boundaries, second acoustic backbone; no independent-human-gold claim'},
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Second acoustic measurements. features/*.npz store LV60 phone log-probability frames; details/*.json '
        'store shared primary MMS word bounds plus second-model phone sequences and frame timing. Original '
        'primary detail SHA retained. items.jsonl pins every waveform and output SHA. runtime.json records '
        'verified second checkpoint/actual vocabulary; no primary classifier weights or probabilities are '
        'applied to these uncalibrated second-model features. Current consumer actual word corroboration and '
        'cross-model disagreement analysis; review2026-11-09, retire redundant frames24h after consumer closes. '
        'No audio copy/HF upload. Acoustic model apache-2.0; shared MMS alignment research restriction remains.',
        ['.venv-indicf5/bin/python -m scripts.extract_phone_crosscheck'])
    for name in ('details','features'):
        (out/name).mkdir()

    def status(phase, **extra):
        p = out/'status.json'; temp = p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(),
            'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2)+'\n'); temp.replace(p)

    try:
        status('waiting_for_verified_second_checkpoint')
        while True:
            state = json.loads((model/'download_status.json').read_text())
            if state['phase']=='failed_explicitly':
                raise RuntimeError('Second acoustic checkpoint download failed')
            if state['phase']=='complete_verified_phone_checkpoint':
                break
            os.kill(state['pid'],0); time.sleep(5)
        inventory = json.loads((model/'inventory.json').read_text())
        for member in inventory['files']:
            path = model/member['path']
            if path.stat().st_size != member['bytes'] or sha(path) != member['sha256']:
                raise ValueError('Second checkpoint byte identity differs')
        vocab = json.loads((model/'vocab.json').read_text())
        if {k:vocab[k] for k in ('<pad>','<s>','</s>','<unk>')} != {'<pad>':0,'<s>':1,'</s>':2,'<unk>':3}:
            raise ValueError('Second phone model special-token schema differs')
        extractor = AutoFeatureExtractor.from_pretrained(model, local_files_only=True)
        phone = AutoModelForCTC.from_pretrained(model, local_files_only=True).eval().to('mps')
        if extractor.sampling_rate != 16000 or phone.config.pad_token_id != 0:
            raise ValueError('Second phone input/blank contract differs')
        primary_runtime = json.loads((base/'runtime.json').read_text())
        (out/'runtime.json').write_text(json.dumps({'phone': inventory, 'vocabulary': vocab,
            'alignment': primary_runtime['alignment'], 'device':'mps', 'shared_alignment_sha256':sha(base/'runtime.json'),
            'parameters':sum(p.numel() for p in phone.parameters()), 'torch':torch.__version__,
            'not_independent_training_corpus':True},indent=2)+'\n')
        receipts = read_rows(base/'items.jsonl')
        if args.rendered_reference:
            synthesis=json.loads((source/'runtime.json').read_text())
            if synthesis['loaded_layers']!=24 or synthesis['weights_sha256']!='2b4636e3aed0e4f2d035630e504be95fc58f494b690a851a91bfe0dd8299516f':
                raise ValueError('Rendered reference source differs from requested teacher')
            if any(row['status']!='ok_rendered_phone_frames' or row['system']!='pocket_teacher_250000_24l' for row in receipts):
                raise ValueError('Unexpected rendered-reference receipt schema/status')
        for ordinal, row in enumerate(receipts):
            path = base/row['detail']
            if sha(path)!=row['detail_sha256']:
                raise ValueError('Shared word detail identity changed')
            primary = json.loads(path.read_text())
            samples = audio(source, {'audio':row['audio'] if args.rendered_reference else row['path'],'sha256':row['sha256']})
            with torch.inference_mode():
                result = phone(**{k:v.to('mps') for k,v in extractor(samples,sampling_rate=16000,return_tensors='pt').items()})
                lp = result.logits[0].log_softmax(-1).cpu().numpy()
            if lp.shape[1]!=len(vocab) or not np.isfinite(lp).all():
                raise ValueError('Invalid second acoustic posterior geometry')
            seconds = len(samples)/16000;dt = seconds/len(lp);tokens = {v:k for k,v in vocab.items()}
            words = []
            for word in primary['words']:
                start=max(0,int(np.floor(word['start']/dt)));end=min(len(lp),int(np.ceil(word['end']/dt)))
                ids=lp[start:end].argmax(1)
                collapsed=[int(t) for i,t in enumerate(ids) if t>3 and (i==0 or t!=ids[i-1])]
                words.append({**{k:word[k] for k in ('word','display','start','end','text_start','text_end')},
                    'phone_probe_sequence':' '.join(tokens[t] for t in collapsed)})
            identity=row['id'] if args.rendered_reference else row['id']+'_'+row['kind'];dp=out/'details'/(identity+'.json');fp=out/'features'/(identity+'.npz')
            dp.write_text(json.dumps({'words':words,'text':row['text'],'seconds':seconds,'phone_frame_seconds':dt,
                'primary_word_detail_sha256':row['detail_sha256'],'source_audio_sha256':row['sha256'],
                'shared_MMS_word_alignment':True},ensure_ascii=False)+'\n')
            np.savez_compressed(fp,phone_log_probs=lp.astype(np.float32))
            receipt={**row,'detail':str(dp.relative_to(out)),'detail_sha256':sha(dp),
                'features':str(fp.relative_to(out)),'features_sha256':sha(fp),'encoder_device':'mps'}
            with (out/'items.jsonl').open('a') as f:f.write(json.dumps(receipt,ensure_ascii=False)+'\n')
            status('extracting_second_acoustic_model',completed=ordinal+1,total=len(receipts))
            if row['id'] in {'real_yt_yt01_005','real_yt_yt01_008','real_yt_yt05_036','real_yt_yt03_018'}:
                print(json.dumps({'id':row['id'],'kind':row['kind'],'words':[w for w in words if w['word'] in {'हैन','किनकि','तलको'}]},ensure_ascii=False),flush=True)
        status('complete_actual_picks_features',completed=len(receipts))
    except Exception:
        status('failed_explicitly',error=traceback.format_exc())
        raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Second acoustic extraction failed explicitly')
        raise
