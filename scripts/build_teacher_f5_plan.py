"""Freeze the requested teacher/IndicF5 comparison with shared short prompts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torchaudio
import uroman

from hindi_detector.artifacts import write_group
from hindi_detector.word_evidence import normalized_words

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
TEACHER_SHA = '2b4636e3aed0e4f2d035630e504be95fc58f494b690a851a91bfe0dd8299516f'
TOKENIZER_SHA = 'd59424239dac0eeed522a28dabd48e17f523464afbfc50c5ccafa6be79f43122'
F5_SHA = 'ba7f3671180fb7784e24bd1dafc96e729a38ce02e7f6d3877cdef32525a1865c'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = ROOT / 'data/pocket_bulk_plan_v11/selection.json'
    old = json.loads(source.read_text()); by_part = defaultdict(list)
    for row in old['texts']:
        if row['target_words']:
            by_part[row['split']].append(row)
    chosen = []
    # Keep all available natural focal texts in held-out partitions; metadata-only
    # selection does not privilege clips that were easy for the old generators.
    for part, quota in [('train', 512), ('development', 128), ('reserved_evaluation', 128)]:
        pool = by_part[part]; natural = [r for r in pool if r['text_kind'] == 'natural_transcript']
        targeted = [r for r in pool if r['text_kind'] == 'authored_targeted_carrier']
        take = min(len(natural), quota // 2)
        chosen.extend(natural[:take] + targeted[:quota-take])
    if len(chosen) != 768 or len({r['id'] for r in chosen}) != 768:
        raise ValueError('Expected768 unique verified focal texts')
    ids = {r['id'] for r in chosen}; jobs = [r for r in old['jobs'] if r['text_id'] in ids]
    # The first eight jobs are a fixed training-only smoke subset with both kinds.
    smoke_ids = {r['id'] for kind in ('natural_transcript', 'authored_targeted_carrier')
                 for r in [r for r in chosen if r['split']=='train' and r['text_kind']==kind][:2]}
    jobs.sort(key=lambda r: (r['text_id'] not in smoke_ids, r['text_id'], r['id']))
    inventory = json.loads((ROOT / 'models/indicf5/inventory.json').read_text())
    if inventory['repo'] != 'ai4bharat/IndicF5' or inventory['revision'] != 'fb652a96e508091045890dae5bb99fe511facbcf':
        raise ValueError('Wrong IndicF5 bundle')
    if sha(ROOT / 'models/indicf5/model.safetensors') != F5_SHA:
        raise ValueError('IndicF5 checkpoint differs')
    write_group(args.out, {'model_pair.json': {'schema':'teacher_indicf5_pair_v14',
        'systems': {'pocket_teacher_250000_24l': {'host':'khalti-gpu','weights':'/home/ampixa/pocket_tts_stage/teacher_run_20261002/model.safetensors',
            'sha256':TEACHER_SHA,'step':250000,'layers':24,'source':'/home/ampixa/pocket_tts_stage/pocket-tts',
            'source_revision':'2f0e24033bf49e8aedcfeb9da2be4fcf2e4cc2c2',
            'config':'/home/ampixa/pocket_tts_stage/pocket-tts/pocket_tts/config/english_2026-04_24l.yaml',
            'tokenizer':'/home/ampixa/pocket_tts_stage/model/tokenizer_clean.model','tokenizer_sha256':TOKENIZER_SHA,
            'inference':{'temperature':.3,'n_steps':1,'cfg_coef':2.,'eos_threshold':-1.,'max_frames':220,'device':'cuda','dtype':'float32'}},
        'indicf5_stock':{'host':'k2','repo':inventory['repo'],'revision':inventory['revision'],'sha256':F5_SHA,
            'model_root':str(ROOT/'models/indicf5'),'inference':{'nfe_step':32,'cfg_strength':2.,'sway_sampling_coef':-1.,'speed':1.,'device':'mps'}}},
        'supersedes':'k2 nepali_v1c step1309 and Pocket Hindi parent roles; old results are historical only',
        'automatic_pronunciation_labels':None},
        'inputs.json':{'old_text_plan_sha256':sha(source),'prior_sha256':sha(ROOT/'data/pocket_bulk_plan_v11/g2p_priors.json'),
            'selection':'fixed metadata-only per-split quotas, max half natural, no old ASR outcomes used',
            'text_counts':dict(Counter(r['split'] for r in chosen)),
            'kind_counts':dict(Counter(r['split']+'/'+r['text_kind'] for r in chosen))},
        'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'User-requested corrected model pair. selection.json freezes768 texts/1536 reference jobs and smoke IDs; model_pair.json explicitly pins teacher24layers and AI4Bharat IndicF5, never legacy fallback. references/*.wav are derived shared <=5s prompts; reference_inventory.json pins original/crop bytes, text, boundaries and alignment. Generator subgroups retain actual per-WAV hashes/status. No pronunciation gold inferred from either model. Current consumer corrected synthesis/ASR and paired word evaluation.',
        ['.venv-indicf5/bin/python -m scripts.build_teacher_f5_plan --out NEW', 'Shared prompts reproduce from original WAV SHA and measured end sample; no normalization, source transcript cropped at aligned word boundary. Source and model licences continue to apply. No HF upload; review audio by2026-11-09 and remove when superseded.'])
    align_dir = ROOT/'models/mms_fa'; inv = json.loads((align_dir/'inventory.json').read_text())
    checkpoint = align_dir/inv['file']
    if sha(checkpoint) != inv['sha256'] or torchaudio.__version__ != inv['torchaudio']:
        raise ValueError('Pinned alignment runtime/checkpoint changed')
    torch.set_num_threads(2); bundle=torchaudio.pipelines.MMS_FA
    model=bundle.get_model(dl_kwargs={'model_dir':str(align_dir),'file_name':checkpoint.name,'weights_only':True}).eval().to('mps')
    tokenizer, aligner, romanizer = bundle.get_tokenizer(),bundle.get_aligner(),uroman.Uroman()
    refs={r['reference_sha256']:r for r in jobs}; crops={}; records=[]
    (args.out/'references').mkdir()
    for digest, row in refs.items():
        original=Path(row['reference_audio'])
        if sha(original)!=digest:
            raise ValueError('Original conditioning bytes changed')
        wave,rate=sf.read(original,dtype='float32',always_2d=True)
        if wave.shape[1]!=1 or not np.isfinite(wave).all():
            raise ValueError('Invalid original conditioning waveform')
        divisor=int(np.gcd(rate,16000));samples=resample_poly(wave[:,0],16000//divisor,rate//divisor).astype('float32')
        words=normalized_words(row['reference_text'],romanizer)
        with torch.inference_mode():
            emissions,_=model(torch.from_numpy(samples)[None].to('mps'))
            spans=aligner(emissions[0].log_softmax(-1).cpu(),tokenizer([w['roman'] for w in words]))
        if len(spans)!=len(words) or any(not s for s in spans):
            raise ValueError('Conditioning alignment lost a word')
        frame_seconds=len(samples)/16000/emissions.shape[1]
        allowed=[i for i,s in enumerate(spans) if s[-1].end*frame_seconds<=4.96]
        if not allowed:
            raise ValueError('No aligned word ends within shared prompt limit')
        last=max(allowed);end=min(len(wave),int((spans[last][-1].end*frame_seconds+.04)*rate))
        text=row['reference_text'][:words[last]['text_end']].strip()
        if end/rate<1.:
            raise ValueError('Shared conditioning prefix too short')
        path=args.out/'references'/(digest[:20]+'.wav');sf.write(path,wave[:end,0],rate,subtype='PCM_16')
        crops[digest]={'reference_audio':str(path.relative_to(args.out)),'reference_sha256':sha(path),'reference_text':text}
        records.append({'source_path':str(original),'source_sha256':digest,'original_text':row['reference_text'],
            'path':str(path.relative_to(args.out)),'sha256':sha(path),'bytes':path.stat().st_size,'sample_rate':rate,
            'end_sample':end,'seconds':end/rate,'text':text,'alignment_checkpoint_sha256':inv['sha256']})
    jobs=[{**r,**crops[r['reference_sha256']],'source_reference_sha256':r['reference_sha256']} for r in jobs]
    (args.out/'selection.json').write_text(json.dumps({'experiment_id':'teacher_indicf5_v14','jobs':jobs,'texts':chosen,
        'word_partitions':old['word_partitions'],'smoke_text_ids':sorted(smoke_ids),'smoke_jobs':8},ensure_ascii=False,indent=2)+'\n')
    (args.out/'reference_inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'texts':len(chosen),'jobs_per_system':len(jobs),'references':len(records),'model_pair':str(args.out/'model_pair.json')}))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        LOG.exception('Corrected source-plan construction failed explicitly');raise
