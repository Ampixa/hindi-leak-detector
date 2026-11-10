"""Cold audio+text inference with the frozen reference-free vowel candidate."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_g2p import DualG2P
from hindi_detector.intrinsic_detector import decisions,load,score
from hindi_detector.intrinsic_inference import analyze_item,assign_decisions
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description='Find Hindi-compatible vowel changes at word intervals; no reference recording required')
    parser.add_argument('--model',type=Path,default=ROOT/'reports/intrinsic_detector_v24/model.pt')
    parser.add_argument('--query',type=Path,required=True);parser.add_argument('--text',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    network,state=load(args.model)
    for relative,key in [('hindi_detector/intrinsic_vowels.py','intrinsic_code_sha256'),
                         ('hindi_detector/phone_lattice.py','projection_code_sha256'),
                         ('hindi_detector/pronunciation_sites.py','site_code_sha256')]:
        if sha(ROOT/relative)!=state[key]:raise ValueError('Intrinsic dependency changed: '+key)
    path=args.query.resolve(strict=True);digest=sha(path);samples=audio(path.parent,{'audio':path.name,'sha256':digest})
    device=('mps' if torch.backends.mps.is_available() and len(samples)<=20*16000 else 'cpu') if args.device=='auto' else args.device
    if device=='mps' and len(samples)>20*16000:raise ValueError('Long input requires CPU')
    phonemizer=DualG2P(ROOT);engine=WordEvidenceEngine(ROOT,device=device,pitch_method='dio',temporal_bins=20,retain_frame_evidence=True)
    if engine.phone_inventory!=state['phone_inventory']:raise ValueError('Trained phone encoder inventory differs')
    detail,arrays=engine.analyze(samples,args.text);item={'detail':detail,'phone_log_probs':arrays['phone_log_probs']}
    words,sites,matrix=analyze_item(item,phonemizer.word,phonemizer.ipa,engine.vocab)
    if len(matrix):
        p=score(network,state,matrix);assign_decisions(words,sites,p,decisions(state,matrix,p))
    else:assign_decisions(words,[],[],[])
    result={'text':args.text,'model_sha256':sha(args.model),'encoder_device':device,'requires_reference_audio':False,
        'input':{'path':str(path),'sha256':digest,'bytes':path.stat().st_size},'words':words,'seconds':detail['seconds'],
        'word_flags':[w['word'] for w in words if w['decision']=='strong_vowel_change'],'tracks':detail['tracks'],
        'content_checked_with_asr':False,'actual_hindi_error_accuracy':None,
        'scope':'Measured Hindi-compatible vowel deletion/quality relative to dual G2P. Controlled-change scores are not Hindi-error probabilities.',
        'limits':['Legitimate Nepali reductions/variation and G2P/probe errors remain possible.',
            'No flag is not a correctness guarantee; untested sounds remain explicit.',
            'Numeric text requires a verified spoken expansion. Hindi-specific intonation accuracy is not established.']}
    write_group(args.out,{'result.json':result,'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
        for p in (Path(__file__),ROOT/'hindi_detector/intrinsic_inference.py',ROOT/'hindi_detector/dual_g2p.py')]},
        'Direct generic reference-free audio+text detector inference. result.json exact input/model hashes, '
        'word intervals, measured32-feature phonetic counterfactuals and decisions; no fixed-case lookup. '
        'No new audio/model copy, NOT on HF. Current consumer cold inference verification/user use; '
        'review2026-11-09, retain compact result through release validation. Inherited source/MMS research licences.',
        ['.venv-indicf5/bin/python -m scripts.detect_intrinsic_words --query WAV --text NEPALI --out NEW'])
    print(json.dumps({'word_flags':result['word_flags'],'tested_sites':len(sites),'model_sha256':result['model_sha256']},ensure_ascii=False))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Reference-free cold inference failed explicitly');raise
