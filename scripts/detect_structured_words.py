"""Direct corrected temporal/site word inference with an explicit Nepali reference."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import require_pair,sha
from hindi_detector.structured_detector import BRANCHES,load_model,paired_inputs,site_vector
from hindi_detector.paired_detector import score_arrays
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    for name in ('model','query','reference','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--text',required=True);parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto');args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    pair_path=ROOT/'data/teacher_indicf5_v14/model_pair.json';require_pair(json.loads(pair_path.read_text()))
    network,state=load_model(args.model)
    if state['model_pair_sha256']!=sha(pair_path):raise ValueError('Structured model was not fitted for the active corrected pair')
    canonical=ROOT/'reports/structured_evidence_v15'
    if sha(canonical/'runtime.json')!=state['feature_runtime_sha256']:raise ValueError('Model and canonical feature runtime differ')
    pins=json.loads((canonical/'inputs.json').read_text())
    for path,key in [(ROOT/'data/pocket_bulk_plan_v11/g2p_priors.json','prior_sha256'),
                     (ROOT/'data/schwa_candidates_v1/candidates.json','candidate_sha256'),
                     (ROOT/'.sources/frontend/data/frontend/ipa_map.tsv','ipa_map_sha256')]:
        if sha(path)!=pins[key]:raise ValueError('Structured lexical/site input changed: '+key)
    priors=json.loads((ROOT/'data/pocket_bulk_plan_v11/g2p_priors.json').read_text())
    bank=[{'word':r['word'],'nepali_g2p':{'ipa_candidate_units':r['nepali_ipa_variants'][0]},
        'hindi_g2p':{'units_with_stress_preserved':r['hindi_espeak_raw'].strip().split('|')}} for r in priors]
    with (ROOT/'.sources/frontend/data/frontend/ipa_map.tsv').open() as f:
        reader=csv.DictReader(f,delimiter='\t')
        if reader.fieldnames!=['phone','ipa_candidate','status','notes']:raise ValueError('Lexical IPA schema differs')
        ipa={r['phone']:r['ipa_candidate'] for r in reader}
    waves=[];identities=[]
    for path in (args.query,args.reference):
        path=path.resolve(strict=True);digest=sha(path)
        waves.append(audio(path.parent,{'audio':path.name,'sha256':digest}));identities.append({'path':str(path),'sha256':digest})
    long_pair=max(map(len,waves))>20*16000
    device=('cpu' if long_pair else 'mps') if args.device=='auto' else args.device
    if device=='mps' and long_pair:raise ValueError('Long MMS inputs exceed the observed MPS kernel limit; use auto or cpu for both query/reference')
    engine=WordEvidenceEngine(ROOT,g2p_word_bank=bank,device=device,pitch_method='dio',temporal_bins=20,
        schwa_candidates=json.loads((ROOT/'data/schwa_candidates_v1/candidates.json').read_text()),lexical_ipa=ipa)
    items=[]
    for samples in waves:
        detail,arrays=engine.analyze(samples,args.text);items.append({'detail':detail,**arrays})
    p,s=paired_inputs(*items);scores=score_arrays(network,state,p,s);words=[];known={r['word'] for r in priors}
    for w,values in zip(items[0]['detail']['words'],scores,strict=True):
        decisions={}
        for b,value in zip(BRANCHES,values):
            op=state['operating_points'][b]
            decisions[b]='unsupported_lexeme' if w['word'] not in known else 'source_shift' if value>=op['high'] else 'teacher_source_like' if value<=op['low'] else 'uncertain'
        words.append({**w,'source_evidence':dict(zip(BRANCHES,map(float,values))),'source_decisions':decisions,
            'exact_schwa_site_supported':bool(site_vector(w)[0]),'actual_hindi_error_probability':None})
    write_group(args.out,{'result.json':{'text':args.text,'model_sha256':sha(args.model),'inputs':identities,'words':words,
        'operating_points':state['operating_points'],'content_checked':False,
        'encoder_device':device,'long_input_cpu_policy':long_pair,
        'warning':'Source evidence, not validated Hindi-error probabilities. Reference must read exactly the same text. CLI does not apply ASR. Unsupported site evidence remains unavailable.'},
        'source_snapshot.json':[{'path':str(q.relative_to(ROOT)),'sha256':sha(q),'source_utf8':q.read_text()} for q in (Path(__file__),ROOT/'hindi_detector/structured_detector.py')]},
        'Direct inference for corrected teacher/IndicF5 structured candidate. result.json contains actual word times, temporal/site diagnostics, separate phone/prosody/combined scores and three-way source decisions. No waveform copy, no real pronunciation gold. Current consumer corrected detector verification; keep compact result until superseded.',
        ['.venv-indicf5/bin/python -m scripts.detect_structured_words --model MODEL --query WAV --reference SAME_TEXT_TEACHER_WAV --text TEXT --out NEW'])


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Structured direct inference failed explicitly');raise
