"""Generic release inference with explicit or cached best-teacher references."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import subprocess

import torch

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_g2p import DualG2P
from hindi_detector.dual_paired_inference import analyze_pair
from hindi_detector.dual_runtime import PhonePair
from hindi_detector.intrinsic_inference import assign_decisions
from hindi_detector.native_observations import load_bank,matching_observations
from hindi_detector.task_vowel_detector import decisions,load,score
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio
from hindi_detector.teacher_reference_cache import ReferenceUnavailableError,references_for_text

ROOT=Path(__file__).resolve().parents[1]



def reference_qualified(site):
    for values in (site['reference_intrinsic_features'],site['secondary']['reference_intrinsic_features']):
        if not (values[2]<0 and values[8]>-1.5 and values[5]>-1.5 and values[30]==1):return False
    return True


def emit_result(args,result):
    write_group(args.out,{'result.json':result,'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
        for p in (Path(__file__),ROOT/'hindi_detector/dual_paired_inference.py',ROOT/'hindi_detector/dual_runtime.py',ROOT/'hindi_detector/dual_g2p.py',ROOT/'hindi_detector/teacher_reference_cache.py',ROOT/'hindi_detector/native_observations.py',ROOT/'scripts/check_query_content.py')]},
        'Cold generic release inference on actual audio/text, explicit, cached or automatically generated content-screened teacher references. '
        'result.json exact inputs/model hashes, per-word/per-reference phonetic evidence, qualifications '
        'and conservative consensus. No fixed-case prediction or word-specific threshold. No raw waveform '
        'or encoder copies; NOT on HF/inherited source/MMS research rights. Current consumer verification '
        'and user inference; review2026-11-09, retain compact result while current model is consumed.',
        ['.venv-indicf5/bin/python -m scripts.detect_nepali_leakage --query WAV --text NEPALI --out NEW [--reference SAME_TEXT_WAV ...]'])
    print(json.dumps({'word_flags':result['word_flags'],'reference_mode':result['reference_mode'],'model_sha256':result['model_sha256'],'status':result['status']},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description='Find Hindi-compatible vowel changes at Nepali word intervals')
    parser.add_argument('--model',type=Path,default=ROOT/'reports/task_vowel_release_v31/model.pt')
    parser.add_argument('--query',type=Path,required=True);parser.add_argument('--text',required=True)
    parser.add_argument('--reference',type=Path,action='append',default=[],help='Trusted same-text reference; repeat for qualified-reference consensus')
    parser.add_argument('--check-query-content',action='store_true',help='Check intended lexical content with Native ASR plus a Hindi audit; gross mismatch yields an explicit unscored result')
    parser.add_argument('--cached-only',action='store_true',help='Refuse new L40 reference generation when text is not cached')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',choices=['auto','mps','cpu'],default='auto')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    networks,state=load(args.model)
    observation_bank=load_bank(args.model.parent,args.model)
    for path,digest in state['code_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Release dependency changed: '+path)
    identities=[];waves=[]
    query=args.query.resolve(strict=True);digest=sha(query);waves.append(audio(query.parent,{'audio':query.name,'sha256':digest}));identities.append({'role':'query','path':str(query),'sha256':digest,'bytes':query.stat().st_size})
    query_content=None
    if args.check_query_content:
        command=[str(ROOT/'.venv-asr-gate/bin/python'),'-m','scripts.check_query_content','--query',str(query),'--text',args.text,'--stdout-only']
        checked=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=300)
        if checked.returncode!=0:raise RuntimeError('Query content check failed explicitly: '+checked.stderr[-4000:])
        query_content=json.loads(checked.stdout)
        if query_content['sha256']!=digest or query_content['text']!=args.text or query_content['decision'] not in {'pass','quarantine','reject'}:
            raise ValueError('Query content receipt differs from requested audio/text')
        if query_content['decision']=='reject':
            emit_result(args,{'status':'unscored_content_mismatch','text':args.text,'model_sha256':sha(args.model),
                'reference_mode':'not_evaluated_query_content_mismatch','inputs':identities,'words':[],'per_reference':[],'word_flags':[],
                'query_content_check':query_content,'content_checked_with_asr':True,'actual_hindi_error_accuracy':None,
                'scope':'No pronunciation judgment: both Nepali recognizers grossly disagree with the intended transcript. Hindi transcript audit is retained; ASR does not supply accent labels.'})
            return
    references=args.reference;reference_mode='explicit_trusted_reference';reference_preparation=None
    if not references:
        try:references,reference_preparation=references_for_text(ROOT,args.text,generate=not args.cached_only)
        except ReferenceUnavailableError as error:
            logging.warning('Explicit reference-quality abstention: %s',error)
            emit_result(args,{'status':'unscored_reference_quality','text':args.text,'model_sha256':sha(args.model),
                'reference_mode':'no_content_qualified_reference','inputs':identities,'words':[],'per_reference':[],'word_flags':[],
                'content_checked_with_asr':args.check_query_content,'query_content_check':query_content,
                'actual_hindi_error_accuracy':None,'scope':str(error)})
            return
        reference_mode=reference_preparation['mode']
    for path in references:
        path=path.resolve(strict=True);digest=sha(path);waves.append(audio(path.parent,{'audio':path.name,'sha256':digest}));identities.append({'role':'reference','path':str(path),'sha256':digest,'bytes':path.stat().st_size})
    device=('mps' if torch.backends.mps.is_available() and max(map(len,waves))<=20*16000 else 'cpu') if args.device=='auto' else args.device
    if device=='mps' and max(map(len,waves))>20*16000:raise ValueError('Long inputs require CPU for the entire comparison')
    phonemizer=DualG2P(ROOT);engine=WordEvidenceEngine(ROOT,device=device,pitch_method='dio',temporal_bins=20,retain_frame_evidence=True)
    runtime=PhonePair(ROOT,state,device=device,primary_engine=engine);items=[]
    for samples in waves:
        detail,arrays=engine.analyze(samples,args.text);other=runtime.encode_one(samples,1)
        primary={'detail':detail,'phone_log_probs':arrays['phone_log_probs']}
        secondary={'detail':{**detail,'phone_frame_seconds':len(samples)/16000/len(other)},'phone_log_probs':other}
        items.append((primary,secondary))
    query,secondary_query=items[0];per_reference=[];groups=defaultdict(list)
    for reference_index,(reference,secondary_reference) in enumerate(items[1:]):
        words,sites,x=analyze_pair(query,reference,secondary_query,secondary_reference,phonemizer.word,phonemizer.ipa,engine.vocab,runtime.vocabularies[1])
        if len(x):
            p=score(networks,state,x);assign_decisions(words,sites,p,decisions(state,x,p))
        else:assign_decisions(words,[],[],[])
        for site in sites:
            site['reference_qualified']=reference_qualified(site);site['reference_index']=reference_index
            key=(site['word_ordinal'],site['site']['kind'],site['site']['index'],tuple(site['native_phones']))
            groups[key].append(site)
        per_reference.append({'reference':identities[reference_index+1],'words':words})
    words=[{**w,'sites':[],'actual_hindi_error_probability':None} for w in query['detail']['words']]
    for key,sites in groups.items():
        valid=[s for s in sites if s['reference_qualified']];values={s['decision'] for s in valid}
        decision=next(iter(values)) if len(values)==1 else 'uncertain'
        if not valid:decision='unsupported_reference'
        words[key[0]]['sites'].append({'site':sites[0]['site'],'native_phones':sites[0]['native_phones'],
            'decision':decision,'qualified_references':len(valid),'available_references':len(sites),
            'reference_evidence':sites,'actual_hindi_error_probability':None})
    for ordinal,word in enumerate(words):
        ds={s['decision'] for s in word['sites']}
        word['decision']='strong_vowel_change' if 'strong_vowel_change' in ds else 'uncertain' if ds&{'uncertain','unsupported_reference'} else 'no_strong_vowel_change' if ds else 'untested_site'
        if word['decision']=='strong_vowel_change' and any(not r['words'][ordinal]['sites'] and not r['words'][ordinal]['unavailable_reasons'] for r in per_reference):
            word['decision']='uncertain';word['native_variant_ambiguity']='A reference-selected Native variant has no supported Hindi contrast'
        for site in word['sites']:
            site['observed_human_variation']=matching_observations(observation_bank,word['word'],site['site'],site['native_phones'])
    result={'status':'scored_vowel_evidence','text':args.text,'model_sha256':sha(args.model),'reference_mode':reference_mode,'encoder_device':device,
        'reference_preparation':reference_preparation,'observed_variation_bank_available':observation_bank is not None,'inputs':identities,'words':words,'per_reference':per_reference,'word_flags':[w['word'] for w in words if w['decision']=='strong_vowel_change'],
        'tracks':query['detail']['tracks'],'seconds':query['detail']['seconds'],'actual_hindi_error_accuracy':None,
        'development_regressions_used_for_release_selection':True,'content_checked_with_asr':args.check_query_content,'query_content_check':query_content,
        'scope':'Word-local Hindi-compatible vowel evidence relative to qualified same-text Native references; never voice/nationality classification.',
        'limits':['Scores are controlled-change model scores, not calibrated Hindi-error probabilities.',
            'One qualified reference is explicitly weaker than replicated support; conflicting qualified references abstain.',
            'No flag is not a correctness guarantee. Unsupported sounds and ambiguous transcripts remain explicit.',
            'Prosody is diagnostic; Hindi-specific intonation accuracy is not established.']}
    emit_result(args,result)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Release inference failed explicitly');raise
