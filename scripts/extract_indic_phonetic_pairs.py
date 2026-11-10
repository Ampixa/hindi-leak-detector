"""Audit realized Hindi-compatible sites in the verified teacher/IndicF5 corpus."""
from __future__ import annotations

from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import logging
import os
from pathlib import Path
import traceback

import numpy as np
import torch
from transformers import AutoFeatureExtractor,AutoModelForCTC

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import require_pair,sha
from hindi_detector.dual_g2p import DualG2P
from hindi_detector.intrinsic_inference import analyze_item
from hindi_detector.intrinsic_vowels import FEATURE_NAMES
from hindi_detector.pronunciation_sites import compile_word
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows
from scripts.prepare_intrinsic_minimal import qualified

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/indic_phonetic_pairs_v25';base=ROOT/'reports/structured_evidence_v15';plan=ROOT/'data/teacher_indicf5_v14'
    if out.exists():raise FileExistsError(out)
    require_pair(json.loads((plan/'model_pair.json').read_text()))
    raw_plan=json.loads((plan/'selection.json').read_text());runtime=json.loads((ROOT/'reports/swar_corpus_features_v18/runtime.json').read_text())
    secondary_runtime=json.loads((ROOT/'reports/swar_phone_crosscheck_v20/runtime.json').read_text())
    forbidden={r['word'] for r in json.loads((ROOT/'data/swar_phone_plan_v18/words.json').read_text())}
    forbidden|={r['word'] for r in json.loads((ROOT/'data/minimal_vowels_v23/selection.json').read_text())['words']}
    phonemizer=DualG2P(ROOT);lexical={};unsupported={}
    target_words={w for r in raw_plan['jobs'] for w in r['target_words']}-forbidden
    for word in sorted(target_words):
        try:
            raw=phonemizer.word(word)
            for vocab in (runtime['vocabulary'],secondary_runtime['vocabulary']):
                if not any(v['contrasts'] for v in compile_word(raw,phonemizer.ipa,vocab)['variants']):
                    raise ValueError('No dual-G2P vowel contrast in one acoustic vocabulary')
            lexical[word]=raw
        except ValueError as error:unsupported[word]=str(error)
    original={}
    for journal in (base/'items.jsonl',base/'cpu_recovery/items.jsonl'):
        for receipt in read_rows(journal):
            if receipt['kind']!='original' or receipt['status']!='ok_structured_features':continue
            row=receipt['row'];key=row['clip_id']
            if key in original:
                if original[key]['row']['sha256']!=row['sha256']:raise ValueError('Recovered source waveform identity differs')
                continue
            original[key]=receipt
    groups=defaultdict(dict)
    for receipt in original.values():
        row=receipt['row']
        if row['content_quality']['decision']!='pass':raise ValueError('Unscreened source reached original feature receipts')
        groups[row['id']][row['system']]=receipt
    pairs=[];excluded=[]
    for key,sources in sorted(groups.items()):
        if set(sources)!={'pocket_teacher_250000_24l','indicf5_stock'}:
            excluded.append({'id':key,'reason':'Missing complete content-screened source pair'});continue
        pair=[sources[s] for s in ('pocket_teacher_250000_24l','indicf5_stock')];a,b=[r['row'] for r in pair]
        if any(a[k]!=b[k] for k in ('id','text','text_id','speaker_id','split','reference_sha256')):
            raise ValueError('Native/Indic source conditioning identity differs')
        targets=[w for w in a['target_words'] if w in lexical and raw_plan['word_partitions'][w]==a['split']]
        if not targets:continue
        if max(a['seconds'],b['seconds'])>20:
            excluded.append({'id':key,'reason':'Long input requires a separate whole-pair CPU audit'});continue
        pairs.append({'id':key,'target_words':targets,'receipts':pair,'split':'test' if a['split']=='reserved_evaluation' else a['split']})
    write_group(out,{'selection.json':pairs,'words.json':list(lexical.values()),'unsupported_words.json':unsupported,'preflight_exclusions.json':excluded,
        'protocol.json':{'feature_names':FEATURE_NAMES,'source_plan_sha256':sha(plan/'selection.json'),
            'source_pair_sha256':sha(plan/'model_pair.json'),'excluded_Swar_and_minimal_words':len(forbidden),
            'supervision':'Requested teacher0/IndicF5 1 only when independent-secondary phonetic direction and emissions qualify whole pair; not source-as-error gold',
            'partitions':'Existing word/conditioning-group split; reserved_evaluation maps to test; no fresh source test read by optimizer before freeze'},
        'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
            for p in (Path(__file__),ROOT/'hindi_detector/intrinsic_vowels.py',ROOT/'scripts/prepare_intrinsic_minimal.py')]},
        'New acoustic audit of the verified correct24-layer teacher versus original IndicF5 recordings. '
        'Selection/wordbank frozen before inference; items.jsonl every complete focal vowel pair with primary '
        'and secondary measured features and explicit realized-direction qualification. Rejected items are '
        'not relabeled native/error. arrays_train/development/test.npz isolate32 primary features/labels/ '
        'qualification; index.json row identity and report.json actual byte/SHA/counts. No raw audio/model '
        'copies, no hidden voice/source embeddings. Current consumer larger query-only candidate, review '
        '2026-11-09; retain compact evidence/model pins, remove redundant diagnostics24h after experiment '
        'closes. NOT on HF; inherited TTS/corpus/MMS research licences apply.',
        ['.venv-indicf5/bin/python -m scripts.extract_indic_phonetic_pairs'])
    def status(phase,**kw):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');t.replace(p)
    try:
        status('loading_two_verified_acoustic_models',source_pairs=len(pairs),word_families=len(lexical))
        models=[]
        for directory,pin in [(ROOT/'models/phoneme_probe',runtime['phone']),(ROOT/'models/phoneme_probe_lv60_v20',secondary_runtime['phone'])]:
            for member in pin['files']:
                if sha(directory/member['path'])!=member['sha256']:raise ValueError('Acoustic checkpoint bytes changed')
            extractor=AutoFeatureExtractor.from_pretrained(directory,local_files_only=True)
            network=AutoModelForCTC.from_pretrained(directory,local_files_only=True).eval().to('mps')
            if extractor.sampling_rate!=16000 or network.config.pad_token_id!=0:raise ValueError('Unexpected acoustic input/blank contract')
            models.append((extractor,network))
        (out/'runtime.json').write_text(json.dumps({'primary':runtime['phone'],'secondary':secondary_runtime['phone'],
            'vocabulary':runtime['vocabulary'],'secondary_vocabulary':secondary_runtime['vocabulary'],
            'alignment':runtime['alignment'],'word_times_from':'Verified existing original MMS alignment, not shifted source swaps',
            'device':'mps'},indent=2)+'\n')
        torch.set_num_threads(2);vectors=[];index=[];qualifications=[];failures=[]
        for number,pair in enumerate(pairs):
            encoded=[]
            try:
                for receipt in pair['receipts']:
                    row=receipt['row'];dp=base/receipt['detail']
                    if sha(dp)!=receipt['detail_sha256']:raise ValueError('Original aligned word detail changed')
                    detail=json.loads(dp.read_text());samples=audio(Path(row['source_pack']),row)
                    focal=[{**w,'original_word_ordinal':i} for i,w in enumerate(detail['words']) if w['word'] in pair['target_words']]
                    if not focal:raise ValueError('Declared focal word missing from actual original alignment')
                    outputs=[]
                    for model_index,(extractor,network) in enumerate(models):
                        with torch.inference_mode():
                            result=network(**{k:v.to('mps') for k,v in extractor(samples,sampling_rate=16000,return_tensors='pt').items()})
                            lp=result.logits[0].log_softmax(-1).cpu().numpy()
                        if not np.isfinite(lp).all():raise ValueError('Nonfinite source phone frames')
                        item={'phone_log_probs':lp,'detail':{'words':focal,'phone_frame_seconds':len(samples)/16000/len(lp)}}
                        words,sites,_=analyze_item(item,lexical.__getitem__,phonemizer.ipa,
                            runtime['vocabulary'] if model_index==0 else secondary_runtime['vocabulary'])
                        lookup={(words[s['word_ordinal']]['original_word_ordinal'],s['site']['kind'],s['site']['index'],tuple(s['native_phones'])):s for s in sites}
                        outputs.append(lookup)
                    encoded.append((row,detail,outputs))
                shared=set(encoded[0][2][0])&set(encoded[1][2][0])&set(encoded[0][2][1])&set(encoded[1][2][1])
                for identity in sorted(shared):
                    other=[np.array(e[2][1][identity]['features']) for e in encoded];realized=qualified(other)
                    for label,(row,detail,outputs) in enumerate(encoded):
                        site=outputs[0][identity];f=np.array(site['features'],np.float32);word=detail['words'][identity[0]]['word']
                        record={'row_index':len(index),'key':pair['id'],'word':word,'word_ordinal':identity[0],
                            'kind':identity[1],'unit_index':identity[2],'voice':row['speaker_id'],'split':pair['split'],'label':label,
                            'source_system':row['system'],'source_clip_id':row['clip_id'],'source_audio_sha256':row['sha256'],
                            'secondary_realization_qualified':realized,'features':site['features'],'evidence':site['evidence'],
                            'secondary_features':outputs[1][identity]['features'],'secondary_evidence':outputs[1][identity]['evidence'],
                            'actual_hindi_error_label':None}
                        with (out/'items.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                        vectors.append(f);qualifications.append(realized);index.append({k:record[k] for k in ('row_index','key','word','word_ordinal','kind','unit_index','voice','split','label','secondary_realization_qualified')})
            except ValueError as error:
                logging.warning('Explicit whole source pair exclusion %s: %s',pair['id'],error);failures.append({'id':pair['id'],'reason':str(error)})
            status('measuring_realized_source_vowel_pairs',source_pairs=number+1,total_source_pairs=len(pairs),
                site_pairs=len(index)//2,qualified_site_pairs=sum(qualifications)//2)
        if not vectors:raise ValueError('No realized source-site evidence')
        matrix=np.stack(vectors);labels=np.array([r['label'] for r in index],np.int64);qualification=np.array(qualifications,bool);inventory=[]
        for part in ('train','development','test'):
            mask=np.array([r['split']==part for r in index]);p=out/('arrays_'+part+'.npz')
            np.savez_compressed(p,features=matrix[mask],labels=labels[mask],qualified=qualification[mask])
            inventory.append({'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p),'rows':int(mask.sum()),'qualified_pairs':int(qualification[mask].sum())//2})
        (out/'index.json').write_text(json.dumps(index,ensure_ascii=False)+'\n');(out/'exclusions.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')
        report={'status':'complete_indic_phonetic_matrices','inventory':inventory,'index_sha256':sha(out/'index.json'),
            'runtime_sha256':sha(out/'runtime.json'),'source_pairs':len(pairs),'site_pairs':len(index)//2,'qualified_site_pairs':sum(qualifications)//2,
            'word_families':len(lexical),'actual_hindi_error_accuracy':None,
            'limits':['Secondary acoustic evidence supplies proxy realization labels, not independently audited real Hindi errors.',
                'Native/G2P pronunciation variants and source speech artifacts constrain interpretation.']}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');status(report['status'],site_pairs=report['site_pairs'],qualified_site_pairs=report['qualified_site_pairs'])
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Realized Indic vowel-pair extraction failed explicitly');raise
