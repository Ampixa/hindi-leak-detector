"""Audit a frozen corrected model on recorded speakers, without inventing labels."""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime,timezone
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import SYSTEMS,sha
from hindi_detector.paired_detector import score_arrays
from hindi_detector.structured_detector import load_model,paired_inputs,site_vector
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio

ROOT=Path(__file__).resolve().parents[1];LOG=logging.getLogger(__name__)

ASR_CODE='''import pathlib,json,logging,sys
from hindi_detector.asr_gate import CpuContentGate
from scripts.calibrate_asr_gate import audio
logging.basicConfig(level=logging.INFO)
out=pathlib.Path(sys.argv[1]);root=pathlib.Path.cwd();gate=CpuContentGate(root,threads=2)
(out/'asr_runtime.json').write_text(json.dumps(gate.runtime,indent=2))
for row in json.loads((out/'requests.json').read_text()):
 result=gate.score(audio(pathlib.Path(row['source_pack']),row),row['text'],row['system'])
 with (out/'content_items.jsonl').open('a') as f:f.write(json.dumps({**row,'content_quality':result},ensure_ascii=False,allow_nan=False)+'\\n');f.flush()
'''


def score_cached_audit(out, contexts, cache, final):
    """Score verified cached inputs after the selected model has been frozen."""
    if (out/'report.json').exists() or (out/'predictions.json').exists():
        raise FileExistsError('Native audit result already exists')
    network,model=load_model(final/'model.pt');predictions=[]
    refs=defaultdict(list)
    for item in cache.values():
        if item['row']['kind']=='teacher_reference':refs[item['row']['text_id']].append(item)
    for item in cache.values():
        row=item['row']
        if row['kind']!='human_recording':continue
        context=contexts[row['text_id']]
        for reference in refs[row['text_id']]:
            p,s=paired_inputs(item,reference);values=score_arrays(network,model,p,s)
            words=[{**{k:w[k] for k in ('word','start','end')},'scores':dict(zip(('phonetics','prosody','combined'),map(float,values[i]))),
                'exact_site_supported':bool(site_vector(w)[0]),'actual_hindi_error_label':None} for i,w in enumerate(item['detail']['words']) if w['word'] in row['target_words']]
            predictions.append({'human_clip_id':row['clip_id'],'text_id':row['text_id'],'speaker_id':row['speaker_id'],
                'reference_clip_id':reference['row']['clip_id'],'reference_speaker_id':reference['row']['speaker_id'],
                'reported_state':context['reported_state'],'reported_district':context['reported_district'],'text':row['text'],'words':words})
    summary={}
    for branch in ('phonetics','prosody','combined'):
        high=model['operating_points'][branch]['high'];values=[w['scores'][branch] for r in predictions for w in r['words']]
        paired=defaultdict(list)
        for r in predictions:
            for w in r['words']:paired[r['human_clip_id'],w['word'],w['start'],w['end']].append(w['scores'][branch])
        changes=[abs(v[0]-v[1]) for v in paired.values() if len(v)==2]
        flips=[(v[0]>=high)!=(v[1]>=high) for v in paired.values() if len(v)==2]
        summary[branch]={'observations':len(values),'conservative_source_flag_rate':float(np.mean(np.array(values)>=high)) if values else None,
            'median_source_score':float(np.median(values)) if values else None,'two_reference_word_pairs':len(changes),
            'reference_flag_flip_fraction':float(np.mean(flips)) if flips else None,'median_reference_score_delta':float(np.median(changes)) if changes else None}
    report={'status':'real_recorded_variation_audit_complete_not_error_accuracy','model_sha256':sha(final/'model.pt'),'summary':summary,
        'human_recordings_scored':len({r['human_clip_id'] for r in predictions}),'human_speaker_groups':len({r['speaker_id'] for r in predictions}),
        'human_reference_pairs':len(predictions),'original_human_recordings':len(contexts),'actual_hindi_error_accuracy':None,
        'limits':['Source-score/flag rates on recorded regional variation, not native-error false-positive ground truth.',
            'All selected metadata report West Bengal; Hindi exposure and Nepal residence are not verified.',
            'Human recordings were not used in corrected classifier fitting; teacher pretraining overlap is not established.',
            'Prior-explored bank and overlapping focal lexemes; no claim of untouched human pronunciation labels.',
            'ASR-conditioned coverage can exclude legitimate dialect variation; report unscored counts.']}
    (out/'predictions.json').write_text(json.dumps(predictions,ensure_ascii=False,allow_nan=False)+'\n')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report


def main():
    out=ROOT/'reports/native_variation_audit_v16';plan=ROOT/'data/native_variation_v16'
    if out.exists():raise FileExistsError(out)
    selection=json.loads((plan/'selection.json').read_text());contexts={c['id']:c for c in selection['contexts']}
    write_group(out,{'inputs.json':{'plan_sha256':sha(plan/'selection.json'),'teacher_sha256':'2b4636e3aed0e4f2d035630e504be95fc58f494b690a851a91bfe0dd8299516f',
        'model_policy':'Wait for final corrected model and reserved evaluation to finish before scoring real humans; no feedback into fitting.',
        'human_labels':None},'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Real-recorded-speaker variation audit. requests.json pins original human/teacher query audio identities; content_items.jsonl and asr_runtime.json record content coverage, asr.log explicit child traces. features/*.json and *.npz cache audited word/site/temporal/prosody arrays, feature_items.jsonl pins hashes/errors. predictions.json contains actual human focal scores against2 teacher reference voices; report.json reports score/flag distributions and reference sensitivity, not Hindi-error accuracy. status.json live PID. No human/audio copy or labels fabricated; current consumer frozen corrected detector variation check. Source licences apply; NOT on HF. Review2026-11-09; remove transient feature caches after consumer closes, retain compact metrics/provenance.',
        ['.venv-indicf5/bin/python -m scripts.audit_native_variation','Verified human_matched_v9 contexts excluded from corrected training texts;32 word families/63 recordings/58 source groups; teacher same-text renders only.'])
    def status(phase,**extra):
        p=out/'status.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n');tmp.replace(p)
    try:
        known=ROOT/'runtime/l40_known_hosts_v14';remote='/home/ampixa/hindi-detector-teacher-f5-v14'
        ssh=['ssh','-4','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','UserKnownHostsFile='+str(known),'-o','StrictHostKeyChecking=yes']
        status('waiting_for_native_context_teacher_renders')
        while True:
            state=json.loads(subprocess.check_output(ssh+['ampixa@khalti-gpu','cat '+shlex.quote(remote+'/data/native_teacher/run.json')],text=True,timeout=30))
            if state['status']=='complete_corrected_audio':break
            if state['status']=='failed_explicitly':raise RuntimeError('Native-context teacher render failed')
            command=subprocess.check_output(ssh+['ampixa@khalti-gpu','ps -p '+str(state['pid'])+' -o args='],text=True,timeout=30)
            if 'scripts.run_corrected_synthesis' not in command or 'native_teacher' not in command:raise RuntimeError('Native teacher process handle differs')
            status('waiting_for_native_context_teacher_renders',processed=state['processed_clips']);time.sleep(10)
        subprocess.run(['rsync','-a','--exclude=*.partial','-e',shlex.join(ssh),'ampixa@khalti-gpu:'+remote+'/data/native_teacher/',str(plan/'teacher')+'/'],check=True,timeout=60)
        runtime=json.loads((plan/'teacher/runtime.json').read_text())
        if runtime['loaded_layers']!=24 or runtime['weights_sha256']!='2b4636e3aed0e4f2d035630e504be95fc58f494b690a851a91bfe0dd8299516f':raise ValueError('Wrong native-audit teacher')
        requests=[]
        for line in (plan/'teacher/items.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['status']=='ok_unreviewed':requests.append({**row,'source_pack':str(plan/'teacher'),'kind':'teacher_reference'})
        for context in contexts.values():
            path=Path(context['native_audio']).resolve(strict=True)
            if sha(path)!=context['native_sha256']:raise ValueError('Native source bytes changed')
            requests.append({'clip_id':context['native_clip_id'],'text_id':context['id'],'text':context['text'],
                'target_words':context['target_words'],'speaker_id':context['native_speaker_id'],'system':'human_nepali_unlabeled',
                'audio':path.name,'sha256':context['native_sha256'],'bytes':path.stat().st_size,'source_pack':str(path.parent),'kind':'human_recording'})
        (out/'requests.json').write_text(json.dumps(requests,ensure_ascii=False,indent=2)+'\n')
        status('screening_native_audit_content',clips=len(requests))
        with (out/'asr.log').open('xb') as log:
            subprocess.run([str(ROOT/'.venv-asr-gate/bin/python'),'-u','-c',ASR_CODE,str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        content=[json.loads(line) for line in (out/'content_items.jsonl').read_text().splitlines()]
        priors=json.loads((ROOT/'data/pocket_bulk_plan_v11/g2p_priors.json').read_text())
        bank=[{'word':r['word'],'nepali_g2p':{'ipa_candidate_units':r['nepali_ipa_variants'][0]},'hindi_g2p':{'units_with_stress_preserved':r['hindi_espeak_raw'].strip().split('|')}} for r in priors]
        with (ROOT/'.sources/frontend/data/frontend/ipa_map.tsv').open() as f:
            reader=csv.DictReader(f,delimiter='\t')
            if reader.fieldnames!=['phone','ipa_candidate','status','notes']:raise ValueError('Lexical map schema differs')
            ipa={r['phone']:r['ipa_candidate'] for r in reader}
        engine=WordEvidenceEngine(ROOT,g2p_word_bank=bank,device='mps',pitch_method='dio',temporal_bins=20,
            schwa_candidates=json.loads((ROOT/'data/schwa_candidates_v1/candidates.json').read_text()),lexical_ipa=ipa)
        (out/'features').mkdir();cache={};receipts=[]
        for row in content:
            if row['content_quality']['decision']!='pass':continue
            identity=row['clip_id'];record={'clip_id':identity,'kind':row['kind'],'text_id':row['text_id'],'source_audio_sha256':row['sha256']}
            try:
                detail,arrays=engine.analyze(audio(Path(row['source_pack']),row),row['text'])
                if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite native-audit features')
                dp=out/'features'/(identity+'.json');fp=out/'features'/(identity+'.npz')
                dp.write_text(json.dumps(detail,ensure_ascii=False,allow_nan=False)+'\n');np.savez_compressed(fp,**arrays)
                record.update(status='ok',detail=dp.name,detail_sha256=sha(dp),features=fp.name,features_sha256=sha(fp),bytes=dp.stat().st_size+fp.stat().st_size)
                cache[identity]={'row':row,'detail':detail,**arrays}
            except ValueError:
                LOG.exception('Explicit native alignment/feature exclusion %s',identity);record.update(status='excluded',error=traceback.format_exc())
            receipts.append(record)
            with (out/'feature_items.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            status('extracting_native_variation_features',processed=len(receipts),usable=len(cache),content_pass=sum(r['content_quality']['decision']=='pass' for r in content))
        status('waiting_for_frozen_final_corrected_model',usable_clips=len(cache))
        final=ROOT/'reports/corrected_training_v16/final'
        while not (final/'report.json').is_file():
            training=json.loads((ROOT/'reports/corrected_training_v16/status.json').read_text())
            if training['phase']=='failed_explicitly':raise RuntimeError('Final corrected training failed; native audit will not fall back to pilot')
            os.kill(training['pid'],0);time.sleep(10)
        model_report=json.loads((final/'report.json').read_text())
        if sha(final/'model.pt')!=model_report['model_sha256']:raise ValueError('Frozen final model identity changed')
        report=score_cached_audit(out,contexts,cache,final)
        status('complete_native_variation_audit',human_recordings=report['human_recordings_scored'])
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        LOG.exception('Native-variation audit failed explicitly');raise
