"""Recover known long-input hardware failures on CPU, then fit the full candidate."""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime,timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import select
import subprocess
import time
import traceback

import numpy as np
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import SYSTEMS,sha
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows
from scripts.train_corrected_detector import complete_groups,fit

ROOT=Path(__file__).resolve().parents[1];LOG=logging.getLogger(__name__)


def main():
    training=ROOT/'reports/corrected_training_v16';base=ROOT/'reports/structured_evidence_v15';out=base/'cpu_recovery'
    if out.exists() or (training/'final').exists():raise FileExistsError('Fresh recovery and final groups required')
    write_group(out,{'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()},
        'inputs.json':{'policy':'Recover whole four-source/reference unit and both physical controls on CPU only for observed MPS >65536 Conv1d errors. Original receipts unchanged. Final fit overlays only fully content-passing six-clip recovery groups.',
            'parent_inputs_sha256':sha(base/'inputs.json')}},
        'Known long-input CPU recovery. items.jsonl pins recovered detail/NPZ/audio hashes; report.json counts recovered/quarantined texts and superseded error identities. details/*.json and features/*.npz are same pinned encoders on CPU for the whole matched unit; controls/*.wav are actual source inserts/shams. asr.log and asr_runtime.json record separate content child. Old MPS partial features remain diagnostic; no canonical journal edits. Final fitter overlays recovered groups in memory, preserving frozen pilot inputs. Current consumer full corrected fit; review2026-11-09, retire redundant recovery caches after consumer review. No original WAV/model copy or HF upload.',
        ['.venv-indicf5/bin/python -m scripts.finish_corrected_training','Only known observed MPS-kernel failure is recovered; unknown failures remain explicit exclusions.'])
    for d in ('details','features','controls'):(out/d).mkdir()
    recovered={};attempted=set();engine=None;child=None;log=None
    def status(phase,**extra):
        p=training/'status.json';temp=p.with_suffix('.partial');temp.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),'recovered_texts':len(recovered),**extra},indent=2)+'\n');temp.replace(p)
    def append(path,row):
        with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
    def setup():
        nonlocal engine,child,log
        priors=json.loads((ROOT/'data/pocket_bulk_plan_v11/g2p_priors.json').read_text())
        bank=[{'word':r['word'],'nepali_g2p':{'ipa_candidate_units':r['nepali_ipa_variants'][0]},'hindi_g2p':{'units_with_stress_preserved':r['hindi_espeak_raw'].strip().split('|')}} for r in priors]
        with (ROOT/'.sources/frontend/data/frontend/ipa_map.tsv').open() as f:
            reader=csv.DictReader(f,delimiter='\t')
            if reader.fieldnames!=['phone','ipa_candidate','status','notes']:raise ValueError('Lexical IPA schema changed')
            ipa={r['phone']:r['ipa_candidate'] for r in reader}
        engine=WordEvidenceEngine(ROOT,g2p_word_bank=bank,device='cpu',pitch_method='dio',temporal_bins=20,
            schwa_candidates=json.loads((ROOT/'data/schwa_candidates_v1/candidates.json').read_text()),lexical_ipa=ipa)
        (out/'encoder_runtime.json').write_text(json.dumps({'device':'cpu','alignment':engine.aligner_inventory,'phone':engine.phone_inventory,
            'temporal_bins':20,'whole_unit_same_backend':True},indent=2)+'\n')
        log=(out/'asr.log').open('xb')
        child=subprocess.Popen([str(ROOT/'.venv-asr-gate/bin/python'),'-u','-m','scripts.content_gate_rpc'],cwd=ROOT,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        hello=response()
        if hello['status']!='ready':raise ValueError('Recovery gate handshake differs')
        (out/'asr_runtime.json').write_text(json.dumps(hello,indent=2)+'\n')
    def response():
        ready,_,_=select.select([child.stdout],[],[],60)
        if not ready:raise TimeoutError('Recovery content child timed out')
        line=child.stdout.readline()
        if not line:raise RuntimeError('Recovery content child exited; inspect asr.log')
        return json.loads(line)
    def encode(row,samples,kind):
        detail,arrays=engine.analyze(samples,row['text'])
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite CPU recovery features')
        focal=[i for i,w in enumerate(detail['words']) if w['word'] in row['target_words']]
        if not focal:raise ValueError('Recovered focal word missing')
        identity=row['clip_id'];dp=out/'details'/(identity+'.json');fp=out/'features'/(identity+'.npz')
        detail.update(clip_id=identity,focal_word_indices=focal,source_audio_sha256=row['sha256'])
        dp.write_text(json.dumps(detail,ensure_ascii=False,allow_nan=False)+'\n');np.savez_compressed(fp,**arrays)
        receipt={'row':{**row,'encoder_backend':'cpu','hardware_recovery':True},'kind':kind,'text_id':row['text_id'],'clip_id':identity,
            'status':'ok_structured_features','detail':str(dp.relative_to(base)),'detail_sha256':sha(dp),'detail_bytes':dp.stat().st_size,
            'features':str(fp.relative_to(base)),'features_sha256':sha(fp),'feature_bytes':fp.stat().st_size,'focal_word_indices':focal}
        append(out/'items.jsonl',receipt);return receipt,{'detail':detail,**arrays}
    try:
        status('recovering_long_inputs_then_waiting_for_full_evidence')
        while True:
            state=json.loads((base/'status.json').read_text())
            if state['phase']=='failed_explicitly':raise RuntimeError('Primary corrected extraction failed')
            failed=[r for r in read_rows(base/'items.jsonl') if r['status']=='failed_structured_features' and 'Output channels > 65536' in r.get('error','')]
            targets={r['text_id'] for r in failed}-attempted
            content=read_rows(ROOT/'reports/corrected_pair_v14/content_items.jsonl')
            for tid in sorted(targets):
                rows=[r for r in content if r['text_id']==tid and r['content_quality']['decision']=='pass']
                if len(rows)!=4:raise ValueError('Recovery must preserve the full matched quartet')
                if engine is None:setup()
                status('recovering_long_input_matched_unit',text_id=tid)
                encoded={};receipts=[]
                for row in rows:
                    receipt,item=encode(row,audio(Path(row['source_pack']),row),'original');receipts.append(receipt)
                    encoded[row['system'],row['speaker_id']]=(row,receipt,item)
                voices=sorted({r['speaker_id'] for r in rows});ordinal=int(hashlib.sha256(tid.encode()).hexdigest()[:8],16)%2;voice=voices[ordinal]
                nr,nrec,native=encoded[SYSTEMS[0],voice];hr,_,hindi=encoded[SYSTEMS[1],voice];ref=encoded[SYSTEMS[0],voices[1-ordinal]][0]
                waves=[audio(Path(r['source_pack']),r) for r in (nr,hr)];physical=[]
                for label,donor in enumerate((native,hindi)):
                    samples=waves[0].copy()
                    for i in sorted(nrec['focal_word_indices'],reverse=True):
                        n,h=native['detail']['words'][i],donor['detail']['words'][i];start,end=int(n['start']*16000),int(n['end']*16000)
                        chunk=waves[label][int(h['start']*16000):int(h['end']*16000)].copy();width=min(80,len(chunk)//4,start,len(samples)-end)
                        if width<2:raise ValueError('CPU recovery word boundary unsupported')
                        samples[start-width:start]*=np.linspace(1,0,width);samples[end:end+width]*=np.linspace(0,1,width)
                        chunk[:width]*=np.linspace(0,1,width);chunk[-width:]*=np.linspace(1,0,width)
                        samples=np.concatenate([samples[:start],chunk,samples[end:]])
                    path=out/'controls'/(tid+'_'+str(label)+'.wav');sf.write(path,samples,16000,subtype='PCM_16')
                    row={k:v for k,v in nr.items() if k not in ('generation_wall_seconds','generation_detail','clipping_fraction_before_pcm')}
                    row.update(system=SYSTEMS[1] if label else SYSTEMS[0],condition='f5_word_insert' if label else 'teacher_sham',source_label=label,
                        clip_id=hashlib.sha256(('cpu_recovery:'+tid+':'+str(label)).encode()).hexdigest()[:24],audio=str(path.relative_to(out)),
                        sha256=sha(path),bytes=path.stat().st_size,seconds=len(samples)/16000,source_pack=str(out),reference_clip_id=ref['clip_id'],
                        native_clip_id=nr['clip_id'],donor_clip_id=hr['clip_id'],actual_hindi_error_label=None)
                    child.stdin.write(json.dumps({'pack':str(out),'row':row},ensure_ascii=False)+'\n');child.stdin.flush();reply=response()
                    if reply['id']!=row['clip_id']:raise ValueError('Recovery gate identity differs')
                    row['content_quality']=reply['content_quality'];physical.append(row)
                if all(r['content_quality']['decision']=='pass' for r in physical):
                    for row in physical:
                        receipt,_=encode(row,audio(out,row),'physical');receipts.append(receipt)
                    recovered[tid]=receipts
                attempted.add(tid)
                append(out/'text_outcomes.jsonl',{'text_id':tid,'complete_recovered_pair':tid in recovered,'physical_decisions':[r['content_quality']['decision'] for r in physical]})
            if state['phase']=='complete_corrected_structured_evidence':break
            status('pilot_trained_waiting_for_full_corrected_evidence',recovery_attempts=len(attempted),partition_texts=state['physical_texts'])
            os.kill(state['pid'],0);time.sleep(10)
        partitions=complete_groups(base)
        for tid,receipts in recovered.items():partitions[receipts[0]['row']['split']][tid]=receipts
        counts={p:len(v) for p,v in partitions.items()}
        if min(counts.values())<16:raise RuntimeError('Insufficient complete corrected pairs')
        status('fitting_final_corrected_candidate',partition_texts=counts,recovery_attempts=len(attempted))
        contrast={r['word'] for r in json.loads((ROOT/'data/pocket_bulk_plan_v11/g2p_priors.json').read_text())}
        report=fit(base,training/'final',partitions,contrast,final=True,pairwise=True)
        recovery_report={'attempted_texts':len(attempted),'recovered_texts':len(recovered),'recovered_ids':sorted(recovered),
            'encoder_policy':'CPU for all four original recordings and both physical controls of each recovered text; no cross-source backend mismatch.',
            'final_model_sha256':report['model_sha256'],'original_feature_errors_retained':True}
        (out/'report.json').write_text(json.dumps(recovery_report,indent=2)+'\n')
        (training/'final/data_quality_addendum.json').write_text(json.dumps(recovery_report,indent=2)+'\n')
        status('complete_corrected_training_and_reserved_evaluation',partition_texts=counts,model_sha256=report['model_sha256'])
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise
    finally:
        if child is not None:
            child.stdin.close()
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                LOG.error('Recovery ASR child did not close, terminating owned PID%s',child.pid);child.terminate();code=child.wait(timeout=15)
            if code:LOG.error('Recovery ASR child exit%s; inspect log',code)
        if log is not None:log.close()


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        LOG.exception('Corrected recovery/final fit failed explicitly');raise
