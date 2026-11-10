"""Join verified primary/secondary paired evidence, never fabricated zero fills."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime,timezone
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_paired_vowels import FEATURE_NAMES
from hindi_detector.paired_local_vowels import vector
from scripts.extract_corrected_training import read_rows

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/dual_paired_features_v28';base=ROOT/'reports/paired_local_features_v27';physical=ROOT/'reports/secondary_physical_v28'
    if out.exists():raise FileExistsError(out)
    write_group(out,{'protocol.json':{'feature_names':FEATURE_NAMES,'primary_items_sha256':sha(base/'items.jsonl'),
        'recipe':'Exact primary57 feature identity checked against original frozen matrices; secondary57 from measured corresponding query/reference, no missing-value substitutes',
        'labels':'Physical intervention truth; natural/Indic secondary-qualified proxy labels remain explicit and are not independent of dual-head features',
        'fresh_audit':'minimal_audit_v29 is excluded from this entire dataset'},
        'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
            for p in (Path(__file__),ROOT/'hindi_detector/dual_paired_vowels.py')]},
        'Dual114-feature data. index.json exact row/input identities and qualification; arrays_DOMAIN_PART.npz '
        'isolated train/dev/test features/labels/qualification, report.json bytes/SHA/counts, exclusions.json '
        'explicit unsupported second-probe pairs. No new waveform/model copies. Current consumer dual-probe '
        'candidate; review2026-11-09, keep compact evidence while selected model is consumed. NOT on HF; '
        'inherited source/MMS research rights. Acoustic-qualified labels share probe bias.',
        ['.venv-indicf5/bin/python -m scripts.prepare_dual_paired'])
    def status(phase,**kw):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');t.replace(p)
    try:
        status('waiting_for_measured_second_physical_probe')
        while True:
            previous=json.loads((physical/'status.json').read_text())
            if previous['phase']=='failed_explicitly':raise RuntimeError('Second physical acoustic producer failed')
            if previous['phase']=='complete_secondary_physical':break
            os.kill(previous['pid'],0);time.sleep(5)
        physical_report=json.loads((physical/'report.json').read_text())
        if sha(physical/'items.jsonl')!=physical_report['items_sha256']:raise ValueError('Measured second physical receipts changed')
        secondary={r['source_row_index']:r for r in read_rows(physical/'items.jsonl')}
        old_physical=read_rows(ROOT/'reports/intrinsic_physical_v24/items.jsonl');primary=read_rows(base/'items.jsonl')
        natural={}
        for name in ('intrinsic_minimal_v24','intrinsic_internal_v26'):
            for row in read_rows(ROOT/'reports'/name/'items.jsonl'):
                if row['job_id'] in natural:raise ValueError('Duplicate natural job identity')
                natural[row['job_id']]=row
        source=read_rows(ROOT/'reports/indic_phonetic_pairs_v25/items.jsonl');source_lookup=defaultdict(list)
        for row in source:source_lookup[row['key'],row['word'],row['kind'],row['label']].append(row)
        vectors=[];index=[];exclusions=[];physical_ordinal=0
        for row in primary:
            if row['row_index']!=len(vectors)+len(exclusions):raise ValueError('Primary global row sequence differs')
            if row['domain']=='physical':
                original=old_physical[physical_ordinal]
                if any(row[k]!=original[k] for k in ('key','word','kind','voice','label','split')):raise ValueError('Primary physical row join differs')
                other=secondary.get(physical_ordinal);physical_ordinal+=1
                if other is None:
                    exclusions.append({'primary_row_index':row['row_index'],'reason':'Explicit whole secondary physical pair exclusion'});continue
                second=np.asarray(other['features'],np.float32)
            elif row['domain']=='natural':
                q,r=natural[row['key']],natural[row['reference_id']]
                same=vector(q['features'],r['features'],q['evidence'],r['evidence'],row['kind'])
                if not np.array_equal(same,np.asarray(row['features'],np.float32)):raise ValueError('Primary natural recipe identity changed')
                second=vector(q['secondary_features'],r['secondary_features'],q['secondary_evidence'],r['secondary_evidence'],row['kind'])
            elif row['domain']=='indic':
                matches=[]
                for q in source_lookup[row['key'],row['word'],row['kind'],row['label']]:
                    for r in source_lookup[row['reference_id'],row['word'],row['kind'],0]:
                        if q['unit_index']!=r['unit_index'] or q['word_ordinal']!=r['word_ordinal'] or q['evidence']['native_target_phones']!=r['evidence']['native_target_phones']:continue
                        same=vector(q['features'],r['features'],q['evidence'],r['evidence'],row['kind'])
                        if np.array_equal(same,np.asarray(row['features'],np.float32)):matches.append((q,r))
                if len(matches)!=1:raise ValueError('Ambiguous/missing exact primary Indic reference join')
                q,r=matches[0];second=vector(q['secondary_features'],r['secondary_features'],q['secondary_evidence'],r['secondary_evidence'],row['kind'])
            else:raise ValueError('Unknown primary data domain')
            f=np.r_[row['features'],second].astype(np.float32)
            if f.shape!=(114,) or not np.isfinite(f).all():raise ValueError('Invalid dual evidence geometry')
            record={k:row[k] for k in ('word','voice','split','kind','label','domain','key','reference_id','qualified')}
            record.update(row_index=len(index),primary_row_index=row['row_index']);vectors.append(f);index.append(record)
        if physical_ordinal!=len(old_physical):raise ValueError('Incomplete primary physical join')
        if len(index)%2 or any((a['label'],b['label'])!=(0,1) or a['word']!=b['word'] or a['qualified']!=b['qualified']
            for a,b in zip(index[::2],index[1::2],strict=True)):raise ValueError('Dual rows require intact adjacent pairs')
        primary_report=json.loads((base/'report.json').read_text());primary_index=json.loads((base/'index.json').read_text())
        for domain in ('physical','natural','indic'):
            for part in ('train','development','test'):
                path=base/('arrays_'+domain+'_'+part+'.npz');pin=next(p for p in primary_report['inventory'] if p['path']==path.name)
                if sha(path)!=pin['sha256']:raise ValueError('Original primary matrix changed')
                with np.load(path,allow_pickle=False) as data:old=data['features']
                rows=[r for r in primary if r['domain']==domain and r['split']==part]
                if not np.array_equal(old,np.array([r['features'] for r in rows],np.float32)):raise ValueError('Primary receipt vectors differ from frozen matrix')
        matrix=np.stack(vectors);labels=np.array([r['label'] for r in index],np.int64);q=np.array([r['qualified'] for r in index],bool);inventory=[]
        for domain in ('physical','natural','indic'):
            for part in ('train','development','test'):
                mask=np.array([r['domain']==domain and r['split']==part for r in index]);path=out/('arrays_'+domain+'_'+part+'.npz')
                np.savez_compressed(path,features=matrix[mask],labels=labels[mask],qualified=q[mask])
                inventory.append({'path':path.name,'sha256':sha(path),'bytes':path.stat().st_size,'rows':int(mask.sum()),'qualified_pairs':int(q[mask].sum())//2})
        (out/'index.json').write_text(json.dumps(index,ensure_ascii=False)+'\n');(out/'exclusions.json').write_text(json.dumps(exclusions,indent=2)+'\n')
        report={'status':'complete_dual_paired_matrices','inventory':inventory,'index_sha256':sha(out/'index.json'),
            'feature_names':FEATURE_NAMES,'pairs':len(index)//2,'qualified_pairs':int(q.sum())//2,
            'primary_input_sha256':sha(base/'items.jsonl'),'secondary_input_sha256':sha(physical/'items.jsonl'),'actual_hindi_error_accuracy':None}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');status(report['status'],pairs=report['pairs'],qualified_pairs=report['qualified_pairs'])
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Dual paired data preparation failed explicitly');raise
