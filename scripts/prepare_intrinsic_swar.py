"""Prepare query-only actual Swar evidence with no teacher/human reference."""
from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.intrinsic_inference import analyze_item
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/intrinsic_swar_features_v24';base=ROOT/'reports/swar_corpus_features_v18';second=ROOT/'reports/swar_phone_crosscheck_v20'
    if out.exists():raise FileExistsError(out)
    runtime=json.loads((base/'runtime.json').read_text());sr=json.loads((second/'runtime.json').read_text())
    ipa=json.loads((ROOT/'data/swar_phone_plan_v18/ipa_map.json').read_text())
    lexical={r['word']:r for r in json.loads((ROOT/'data/swar_phone_plan_v18/words.json').read_text())}
    second_rows={(r['id'],r['kind']):r for r in read_rows(second/'items.jsonl')};records=[];vectors=[];indices=[]
    for number,receipt in enumerate(read_rows(base/'items.jsonl')):
        item=load_item(base,receipt);words,sites,matrix=analyze_item(item,lexical.__getitem__,ipa,runtime['vocabulary'])
        other=load_item(second,second_rows[receipt['id'],receipt['kind']]);ow,osites,omatrix=analyze_item(other,lexical.__getitem__,ipa,sr['vocabulary'])
        second_sites={(s['word_ordinal'],s['site']['kind'],s['site']['index'],tuple(s['native_phones'])):s for s in osites}
        for site,f in zip(sites,matrix,strict=True):
            other_site=second_sites.get((site['word_ordinal'],site['site']['kind'],site['site']['index'],tuple(site['native_phones'])))
            site['secondary_evidence']=other_site;site['row_index']=len(vectors)
            vectors.append(f);indices.append({'row_index':site['row_index'],'id':receipt['id'],'source':receipt['kind'],
                'word':words[site['word_ordinal']]['word'],'word_ordinal':site['word_ordinal'],'kind':site['site']['kind']})
        records.append({'id':receipt['id'],'source':receipt['kind'],'words':words,'audio':receipt['path'],
            'audio_sha256':receipt['sha256'],'seconds':item['detail']['seconds'],'tracks':item['detail']['tracks'],
            'source_detail_sha256':receipt['detail_sha256'],'no_reference_audio_used':True})
        if (number+1)%30==0:logging.info('Query-only actual clips measured %s/258',number+1)
    write_group(out,{'cases.json':records,'index.json':indices,'inputs.json':{'primary_runtime_sha256':sha(base/'runtime.json'),
        'secondary_runtime_sha256':sha(second/'runtime.json'),'g2p_plan_sha256':sha(ROOT/'data/swar_phone_plan_v18/words.json'),
        'actual_hindi_error_labels':None},'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
            for p in (Path(__file__),ROOT/'hindi_detector/intrinsic_inference.py')]},
        'Query-only actual258-clip phonetic measurement. cases.json every measured word/site/secondary '
        'probe/interval, index.json row identities, arrays.npz32 primary features. Neither human nor teacher '
        'reference influences these features; secondary directions are corroboration, not gold. No new '
        'audio/model copy. Current consumer reference-free candidate actual regression/coverage checks; '
        'review2026-11-09, retain compact results through release review, NOT on HF/inherited source rights.',
        ['.venv-indicf5/bin/python -m scripts.prepare_intrinsic_swar'])
    np.savez_compressed(out/'arrays.npz',features=np.stack(vectors))
    report={'status':'complete_actual_intrinsic_features','clips':len(records),'sites':len(vectors),
        'array_sha256':sha(out/'arrays.npz'),'cases_sha256':sha(out/'cases.json'),'index_sha256':sha(out/'index.json'),
        'counts':dict(Counter(r['source']+'/'+r['kind'] for r in indices)),'actual_hindi_error_accuracy':None}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Query-only actual evidence preparation failed explicitly');raise
