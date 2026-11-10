"""Rank frozen candidates using the predeclared qualified development domains."""
from __future__ import annotations

from datetime import datetime,timezone
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.corrected_synthesis import sha
from hindi_detector.intrinsic_detector import load,score
from scripts.train_intrinsic_detector import partition

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/intrinsic_selection_v25'
    if (out/'selected.json').exists():raise FileExistsError(out/'selected.json')
    protocol=json.loads((out/'protocol.json').read_text());bases=[ROOT/p for p in protocol['data_groups']]
    domains=[partition(base,'development',True) for base in bases];candidates=[]
    for relative in protocol['candidates']:
        model=ROOT/relative;status=json.loads((model/'status.json').read_text())
        if status['phase']!='complete_intrinsic_candidate':raise ValueError('Candidate is not frozen/completed')
        digest=sha(model/'model.pt')
        if digest!=status['model_sha256']:raise ValueError('Candidate identity changed')
        network,state=load(model/'model.pt');losses={}
        for base,(x,y,rows,q) in zip(bases,domains,strict=True):
            p=np.clip(score(network,state,x).astype(np.float64),1e-7,1-1e-7)
            losses[base.name]={'bce':float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
                'rows':len(rows),'word_families':len({r['word'] for r in rows})}
        candidates.append({'path':relative,'model_sha256':digest,'architecture':state['architecture'],
            'development':losses,'mean_development_bce':float(np.mean([v['bce'] for v in losses.values()]))})
    winner=min(candidates,key=lambda x:(x['mean_development_bce'],x['path']))
    selected={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'selected_by_qualified_development_only',
        'model':winner['path']+'/model.pt','model_sha256':winner['model_sha256'],
        'mean_development_bce':winner['mean_development_bce'],'promotion_checks_pending':True,
        'actual_hindi_error_accuracy':None,'protocol_sha256':sha(out/'protocol.json')}
    (out/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')
    (out/'report.json').write_text(json.dumps({'selected':selected,'candidates':candidates,
        'no_test_feature_or_test_score_consumption_for_ranking':True},indent=2)+'\n')
    (out/'source_snapshot.json').write_text(json.dumps({'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()},indent=2)+'\n')
    print(json.dumps({'selected':selected,'development_scores':{r['path']:r['mean_development_bce'] for r in candidates}}))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Development-only intrinsic selection failed explicitly');raise
