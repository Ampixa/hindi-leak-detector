"""Shared generic inference over verified same-text query/reference frames."""
from __future__ import annotations

import logging

import numpy as np

from hindi_detector.intrinsic_vowels import measure
from hindi_detector.paired_local_vowels import FEATURE_NAMES,vector
from hindi_detector.pronunciation_sites import compile_word,select_reference_structure,windows


def analyze_pair(query,reference,word_lookup,ipa,vocabulary):
    if [w['word'] for w in query['detail']['words']]!=[w['word'] for w in reference['detail']['words']]:
        raise ValueError('Same-text query/reference canonical words differ')
    words=[];sites=[];vectors=[]
    for ordinal,word in enumerate(query['detail']['words']):
        record={**word,'sites':[],'unavailable_reasons':[],'actual_hindi_error_probability':None}
        try:
            compiled=compile_word(word_lookup(word['word']),ipa,vocabulary)
            qc=windows(query,ordinal);rc=windows(reference,ordinal)
            variant,_=select_reference_structure(compiled,rc)
            for contrast in variant['contrasts']:
                try:
                    q,qe=measure(compiled,variant,contrast,qc);r,re=measure(compiled,variant,contrast,rc)
                    f=vector(q,r,qe,re,contrast['kind'])
                    site={'word_ordinal':ordinal,'site':{k:v for k,v in contrast.items() if k!='alternative_unit'},
                        'native_phones':variant['phones'],'features':f.tolist(),'query':qe,'reference':re,
                        'query_intrinsic_features':q.tolist(),'reference_intrinsic_features':r.tolist()}
                    record['sites'].append(site);sites.append(site);vectors.append(f)
                except ValueError as error:
                    logging.warning('Explicit unavailable local paired site %s: %s',word['word'],error);record['unavailable_reasons'].append(str(error))
        except ValueError as error:
            logging.warning('Explicit unsupported local paired word %s: %s',word['word'],error);record['unavailable_reasons'].append(str(error))
        words.append(record)
    return words,sites,np.stack(vectors) if vectors else np.empty((0,len(FEATURE_NAMES)),np.float32)
