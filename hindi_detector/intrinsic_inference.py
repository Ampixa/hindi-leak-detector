"""Generic query-only word analysis shared by batch and cold inference."""
from __future__ import annotations

import logging

import numpy as np

from hindi_detector.intrinsic_vowels import measure
from hindi_detector.pronunciation_sites import compile_word,select_reference_structure,windows


def analyze_item(item,word_lookup,ipa,vocabulary):
    words=[];sites=[];vectors=[]
    for ordinal,word in enumerate(item['detail']['words']):
        record={**word,'sites':[],'unavailable_reasons':[],'actual_hindi_error_probability':None}
        try:
            hypothesis=word_lookup(word['word']);compiled=compile_word(hypothesis,ipa,vocabulary)
            crops=windows(item,ordinal)
            # This is the observed query, not a hidden same-text reference.
            variant,_=select_reference_structure(compiled,crops)
            for contrast in variant['contrasts']:
                try:
                    vector,evidence=measure(compiled,variant,contrast,crops)
                    site={'word_ordinal':ordinal,'site':{k:v for k,v in contrast.items() if k!='alternative_unit'},
                        'native_phones':variant['phones'],'features':vector.tolist(),'evidence':evidence}
                    record['sites'].append(site);sites.append(site);vectors.append(vector)
                except ValueError as error:
                    logging.warning('Explicit unavailable query-only site %s: %s',word['word'],error)
                    record['unavailable_reasons'].append(str(error))
        except ValueError as error:
            logging.warning('Explicit unsupported query-only word %s: %s',word['word'],error)
            record['unavailable_reasons'].append(str(error))
        words.append(record)
    return words,sites,np.stack(vectors) if vectors else np.empty((0,32),np.float32)


def assign_decisions(words,sites,probabilities,decisions):
    for site,p,decision in zip(sites,probabilities,decisions,strict=True):
        site.update(controlled_vowel_change_score=float(p),decision=decision,actual_hindi_error_probability=None)
    for word in words:
        ds={s['decision'] for s in word['sites']}
        word['decision']=('strong_vowel_change' if 'strong_vowel_change' in ds else 'uncertain' if 'uncertain' in ds
            else 'no_strong_vowel_change' if ds else 'untested_site')
