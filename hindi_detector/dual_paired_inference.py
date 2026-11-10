"""Join two independently measured backbones for the same phonetic site."""
from __future__ import annotations

import numpy as np

from hindi_detector.dual_paired_vowels import FEATURE_NAMES
from hindi_detector.paired_local_inference import analyze_pair as single_pair


def analyze_pair(query,reference,secondary_query,secondary_reference,word_lookup,ipa,vocabulary,secondary_vocabulary):
    words,sites,features=single_pair(query,reference,word_lookup,ipa,vocabulary)
    other_words,other_sites,other_features=single_pair(secondary_query,secondary_reference,word_lookup,ipa,secondary_vocabulary)
    if [w['word'] for w in words]!=[w['word'] for w in other_words]:raise ValueError('Two probes lost canonical word identity')
    def key(site):return site['word_ordinal'],site['site']['kind'],site['site']['index'],tuple(site['native_phones'])
    lookup={key(s):s for s in other_sites}
    if len(lookup)!=len(other_sites):raise ValueError('Duplicate second-probe phonetic identity')
    retained=[];vectors=[]
    for site in sites:
        other=lookup.get(key(site))
        if other is None:
            words[site['word_ordinal']]['unavailable_reasons'].append('Second probe cannot support the same reference-selected Native variant/site')
            continue
        primary=site['features'];secondary=other['features'];f=np.r_[primary,secondary].astype(np.float32)
        site.update(primary_features=primary,secondary_features=secondary,features=f.tolist(),secondary=other)
        retained.append(site);vectors.append(f)
    for ordinal,word in enumerate(words):word['sites']=[s for s in retained if s['word_ordinal']==ordinal]
    return words,retained,np.stack(vectors) if vectors else np.empty((0,len(FEATURE_NAMES)),np.float32)
