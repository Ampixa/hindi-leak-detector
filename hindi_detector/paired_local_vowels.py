"""Reference-conditioned vowel path and local nucleus evidence."""
from __future__ import annotations

import numpy as np

from hindi_detector.intrinsic_vowels import FEATURE_NAMES as INTRINSIC_NAMES
from hindi_detector.pronunciation_sites import FEATURE_NAMES as MARGIN_NAMES,paired_site_features

LOCAL_INDICES=list(range(8,21))
FEATURE_NAMES=MARGIN_NAMES+[prefix+INTRINSIC_NAMES[i] for prefix in ('query_','reference_','delta_') for i in LOCAL_INDICES]


def vector(query,reference,query_evidence,reference_evidence,kind):
    def convert(evidence):
        return {'windows':[{'alternative_minus_target_per_frame':w['margin'],
            'target_score_per_frame':w['native_fit'],'alternative_score_per_frame':w['alternative_fit']} for w in evidence['windows']]}
    margin=paired_site_features(convert(query_evidence),convert(reference_evidence),{'kind':kind})
    q=np.asarray(query,np.float32)[LOCAL_INDICES];r=np.asarray(reference,np.float32)[LOCAL_INDICES]
    result=np.r_[margin,q,r,q-r].astype(np.float32)
    if result.shape!=(len(FEATURE_NAMES),) or not np.isfinite(result).all():raise ValueError('Invalid paired local vowel geometry')
    return result


def supported(features):
    x=np.asarray(features)
    # Quality needs actual alternative vowel emission; both causes require
    # actual expected reference vowel. Weak consonant probes do not veto every
    # deletion: that rejected policy destroyed measured intervention recall.
    return ((x[:,1]>0)&(x[:,8]>0)&(x[:,14]==1)&(x[:,7]<0)&(x[:,13]>-1.5)&
            (x[:,31]>-1.5)&((x[:,17]==0)|(x[:,19]>-1.5)))
