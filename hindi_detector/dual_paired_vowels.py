"""Two-backbone paired phonetic evidence with explicit local diagnosis gates."""
from __future__ import annotations

import numpy as np

from hindi_detector.paired_local_vowels import FEATURE_NAMES as SINGLE_NAMES

FEATURE_NAMES=['primary_'+n for n in SINGLE_NAMES]+['secondary_'+n for n in SINGLE_NAMES]


def supported(features):
    x=np.asarray(features);a,b=x[:,:57],x[:,57:]
    reference=(a[:,31]>-1.5)&(b[:,31]>-1.5)&(a[:,7]<0)&(b[:,7]<0)
    shift=(a[:,9]>0)&(b[:,9]>0)&(a[:,13]>-1.5)&(b[:,13]>-1.5)
    deletion=(a[:,30]>0)&(b[:,30]>0)&((a[:,44]<0)|(b[:,44]<0))
    quality=(a[:,19]>-1.5)&(b[:,19]>-1.5)&((a[:,2]>0)|(b[:,2]>0))
    return reference&shift&np.where(a[:,16]>0,deletion,quality)
