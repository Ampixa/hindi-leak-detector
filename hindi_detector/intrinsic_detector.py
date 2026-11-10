"""Small reference-free head over explicitly measured phonetic evidence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from hindi_detector.intrinsic_vowels import FEATURE_NAMES


class IntrinsicNetwork(nn.Module):
    def __init__(self,architecture='linear'):
        super().__init__()
        if architecture=='linear':self.head=nn.Linear(len(FEATURE_NAMES),1)
        elif architecture=='mlp16':self.head=nn.Sequential(nn.Linear(len(FEATURE_NAMES),16),nn.Tanh(),nn.Linear(16,1))
        else:raise ValueError('Unknown intrinsic network architecture')

    def forward(self,values):return self.head(values).squeeze(-1)


def score(network,state,features):
    values=np.asarray(features,np.float32)
    if values.ndim!=2 or values.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError('Invalid intrinsic feature matrix')
    normalized=np.clip((values-state['mean'])/state['scale'],-8,8).astype(np.float32)
    with torch.inference_mode():return torch.sigmoid(network(torch.from_numpy(normalized))).cpu().numpy()


def supported(features):
    values=np.asarray(features)
    # A learned score cannot reverse the measured direction. Quality additionally
    # requires actual alternative-vowel emission, not a ratio of two weak paths.
    return ((values[:,1]>0)&(values[:,6]>-1.5)&(values[:,31]==1)&
            ((values[:,29]==0)|(values[:,9]>-1.5)))


def decisions(state,features,probabilities):
    output=[]
    for row,value,available in zip(features,probabilities,supported(features),strict=True):
        points=state['operating_points']['vowel_deletion' if row[28] else 'vowel_quality']
        output.append('strong_vowel_change' if available and value>=points['high'] else
                      'no_strong_vowel_change' if value<=points['low'] else 'uncertain')
    return output


def load(path:Path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['schema']!='intrinsic_vowel_site_v24' or state['feature_names']!=FEATURE_NAMES:
        raise ValueError('Intrinsic checkpoint schema differs')
    network=IntrinsicNetwork(state['architecture']);network.load_state_dict(state['weights'],strict=True)
    return network.eval(),state
