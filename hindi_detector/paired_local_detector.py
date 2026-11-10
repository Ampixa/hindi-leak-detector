"""Small learned heads over reference-conditioned local vowel evidence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from hindi_detector.paired_local_vowels import FEATURE_NAMES,supported


class PairedLocalNetwork(nn.Module):
    def __init__(self,architecture,feature_mode):
        super().__init__()
        if feature_mode not in {'margins18','local57'}:raise ValueError('Unknown local feature mode')
        size=18 if feature_mode=='margins18' else len(FEATURE_NAMES)
        if architecture=='linear':self.head=nn.Linear(size,1)
        elif architecture=='mlp16':self.head=nn.Sequential(nn.Linear(size,16),nn.Tanh(),nn.Linear(16,1))
        else:raise ValueError('Unknown paired local network')

    def forward(self,values):return self.head(values).squeeze(-1)


def score(network,state,features):
    x=np.asarray(features,np.float32)
    if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(x).all():raise ValueError('Wrong local paired matrix')
    values=np.clip((x-state['mean'])/state['scale'],-8,8).astype(np.float32)
    if state['feature_mode']=='margins18':values=values[:,:18]
    with torch.inference_mode():return torch.sigmoid(network(torch.from_numpy(values))).cpu().numpy()


def decisions(state,features,probabilities):
    result=[]
    for x,p,eligible in zip(features,probabilities,supported(features),strict=True):
        op=state['operating_points']['vowel_deletion' if x[16] else 'vowel_quality']
        result.append('strong_vowel_change' if eligible and p>=op['high'] else 'no_strong_vowel_change' if p<=op['low'] else 'uncertain')
    return result


def load(path:Path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['schema']!='paired_local_vowel_v27' or state['feature_names']!=FEATURE_NAMES:raise ValueError('Wrong paired local schema')
    network=PairedLocalNetwork(state['architecture'],state['feature_mode']);network.load_state_dict(state['weights'],strict=True)
    return network.eval(),state
