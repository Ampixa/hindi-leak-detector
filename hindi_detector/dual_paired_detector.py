"""Ablated small heads over dual-backbone paired phonetic measurements."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from hindi_detector.dual_paired_vowels import FEATURE_NAMES,supported


def select_inputs(values,mode):
    if mode=='primary57':return values[:,:57]
    if mode=='secondary57':return values[:,57:]
    if mode=='dual114':return values
    raise ValueError('Unknown dual paired feature mode')


class DualPairedNetwork(nn.Module):
    def __init__(self,architecture,feature_mode):
        super().__init__();size={'primary57':57,'secondary57':57,'dual114':114}.get(feature_mode)
        if size is None:raise ValueError('Unknown dual paired feature mode')
        if architecture=='linear':self.head=nn.Linear(size,1)
        elif architecture=='mlp16':self.head=nn.Sequential(nn.Linear(size,16),nn.Tanh(),nn.Linear(16,1))
        else:raise ValueError('Unknown dual paired architecture')

    def forward(self,values):return self.head(values).squeeze(-1)


def score(network,state,features):
    x=np.asarray(features,np.float32)
    if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(x).all():raise ValueError('Invalid dual paired matrix')
    values=select_inputs(np.clip((x-state['mean'])/state['scale'],-8,8).astype(np.float32),state['feature_mode'])
    with torch.inference_mode():return torch.sigmoid(network(torch.from_numpy(values.copy()))).cpu().numpy()


def decisions(state,features,probabilities):
    output=[]
    for x,p,eligible in zip(features,probabilities,supported(features),strict=True):
        op=state['operating_points']['vowel_deletion' if x[16] else 'vowel_quality']
        output.append('strong_vowel_change' if eligible and p>=op['high'] else 'no_strong_vowel_change' if p<=op['low'] else 'uncertain')
    return output


def load(path:Path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['schema']!='dual_paired_vowel_v28' or state['feature_names']!=FEATURE_NAMES:raise ValueError('Invalid dual paired schema')
    network=DualPairedNetwork(state['architecture'],state['feature_mode']);network.load_state_dict(state['weights'],strict=True)
    return network.eval(),state
