"""Separate learned deletion/quality heads with shared dual acoustic evidence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hindi_detector.dual_paired_detector import DualPairedNetwork,select_inputs
from hindi_detector.dual_paired_vowels import FEATURE_NAMES,supported


def score(networks,state,features):
    x=np.asarray(features,np.float32)
    if x.ndim!=2 or x.shape[1]!=114 or not np.isfinite(x).all():raise ValueError('Invalid task-specific dual features')
    if not np.all(x[:,16]+x[:,17]==1):raise ValueError('Every measured site must have one vowel-change kind')
    result=np.empty(len(x),np.float32)
    for kind,index in [('vowel_deletion',16),('vowel_quality',17)]:
        mask=x[:,index]>0
        if not mask.any():continue
        head=state['heads'][kind];values=select_inputs(x[mask],head['feature_mode'])
        normalized=np.clip((values-head['mean'])/head['scale'],-8,8).astype(np.float32)
        with torch.inference_mode():result[mask]=torch.sigmoid(networks[kind](torch.from_numpy(normalized.copy()))).cpu().numpy()
    return result


def decisions(state,features,probabilities):
    result=[]
    for x,p,eligible in zip(features,probabilities,supported(features),strict=True):
        kind='vowel_deletion' if x[16] else 'vowel_quality';op=state['heads'][kind]['operating_points']
        result.append('strong_vowel_change' if eligible and p>=op['high'] else 'no_strong_vowel_change' if p<=op['low'] else 'uncertain')
    return result


def load(path:Path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['schema']!='task_vowel_v30' or state['feature_names']!=FEATURE_NAMES:raise ValueError('Wrong task-vowel checkpoint schema')
    networks={}
    for kind,head in state['heads'].items():
        network=DualPairedNetwork(head['architecture'],head['feature_mode']);network.load_state_dict(head['weights'],strict=True);networks[kind]=network.eval()
    if set(networks)!={'vowel_deletion','vowel_quality'}:raise ValueError('Both task heads are required')
    return networks,state
