"""Temporal phone/site and contextual prosody inputs for paired word candidates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hindi_detector.paired_detector import PairedWordNetwork, sha, score_arrays

BRANCHES=('phonetics','prosody','combined')
SITE_NAMES=('available','margin_mean','margin_std','margin_min','margin_max','positive_window_fraction',
            'normalized_gap','native_count','alternative_count','supported_site_count')


def site_vector(word: dict) -> np.ndarray:
    values=[];positions=[];native=[];alternative=[];sites=0
    for candidate in word.get('schwa_site_evidence',[]):
        for site in candidate.get('sites',[]):
            supported=[w['alternative_minus_lexicon_per_frame'] for w in site['windows'] if w['status']=='supported']
            if len(supported)!=3:continue
            sites+=1;values.extend(supported)
            positions.append(site['gap_index']/max(1,site['skeleton_length']))
            native.append(site['lexicon_schwa_count']);alternative.append(site['alternative_schwa_count'])
    if not values:return np.zeros(len(SITE_NAMES),np.float32)
    a=np.array(values)
    if not np.isfinite(a).all():raise ValueError('Nonfinite measured site evidence')
    return np.array([1,a.mean(),a.std(),a.min(),a.max(),np.mean(a>0),np.mean(positions),np.mean(native),np.mean(alternative),sites],np.float32)


def context(values: np.ndarray) -> np.ndarray:
    # Same-text ordinal neighbors, with explicit availability masks at boundaries.
    pieces=[]
    for offset in (-2,-1,0,1,2):
        indices=np.arange(len(values))+offset;available=(indices>=0)&(indices<len(values))
        block=np.zeros_like(values);block[available]=values[indices[available]]
        pieces.extend([block,available[:,None].astype(np.float32)])
    return np.concatenate(pieces,axis=1)


def paired_inputs(query: dict, reference: dict) -> tuple[np.ndarray,np.ndarray]:
    a,b=query['detail']['words'],reference['detail']['words']
    if [w['word'] for w in a]!=[w['word'] for w in b] or not a:
        raise ValueError('Structured comparison requires identical aligned word sequences')
    for item in (query,reference):
        if item['temporal_phone'].shape!=(len(a),20,8):raise ValueError('Wrong temporal phone geometry')
    temporal=query['temporal_phone'].reshape(len(a),-1)-reference['temporal_phone'].reshape(len(a),-1)
    # Mean posterior by phone; hidden speech embeddings are deliberately absent.
    dimension=query['phonetics'].shape[1]//2
    posterior=query['phonetics'][:,:dimension]-reference['phonetics'][:,:dimension]
    qs=np.stack([site_vector(w) for w in a]);rs=np.stack([site_vector(w) for w in b]);sd=qs-rs
    p=np.concatenate([posterior,np.abs(posterior),temporal,np.abs(temporal),qs,rs,sd,np.abs(sd)],axis=1)
    qpros=context(query['prosody']);rpros=context(reference['prosody']);delta=qpros-rpros
    s=np.concatenate([delta,np.abs(delta)],axis=1)
    if not np.isfinite(p).all() or not np.isfinite(s).all():raise ValueError('Nonfinite structured paired features')
    return p.astype(np.float32),s.astype(np.float32)


def load_feature(directory: Path,receipt: dict) -> dict:
    if receipt['status']!='ok_structured_features':raise ValueError('Cannot load failed/quarantined structured evidence')
    result={}
    for name,key in [('detail','detail_sha256'),('features','features_sha256')]:
        p=(directory/receipt[name]).resolve(strict=True)
        if not p.is_relative_to(directory.resolve()) or sha(p)!=receipt[key]:raise ValueError('Structured feature identity/path mismatch')
        if name=='detail':result['detail']=json.loads(p.read_text())
        else:
            with np.load(p,allow_pickle=False) as arrays:
                result.update({k:arrays[k].astype(np.float32) for k in ('phonetics','prosody','temporal_phone')})
    if any(not np.isfinite(result[k]).all() for k in ('phonetics','prosody','temporal_phone')):raise ValueError('Invalid retained structured arrays')
    return result


def load_model(path: Path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['schema']!='structured_word_source_v15':raise ValueError('Wrong structured model schema')
    network=PairedWordNetwork(**state['dimensions']);network.load_state_dict(state['state_dict'],strict=True)
    return network.eval(),state
