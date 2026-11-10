"""Reference-conditioned word source candidate; outputs are not accent probabilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

BRANCHES = ('phonetics', 'prosody', 'combined')
FEATURE_KEYS = ('phonetics', 'prosody', 'hidden_word_12', 'hidden_word_24')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_feature(directory: Path, receipt: dict) -> dict:
    if receipt['status'] != 'ok_word_features':
        raise ValueError('Cannot load an explicitly failed word extraction')
    result = {'receipt': receipt}
    for name, digest_key in (('detail', 'detail_sha256'), ('features', 'features_sha256')):
        p = (directory / receipt[name]).resolve(strict=True)
        if not p.is_relative_to(directory.resolve()) or sha(p) != receipt[digest_key]:
            raise ValueError('Word artifact path or checksum mismatch')
        if name == 'detail':
            result['detail'] = json.loads(p.read_text())
        else:
            with np.load(p, allow_pickle=False) as arrays:
                result.update({key: np.asarray(arrays[key], dtype=np.float32) for key in FEATURE_KEYS})
    n = len(result['detail']['words'])
    if any(v.ndim != 2 or len(v) != n or not np.isfinite(v).all() for v in (result[k] for k in FEATURE_KEYS)):
        raise ValueError('Actual word feature geometry or values invalid')
    return result


def paired_features(query: dict, reference: dict, projection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qw = [w['word'] for w in query['detail']['words']]
    rw = [w['word'] for w in reference['detail']['words']]
    if qw != rw or not qw:
        raise ValueError('Reference must contain the exact same aligned word sequence')
    hidden = lambda r: np.concatenate([r['hidden_word_12'], r['hidden_word_24']], axis=1)
    qh, rh = hidden(query), hidden(reference)
    if projection.ndim != 2 or qh.shape[1] != projection.shape[0] or qh.shape != rh.shape:
        raise ValueError('Pinned hidden projection and actual encoder dimensions disagree')
    pd = query['phonetics'] - reference['phonetics']
    hd = (qh - rh) @ projection
    phonetic = np.concatenate([pd, np.abs(pd), hd, np.abs(hd)], axis=1).astype(np.float32)
    sd = query['prosody'] - reference['prosody']
    prosody = np.concatenate([sd, np.abs(sd)], axis=1).astype(np.float32)
    if not np.isfinite(phonetic).all() or not np.isfinite(prosody).all():
        raise ValueError('Nonfinite paired word inputs')
    return phonetic, prosody


class PairedWordNetwork(nn.Module):
    def __init__(self, phonetic_dimension: int, prosody_dimension: int):
        super().__init__()
        self.phonetic = nn.Sequential(nn.Linear(phonetic_dimension, 96), nn.LayerNorm(96), nn.GELU(),
                                      nn.Dropout(.25), nn.Linear(96, 48), nn.GELU())
        self.prosody = nn.Sequential(nn.Linear(prosody_dimension, 48), nn.LayerNorm(48), nn.GELU(),
                                     nn.Dropout(.2), nn.Linear(48, 24), nn.GELU())
        self.phonetic_head = nn.Linear(48, 1)
        self.prosody_head = nn.Linear(24, 1)
        self.combined_head = nn.Sequential(nn.Linear(72, 32), nn.GELU(), nn.Dropout(.15), nn.Linear(32, 1))

    def forward(self, phonetics, prosody):
        p, s = self.phonetic(phonetics), self.prosody(prosody)
        return torch.cat([self.phonetic_head(p), self.prosody_head(s), self.combined_head(torch.cat([p, s], dim=-1))], dim=-1)


def score_arrays(network: PairedWordNetwork, state: dict, phonetics: np.ndarray, prosody: np.ndarray) -> np.ndarray:
    arrays = []
    for name, values in (('phonetics', phonetics), ('prosody', prosody)):
        mean, scale = state['normalization'][name]['mean'], state['normalization'][name]['scale']
        arrays.append(torch.as_tensor(np.clip((values - mean) / scale, -8, 8), dtype=torch.float32))
    network.eval()
    with torch.inference_mode():
        result = network(*arrays).sigmoid().numpy()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite detector scores')
    return result


def load_model(path: Path) -> tuple[PairedWordNetwork, dict]:
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['schema'] != 'paired_word_source_candidate_v11':
        raise ValueError('Unsupported detector checkpoint recipe')
    network = PairedWordNetwork(**state['dimensions'])
    network.load_state_dict(state['state_dict'], strict=True)
    return network.eval(), state
