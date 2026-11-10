"""Unknown-context decisions bounded by measured no-change score noise."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_paired_detector import DualPairedNetwork
from hindi_detector.dual_paired_vowels import FEATURE_NAMES
from hindi_detector.task_vowel_detector import decisions as base_decisions, score


def decisions(state, features, probabilities):
    values = np.asarray(features, np.float32)
    results = base_decisions(state, values, probabilities)
    floor = state['null_effect_floor']['vowel_deletion']
    if not np.isfinite(floor) or floor < 0:
        raise ValueError('Invalid measured null-effect floor')
    for index, row in enumerate(values):
        if row[16] and results[index] == 'strong_vowel_change' and min(row[9], row[66]) <= floor:
            results[index] = 'uncertain'
    return results


def load(path: Path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['schema'] != 'context_task_vowels_v61b' or state['feature_names'] != FEATURE_NAMES:
        raise ValueError('Unsupported null-calibrated context checkpoint')
    root = Path(__file__).resolve().parents[1]
    for name, digest in state['code_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError('Calibrated context source changed: ' + name)
    if sha(root / 'data/panphon_context_v58/inventory.json') != state['class_resource_inventory_sha256']:
        raise ValueError('Unknown-context feature resource changed')
    networks = {}
    if set(state['heads']) != {'vowel_deletion', 'vowel_quality'}:
        raise ValueError('Both cause-specific heads required')
    for kind, head in state['heads'].items():
        network = DualPairedNetwork(head['architecture'], head['feature_mode'])
        network.load_state_dict(head['weights'], strict=True)
        networks[kind] = network.eval()
    return networks, state
