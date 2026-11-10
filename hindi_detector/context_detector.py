"""Research fallback heads for explicitly unknown surrounding consonants."""
from __future__ import annotations

from pathlib import Path

import torch

from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_paired_detector import DualPairedNetwork
from hindi_detector.dual_paired_vowels import FEATURE_NAMES
from hindi_detector.task_vowel_detector import score, decisions


def load(path: Path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['schema'] != 'context_task_vowels_v61' or state['feature_names'] != FEATURE_NAMES:
        raise ValueError('Unsupported context fallback checkpoint schema')
    root = Path(__file__).resolve().parents[1]
    for name, digest in state['code_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError('Context fallback source dependency changed: ' + name)
    if sha(root / 'data/panphon_context_v58/inventory.json') != state['class_resource_inventory_sha256']:
        raise ValueError('Context class resource changed')
    if set(state['heads']) != {'vowel_deletion', 'vowel_quality'}:
        raise ValueError('Both cause-specific heads required')
    networks = {}
    for kind, head in state['heads'].items():
        network = DualPairedNetwork(head['architecture'], head['feature_mode'])
        network.load_state_dict(head['weights'], strict=True)
        networks[kind] = network.eval()
    return networks, state
