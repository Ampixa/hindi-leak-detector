"""Descriptive human-variation context; never an input to model decisions."""
from __future__ import annotations

import json
from pathlib import Path

from hindi_detector.corrected_synthesis import sha


def load_bank(directory: Path, model: Path) -> dict | None:
    path = directory / 'native_observations.json'
    if not path.exists():
        return None
    bank = json.loads(path.read_text())
    if bank['schema'] != 'observed_real_variation_v1' or bank['model_sha256'] != sha(model):
        raise ValueError('Observed-variation bank model identity differs')
    return bank


def matching_observations(bank: dict | None, word: str, site: dict, native_phones: list, *, exclude_case: str | None = None) -> dict:
    if bank is None:
        return {'status': 'bank_unavailable', 'used_for_prediction': False}
    entries = [entry for entry in bank['entries'] if entry['word'] == word
               and entry['site']['kind'] == site['kind'] and entry['site']['index'] == site['index']
               and entry['native_phones'] == native_phones
               and entry['site'].get('alternative') == site.get('alternative')]
    cases = [case for entry in entries for case in entry['recordings'] if case['id'] != exclude_case]
    return {'status': 'observed_in_other_real_recordings' if cases else 'not_observed_in_limited_bank',
            'cases': cases, 'source_groups': len({case['voice'] for case in cases}),
            'human_error_label': None, 'used_for_prediction': False,
            'meaning': 'Observed probe-supported alternative in recorded speech, potentially including contextual or regional variation; acceptability and Hindi influence remain unconfirmed.'}
