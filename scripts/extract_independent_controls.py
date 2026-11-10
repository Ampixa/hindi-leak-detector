"""Measure independent phone evidence on existing full-region physical controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from hindi_detector.artifacts import write_group
from hindi_detector.aspirate_context import AspirateContextTable
from hindi_detector.consonant_context import NonSyllabicTable
from hindi_detector.context_templates import template
from hindi_detector.corrected_synthesis import sha
from hindi_detector.independent_phone import IndependentPhone
from hindi_detector.intrinsic_vowels import FEATURE_NAMES, measure
from hindi_detector.pronunciation_sites import windows
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', action='store_true', help='Already-opened v62 diagnosis only; never independent fresh validation')
    parser.add_argument('--voiced-aspirates', action='store_true', help='Class-only context with explicit pinned-table nonsyllabicity witnesses')
    args = parser.parse_args()
    suffix = 'v64b' if args.voiced_aspirates else 'v64'
    out = ROOT / ('reports/independent_context_diagnosis_' + suffix if args.fresh else 'reports/independent_full_controls_' + suffix)
    if out.exists():
        raise FileExistsError(out)
    groups = [('full_context_evaluation_v62', 'minimal_context_audit_v62')] if args.fresh else [
        ('full_context_controls_v60', 'minimal_vowels_v23'), ('full_context_self_v60b', 'minimal_vowels_v23')]
    write_group(out, {'protocol.json': {'groups': groups, 'features': FEATURE_NAMES, 'class_only_voiced_aspirates': args.voiced_aspirates,
        'selection': 'Every existing full-region control; explicit unsupported-template exclusions only. No learned score input.',
        'fresh_independent_accuracy': False if args.fresh else None,
        'scope': 'Reference-free third-encoder vowel presence/quality evidence, not observed Hindi-error gold.',
        'retained_raw_encoder_arrays': False},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
            for p in (Path(__file__), ROOT / 'hindi_detector/independent_phone.py', ROOT / 'hindi_detector/aspirate_context.py')]},
        'Independent Allosaurus reference-free physical-control evidence. items.jsonl each exact existing '
        'WAV SHA, physical geometry, word,32 intrinsic features and unconditioned free phones; '
        'exclusions.json explicit template/measurement limitations; runtime.json source/model SHA and '
        'vocabulary; report.json counts. No new WAV/model/raw posterior copies. Current consumer '
        'query-only vowel-presence research; owner Ampixa/user, review2026-11-10. Retain compact '
        'evidence during task, not on HF. Third encoder code GPL-3.0, checkpoint terms not supplied, '
        'internal research only/no redistribution. Source audio/model rights inherited. Opened v62 '
        'controls are diagnosis and cannot be reused as independent validation after this change.',
        ['.venv-indicf5/bin/python -m scripts.extract_independent_controls' + (' --fresh' if args.fresh else '') + (' --voiced-aspirates' if args.voiced_aspirates else '')])
    torch.set_num_threads(4)
    runtime = IndependentPhone(ROOT)
    (out / 'runtime.json').write_text(json.dumps(runtime.metadata, ensure_ascii=False, indent=2) + '\n')
    table = (AspirateContextTable if args.voiced_aspirates else NonSyllabicTable)(ROOT / 'data/panphon_context_v58')
    ipa = json.loads((ROOT / 'data/swar_phone_plan_v18/ipa_map.json').read_text())
    records, exclusions = [], []
    for group, plan in groups:
        directory = ROOT / 'reports' / group
        words = {row['word_id']: row for row in json.loads((ROOT / 'data' / plan / 'selection.json').read_text())['words']}
        for row in read_rows(directory / 'items.jsonl'):
            path = directory / row['audio']
            if sha(path) != row['audio_sha256']:
                raise ValueError('Physical waveform changed: ' + str(path))
            try:
                compiled, variant, contrast = template(words[row['word_id']], ipa, runtime.vocabulary, table)
            except ValueError as error:
                logging.warning('Explicit independent template rejection %s: %s', row['job_id'], error)
                exclusions.append({'group': group, 'job_id': row['job_id'], 'label': row['label'], 'reason': str(error)})
                continue
            samples, rate = sf.read(path, dtype='float32')
            values = runtime.encode(samples, rate)
            item = {'detail': {'words': [row['geometry']['mapped_word']], 'phone_frame_seconds': runtime.frame_seconds}, 'phone_log_probs': values}
            try:
                vector, evidence = measure(compiled, variant, contrast, windows(item, 0))
            except ValueError as error:
                logging.warning('Explicit independent measurement rejection %s: %s', row['job_id'], error)
                exclusions.append({'group': group, 'job_id': row['job_id'], 'label': row['label'], 'reason': str(error)})
                continue
            record = {key: row[key] for key in ('word', 'word_id', 'job_id', 'split', 'kind', 'label', 'voice', 'reference_voice', 'geometry', 'audio_sha256')}
            record.update(group=group, features=vector.tolist(), evidence=evidence, free_decode=runtime.decode(values),
                          projections=variant['projections'], actual_hindi_error_label=None)
            records.append(record)
            with (out / 'items.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            (out / 'status.json').write_text(json.dumps({'phase': 'extracting', 'rows': len(records), 'group': group,
                'updated_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
    (out / 'exclusions.json').write_text(json.dumps(exclusions, ensure_ascii=False, indent=2) + '\n')
    report = {'status': 'complete_independent_physical_evidence', 'rows': len(records), 'excluded_rows': len(exclusions),
              'word_families': len({row['word'] for row in records}), 'actual_hindi_error_accuracy': None,
              'items_sha256': sha(out / 'items.jsonl'), 'production_changed': False}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Independent physical phone extraction failed explicitly; no candidate promotion')
        raise
