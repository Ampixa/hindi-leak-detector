"""Unfiltered canonical/edited TTS-request evidence; never automatic gold."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.anchored_vowels import measure
from hindi_detector.artifacts import write_group
from hindi_detector.aspirate_context import AspirateContextTable
from hindi_detector.context_templates import template
from hindi_detector.corrected_synthesis import sha
from hindi_detector.pronunciation_sites import windows
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', required=True, choices=['train', 'development', 'test'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    plan, frames = ROOT / 'data/native_main_v71', ROOT / 'reports/native_main_features_v71'
    selection = json.loads((plan / 'selection.json').read_text())
    words = {word['word_id']: word for word in selection['words']}
    receipts = {row['id']: row for row in read_rows(frames / 'items.jsonl')}
    jobs = [job for job in selection['jobs'] if job['split'] == args.partition]
    write_group(args.out, {'protocol.json': {'partition': args.partition, 'plan_sha256': sha(plan / 'selection.json'),
        'scope': 'Every requested canonical/edited render, including sources ineligible for physical controls. No generator request is asserted to be observed-correct or observed-error.',
        'labels': None, 'source_filter': None, 'classifier_scores': False}},
        'Unfiltered requested-readings diagnostic. items.jsonl exact raw-source/frame identities,24 measured local features or explicit abstention, requested text edit and null realized label. '
        'report.json all requested/measurement counts, not accuracy. No new waveform/encoder/model copy, no source filtering or fitting. Owner Ampixa/user, review2026-11-10; active consumer '
        'check selected-source bias and splice-free requested-deletion behavior after a frozen fit. Keep compact evidence while benchmark consumes it; NOT on HF, inherited source/MMS/PanPhon rights.',
        ['python -m scripts.measure_native_requests --partition PART --out NEW'])
    vocabulary = [json.loads((ROOT / 'models' / folder / 'vocab.json').read_text()) for folder in ('phoneme_probe', 'phoneme_probe_lv60_v20')]
    table = AspirateContextTable(ROOT / 'data/panphon_context_v58')
    ipa = json.loads((ROOT / 'data/swar_phone_plan_v18/ipa_map.json').read_text())
    records = []
    for job in jobs:
        record = {'job_id': job['id'], 'pair_id': job['id'].rsplit('_c', 1)[0], 'word': words[job['word_id']]['word'],
            'word_id': job['word_id'], 'voice': job['speaker_id'], 'split': job['split'],
            'phone_collision_group': words[job['word_id']]['phone_collision_group'],
            'requested_change_class': job['requested_vowel_change_class'], 'realized_error_label': None,
            'source_audio_sha256': receipts[job['id']]['sha256']}
        try:
            receipt = receipts[job['id']]
            item = load_item(frames, receipt)
            at = [i for i, word in enumerate(item['detail']['words']) if word['word'] == record['word']]
            if len(at) != 1:
                raise ValueError('Expected canonical target not uniquely aligned')
            with np.load(frames / receipt['features'], allow_pickle=False) as arrays:
                arrays = [arrays['phone_log_probs'].astype(float), arrays['secondary_phone_log_probs'].astype(float)]
            features, evidence = [], []
            for probe in (0, 1):
                detail = {**item['detail'], 'phone_frame_seconds': item['detail']['seconds'] / len(arrays[probe])}
                vector, local = measure(*template(words[job['word_id']], ipa, vocabulary[probe], table),
                                        windows({'detail': detail, 'phone_log_probs': arrays[probe]}, at[0]))
                features.extend(vector.tolist())
                evidence.append(local)
            record.update(status='measured', features=features, evidence=evidence)
        except ValueError as error:
            logging.warning('Retained unfiltered request abstention %s: %s', job['id'], error)
            record.update(status='abstain', features=None, reason=str(error))
        records.append(record)
        with (args.out / 'items.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    report = {'status': 'complete_unfiltered_request_evidence', 'requested_rows': len(records),
        'outcomes': dict(Counter(str(row['requested_change_class']) + '/' + row['status'] for row in records)),
        'realized_error_labels': None, 'accuracy': None, 'items_sha256': sha(args.out / 'items.jsonl')}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Unfiltered request measurement failed explicitly; do not invent labels')
        raise
