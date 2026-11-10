"""Stress-test frozen vowel heads when adjacent consonant identity is unknown."""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import traceback

import numpy as np
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.consonant_context import NonSyllabicTable
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_runtime import PhonePair
from hindi_detector.evaluation import binary_metrics
from hindi_detector.intrinsic_vowels import measure
from hindi_detector.paired_local_vowels import vector
from hindi_detector.pronunciation_sites import windows
from hindi_detector.task_vowel_detector import decisions, load, score
from scripts.extract_corrected_training import read_rows
from scripts.prepare_intrinsic_minimal import template
from scripts.score_swar_sites import load_item

ROOT = Path(__file__).resolve().parents[1]
COHORTS = [
    ('gap_v32', 'reports/fresh_gap_audit_v32', 'data/minimal_audit_v29', 'reports/minimal_audit_features_v29'),
    ('quality_v38', 'reports/release_quality_gap_audit_v38', 'data/minimal_quality_audit_v38', 'reports/minimal_quality_features_v38'),
]
MODES = ('strict', 'left_context_unknown', 'right_context_unknown', 'all_consonants_unknown')


def context_template(strict, ipa, vocabulary, table, mode):
    compiled, variant, contrast = copy.deepcopy(strict)
    at = contrast['index']
    consonants = [i for i, phone in enumerate(variant['phones']) if table.classify(ipa[phone])['status'] == 'pure_non_syllabic']
    left = [i for i in consonants if i < at]
    right = [i for i in consonants if i > at]
    selected = [] if mode == 'strict' else [left[-1]] if mode == 'left_context_unknown' and left else [right[0]] if mode == 'right_context_unknown' and right else consonants if mode == 'all_consonants_unknown' else []
    for index in selected:
        emission, _, projection = table.representation(ipa[variant['phones'][index]], vocabulary, mask_all=True)
        variant['units'][index] = emission
        variant['projections'][index] = projection
    return compiled, variant, contrast, selected


def main():
    out = ROOT / 'reports/consonant_control_audit_v59'
    if out.exists():
        raise FileExistsError(out)
    model = ROOT / 'reports/task_vowel_release_v31/model.pt'
    networks, state = load(model)
    for name, digest in state['code_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen release evidence dependency changed')
    write_group(out, {'protocol.json': {'model_sha256': sha(model), 'modes': MODES,
        'inputs': COHORTS, 'fit_or_threshold_updates': False,
        'scope': 'Known physically realized vowel edits, not real Hindi-error truth.',
        'strict_reproduction_tolerance': .0001,
        'admissibility': 'No new Native-control flags; compare every partial/fully unknown context to strict on the exact same physical WAVs. Unavailable rows remain denominator/abstention, never silently excluded.'},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
                               for p in (Path(__file__), ROOT / 'hindi_detector/consonant_context.py')]},
        'Frozen-head class-context stress test on64 existing full-vowel-region physical pairs. '
        'items.jsonl retains exact source/WAV SHA,114 features and decisions for strict, '
        'nearest-left, nearest-right and all-consonant unknown contexts. No G2P contrast '
        'or audio edits changed; only contextual acoustic units are deliberately pooled. '
        'Strict re-encoding must reproduce existing features within stated tolerance. '
        'report.json all-row counts, explicit abstentions and changes versus strict; '
        'runtime.json exact token pools/probe identities. No model fitting or source/error '
        'labels. Current consumer unknown-consonant admission decision, review2026-11-10. '
        'Compact evidence retained; no new waveform/weight/frame copies, NOT on HF, '
        'inherited acoustic/MMS/source rights and upstream PanPhon licence apply.',
        ['.venv-indicf5/bin/python -m scripts.audit_context_controls'])

    def status(phase, **values):
        p = out / 'status.json'
        temp = p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(), **values}, indent=2) + '\n')
        temp.replace(p)

    records = []
    try:
        status('loading_frozen_phone_runtime')
        runtime = PhonePair(ROOT, state)
        table = NonSyllabicTable(ROOT / 'data/panphon_context_v58')
        ipa = json.loads((ROOT / 'data/swar_phone_plan_v18/ipa_map.json').read_text())
        (out / 'runtime.json').write_text(json.dumps({'phone_inventory': state['phone_inventory'],
            'secondary_phone_inventory': state['secondary_phone_inventory'],
            'context_pools': [table.pool(v)[1] for v in runtime.vocabularies],
            'resource_inventory_sha256': sha(ROOT / 'data/panphon_context_v58/inventory.json')}, ensure_ascii=False, indent=2) + '\n')
        max_drift = 0.
        for cohort, controls_name, plan_name, frames_name in COHORTS:
            controls, plan, frames = [ROOT / p for p in (controls_name, plan_name, frames_name)]
            protocol = json.loads((controls / 'protocol.json').read_text())
            if protocol['source_plan_sha256'] != sha(plan / 'selection.json'):
                raise ValueError('Control source-plan identity differs')
            selection = json.loads((plan / 'selection.json').read_text())
            wordbank = {word['word_id']: word for word in selection['words']}
            receipts = {row['id']: row for row in read_rows(frames / 'items.jsonl')}
            reference_cache = {}
            rows = read_rows(controls / 'items.jsonl')
            for row in rows:
                receipt = receipts[row['job_id']]
                if row['job_id'] not in reference_cache:
                    original = load_item(frames, receipt)
                    fp = frames / receipt['features']
                    if sha(fp) != receipt['features_sha256']:
                        raise ValueError('Control reference frame file changed')
                    with np.load(fp, allow_pickle=False) as arrays:
                        secondary = arrays['secondary_phone_log_probs'].astype(np.float32)
                    reference_cache[row['job_id']] = original, secondary
                original, secondary = reference_cache[row['job_id']]
                ordinals = [i for i, w in enumerate(original['detail']['words']) if w['word'] == row['word']]
                if len(ordinals) != 1:
                    raise ValueError('Focal word does not occur exactly once in verified carrier')
                original_word = original['detail']['words'][ordinals[0]]
                for bound in ('start', 'end'):
                    if abs(original_word[bound] - row['geometry']['original_word'][bound]) > 1e-8:
                        raise ValueError('Original control word geometry changed')
                mapped = {**original_word, **row['geometry']['mapped_word']}
                references = [
                    {'detail': {'words': [original_word], 'phone_frame_seconds': original['detail']['phone_frame_seconds']}, 'phone_log_probs': original['phone_log_probs']},
                    {'detail': {'words': [original_word], 'phone_frame_seconds': original['detail']['secondary_phone_frame_seconds']}, 'phone_log_probs': secondary},
                ]
                path = controls / row['audio']
                if sha(path) != row['audio_sha256']:
                    raise ValueError('Actual physical control waveform changed')
                samples, rate = sf.read(path, dtype='float32')
                if rate != 16000 or samples.ndim != 1:
                    raise ValueError('Physical control format differs')
                lp = runtime.encode(samples)
                word = wordbank[row['word_id']]
                strict = [template(word, ipa, vocabulary) for vocabulary in runtime.vocabularies]
                for mode in MODES:
                    record = {'cohort': cohort, 'job_id': row['job_id'], 'word': row['word'], 'word_id': row['word_id'],
                        'kind': row['kind'], 'label': row['label'], 'mode': mode, 'audio_sha256': row['audio_sha256']}
                    try:
                        features = []
                        masked = []
                        for probe, vocabulary in enumerate(runtime.vocabularies):
                            compiled, variant, contrast, selected = context_template(strict[probe], ipa, vocabulary, table, mode)
                            query = {'detail': {'words': [mapped], 'phone_frame_seconds': len(samples) / 16000 / len(lp[probe])}, 'phone_log_probs': lp[probe]}
                            q, qe = measure(compiled, variant, contrast, windows(query, 0))
                            rf, re = measure(compiled, variant, contrast, windows(references[probe], 0))
                            features.append(vector(q, rf, qe, re, row['kind']))
                            masked.append(selected)
                        x = np.concatenate(features)[None].astype(np.float32)
                        p = score(networks, state, x)
                        record.update(status='measured', features=x[0].tolist(), score=float(p[0]),
                                      decision=decisions(state, x, p)[0], masked_native_indices=masked)
                        if mode == 'strict':
                            drift = float(np.max(np.abs(x[0] - row['features'])))
                            max_drift = max(max_drift, drift)
                            if drift > .0001 or record['decision'] != row['release']['decision']:
                                raise RuntimeError('Strict original control features/decision no longer reproduce')
                    except ValueError as error:
                        logging.warning('Explicit pooled-context abstention %s/%s: %s', row['word'], mode, error)
                        record.update(status='unavailable', reason=str(error), decision='uncertain', score=None)
                    records.append(record)
                    with (out / 'items.jsonl').open('a') as f:
                        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                status('measuring_context_masks', source_waveforms=len(records) // len(MODES), completed_records=len(records))
        metrics = {}
        for mode in MODES:
            metrics[mode] = {}
            for kind in ('vowel_deletion', 'vowel_quality'):
                subset = [row for row in records if row['mode'] == mode and row['kind'] == kind]
                labels = np.array([row['label'] for row in subset])
                predictions = np.array([row['decision'] == 'strong_vowel_change' for row in subset], float)
                metrics[mode][kind] = {'all_rows': binary_metrics(labels, predictions, .5),
                                      'unavailable': sum(row['status'] == 'unavailable' for row in subset)}
        baseline = {(row['audio_sha256'], row['cohort']): row for row in records if row['mode'] == 'strict'}
        changes = {mode: {'added_negative_flags': sum(row['label'] == 0 and row['decision'] == 'strong_vowel_change'
            and baseline[row['audio_sha256'], row['cohort']]['decision'] != 'strong_vowel_change' for row in records if row['mode'] == mode),
            'lost_positive_flags': sum(row['label'] == 1 and row['decision'] != 'strong_vowel_change'
            and baseline[row['audio_sha256'], row['cohort']]['decision'] == 'strong_vowel_change' for row in records if row['mode'] == mode)} for mode in MODES[1:]}
        report = {'status': 'complete_frozen_context_control_audit', 'model_sha256': sha(model),
            'physical_pairs': len(records) // (2 * len(MODES)), 'word_families': len({row['word'] for row in records}),
            'strict_max_absolute_feature_drift': max_drift, 'metrics': metrics, 'changes_vs_strict': changes,
            'actual_hindi_error_accuracy': None, 'production_changed': False}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        status(report['status'], pairs=report['physical_pairs'])
        print(json.dumps(report))
    except Exception:
        status('failed_explicitly', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Unknown-consonant controlled audit failed explicitly; production remains unchanged')
        raise
