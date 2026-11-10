"""Open the new unsupported-context controls after candidate/noise-floor freeze."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.context_calibrated import load, score, decisions
from hindi_detector.corrected_synthesis import sha
from hindi_detector.evaluation import binary_metrics
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def metrics(rows):
    result = {}
    for kind in ('vowel_deletion', 'vowel_quality'):
        selected = [row for row in rows if row['kind'] == kind]
        if selected:
            result[kind] = {'counts': binary_metrics(np.array([row['label'] for row in selected]),
                np.array([row['decision'] == 'strong_vowel_change' for row in selected], float), .5),
                'word_families': len({row['word'] for row in selected}),
                'conditioning_groups': sorted({row['voice'] for row in selected}),
                'uncertain_rows': sum(row['decision'] == 'uncertain' for row in selected)}
    return result


def main():
    directory = ROOT / 'reports/full_context_evaluation_v62'
    out = ROOT / 'reports/context_evaluation_v63'
    model = ROOT / 'reports/context_fallback_v61b/model.pt'
    if out.exists():
        raise FileExistsError(out)
    if json.loads((directory / 'report.json').read_text())['status'] != 'complete_full_region_context_pairs':
        raise ValueError('Fresh physical controls incomplete')
    frozen = json.loads((model.parent / 'frozen_selection.json').read_text())
    if sha(model) != frozen['model_sha256'] or frozen['v62_scores_opened']:
        raise ValueError('Candidate/model freeze receipt differs')
    networks, state = load(model)
    # Threshold/data-adequacy policy is recorded before loading any held-out
    # feature vector for scoring. This script never refits or retunes a head.
    write_group(out, {'protocol.json': {'model_sha256': sha(model), 'frozen_utc': frozen['created_utc'],
        'physical_items_sha256': sha(directory / 'items.jsonl'),
        'content_receipts_sha256': sha(ROOT / 'reports/minimal_context_content_v62/result.json'),
        'scope': 'Known full-vowel physical controls on new unsupported-context words; not real Hindi-error gold.',
        'admission_per_kind': {'minimum_positive_and_negative_readings': 10, 'minimum_word_families': 6,
            'native_control_flags': 0, 'minimum_deletion_recall': .75, 'minimum_quality_recall': .6},
        'counts': 'Base edit pairs only; source/reference conditioning groups disjoint from fitting; no context-view augmentation in primary test.',
        'policy': 'Report every physically eligible pair, plus strict source/reference/donor-content-pass subset; no test-driven threshold adjustments.'}},
        'Fresh unsupported-context candidate evaluation after model/noise bound freeze. '
        'protocol.json identities and admission policy before score loading; scores.json '
        'all known physical rows with content acceptance and scored features/decision; '
        'report.json both all-eligible/content-pass counts and per-cause gates. No weight '
        'change or automatic production promotion. Current consumer context-fallback '
        'admission, owner Ampixa/user, review2026-11-10. Keep compact evaluation/input pins '
        'while consumed; no raw audio/frame/encoder copy or HF upload. Source/MMS/PanPhon '
        'rights inherited; data qualification by shared probes limits independence.',
        ['.venv-indicf5/bin/python -m scripts.evaluate_context_fallback'])
    content = json.loads((ROOT / 'reports/minimal_context_content_v62/result.json').read_text())
    by_id = {row['id']: row for row in content['rows']}
    accepted = set(content['accepted_ids'])
    fitted = json.loads((ROOT / 'reports/context_fallback_v61/fitting_rows.json').read_text())
    fit_words = {row['word'] for row in fitted}
    fit_groups = {row[key] for row in fitted for key in ('voice', 'reference_voice')}
    records = []
    for row in read_rows(directory / 'items.jsonl'):
        if row['word'] in fit_words or {row['voice'], row['reference_voice']} & fit_groups:
            raise ValueError('Fresh word/conditioning group overlaps fitting')
        if sha(directory / row['audio']) != row['audio_sha256']:
            raise ValueError('Fresh physical waveform identity changed')
        needed = {row['job_id'], row['reference_job_id']}
        if row['donor_job_id'] is not None:
            needed.add(row['donor_job_id'])
        if by_id[row['job_id']]['sha256'] != row['source_audio_sha256'] or by_id[row['reference_job_id']]['sha256'] != row['reference_audio_sha256']:
            raise ValueError('Fresh content/source receipts differ')
        if row['donor_job_id'] is not None and by_id[row['donor_job_id']]['sha256'] != row['donor']['sha256']:
            raise ValueError('Fresh quality donor content receipt differs')
        x = np.asarray(row['modes']['strict']['features'], np.float32)[None]
        probability = score(networks, state, x)
        decision = decisions(state, x, probability)[0]
        records.append({'word': row['word'], 'kind': row['kind'], 'job_id': row['job_id'], 'reference_job_id': row['reference_job_id'],
            'voice': row['voice'], 'reference_voice': row['reference_voice'], 'label': row['label'],
            'source_content_pass': needed <= accepted,
            'source_content_decisions': {key: by_id[key]['decision'] for key in sorted(needed)},
            'audio_sha256': row['audio_sha256'], 'features': x[0].tolist(), 'controlled_edit_score': float(probability[0]),
            'decision': decision, 'actual_hindi_error_label': None})
    passed = [row for row in records if row['source_content_pass']]
    all_metrics, pass_metrics = metrics(records), metrics(passed)
    admission = {}
    for kind in ('vowel_deletion', 'vowel_quality'):
        value = pass_metrics.get(kind)
        if value is None:
            admission[kind] = {'admitted': False, 'reason': 'No content-qualified physical controls'}
            continue
        counts = value['counts']
        enough = min(counts['positive_labels'], counts['negative_labels']) >= 10 and value['word_families'] >= 6
        recall_floor = .75 if kind == 'vowel_deletion' else .6
        admitted = enough and counts['fp'] == 0 and counts['recall'] >= recall_floor
        admission[kind] = {'admitted': bool(admitted), 'enough_data': enough, 'recall_floor': recall_floor,
                           'actual_hindi_error_accuracy': None}
    report = {'status': 'complete_fresh_context_candidate_evaluation', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_sha256': sha(model), 'all_physical_pairs': len(records) // 2, 'all_word_families': len({row['word'] for row in records}),
        'content_pass_pairs': len(passed) // 2, 'all_eligible': all_metrics, 'content_pass': pass_metrics,
        'admission': admission, 'native_source_requested_word_pool': 64, 'native_generated_waveforms': 256,
        'actual_hindi_error_accuracy': None, 'production_changed': False,
        'limits': ['Only clearly observed Native source/reference vowels enter physical controls; qualification uses the same two phone probes.',
                   'Two conditioning groups and few words give limited evidence about general speakers/contexts.',
                   'Source labels are not Hindi-error labels; physical-edit detection does not establish Hindi-specific prosody accuracy.']}
    for name, value in [('scores.json', records), ('report.json', report)]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if sha(model) != frozen['model_sha256']:
        raise ValueError('Candidate changed after held-out scoring')
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Fresh context evaluation failed explicitly; do not retune test scores or promote')
        raise
