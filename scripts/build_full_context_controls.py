"""Full-vowel physical controls, cross-voice references, measured context masks."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
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
from hindi_detector.context_templates import template as unknown_template
from hindi_detector.dual_runtime import PhonePair
from hindi_detector.edit_geometry import map_word
from hindi_detector.intrinsic_vowels import measure
from hindi_detector.paired_local_vowels import vector
from hindi_detector.pronunciation_sites import windows
from hindi_detector.task_vowel_detector import load
from hindi_detector.teacher_reference_cache import verified_paths
from scripts.audit_context_controls import context_template, MODES
from scripts.audit_fresh_waveforms import full_gap, native_observed
from scripts.build_vowel_controls import splice
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows
from scripts.prepare_intrinsic_minimal import template
from scripts.score_swar_sites import load_item

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-reference-backoff', action='store_true', help='Additional controls only where other-voice vowel reference fails; never duplicate cross-voice controls')
    parser.add_argument('--context-evaluation', action='store_true', help='New held-out v62 unsupported-consonant words; no fitted classifier scores')
    args = parser.parse_args()
    if args.self_reference_backoff and args.context_evaluation:
        raise ValueError('Fresh evaluation requires an independently qualified other-voice reference')
    plan = ROOT / ('data/minimal_context_audit_v62' if args.context_evaluation else 'data/minimal_vowels_v23')
    frames = ROOT / ('reports/minimal_context_features_v62' if args.context_evaluation else 'reports/minimal_vowel_features_v23')
    intrinsic = ROOT / 'reports/intrinsic_minimal_v24'
    out = ROOT / ('reports/full_context_evaluation_v62' if args.context_evaluation else 'reports/full_context_self_v60b' if args.self_reference_backoff else 'reports/full_context_controls_v60')
    if out.exists():
        raise FileExistsError(out)
    selection = json.loads((plan / 'selection.json').read_text())
    jobs = {job['id']: job for job in selection['jobs']}
    wordbank = {word['word_id']: word for word in selection['words']}
    tuples = {(job['word_id'], job['voice_ordinal'], job['requested_vowel_change_class']): job['id'] for job in jobs.values()}
    if len(tuples) != len(jobs):
        raise ValueError('Native minimal-job tuple identities repeat')
    receipts = {row['id']: row for row in read_rows(frames / 'items.jsonl')}
    observed = {} if args.context_evaluation else {row['job_id']: row for row in read_rows(intrinsic / 'items.jsonl')}
    # All source waveform/runtime receipts are checked by the same strict24-
    # layer/tokenizer/revision/weights gate used by the live reference cache.
    verified_paths(plan, list(jobs.values()))
    actual = json.loads((ROOT / 'reports/task_vowel_release_v31/actual_swar/cases.json').read_text())
    excluded_words = {word['word'] for case in actual for word in case['analysis'].get('real', {}).get('words', [])} | {'हैन', 'किनकि'}
    write_group(out, {'protocol.json': {'source_plan_sha256': sha(plan / 'selection.json'),
        'source_frames_sha256': sha(frames / 'items.jsonl'), 'source_intrinsic_sha256': None if args.context_evaluation else sha(intrinsic / 'items.jsonl'),
        'partitions': ['test'] if args.context_evaluation else ['train', 'development'], 'original_test_arrays_never_used': not args.context_evaluation,
        'context_evaluation': args.context_evaluation, 'learned_scores_used_for_selection': False,
        'reference': ('Training-only self reference where source vowel is observed by both probes but other-voice reference is not qualified; cross-voice controls remain separate v60.' if args.self_reference_backoff else
                      'Opposite conditioning voice reading the exact same canonical text; qualified expected vowel in both probes.'),
        'negative': 'Reinsert the complete original observed vowel region with the same2ms join fades.',
        'positive': 'Remove complete observed vowel region, or replace it with full observed alternative from same24-layer teacher/conditioning voice edited-word render.',
        'geometry': '40..300ms fully owned vowel region between neighboring consonants; quality donor0.5..2 duration ratio; exact piecewise mapping of both word boundaries.',
        'modes': MODES, 'input_selection': 'Actual vowel-path support in both probes; no learned classifier score used.',
        'labels': 'Known physical full-vowel edits, not generator identity and not real Hindi-error labels.',
        'fit_gate': 'Source, cross-voice reference and optional quality donor must subsequently pass isolated lexical ASR before any fitting.',
        'all_actual_review_words_excluded': sorted(excluded_words)},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
            for p in (Path(__file__), ROOT / 'hindi_detector/consonant_context.py', ROOT / 'hindi_detector/edit_geometry.py',
                      ROOT / 'scripts/audit_context_controls.py', ROOT / 'scripts/audit_fresh_waveforms.py', ROOT / 'hindi_detector/context_templates.py')]},
        'New training/development data from full observed vowels, replacing short-peak edit '
        'supervision. controls/*.wav same-query-voice sham/edit waveforms; source and references '
        'are verified24-layer Native teacher, reference from a different conditioning group. '
        'items.jsonl stores every committed WAV/source SHA, exact geometry, original phonetic '
        'site and114 measured features for four context masks. exclusions.json explicit '
        'whole-pair exclusions; waveform_inventory.json every retained waveform including '
        'uncommitted diagnostics; source_job_ids.json exact canonical/reference/donor source '
        'jobs requiring ASR before fitting. No source model/encoder copies; no fitting in this '
        'builder, source identity never a positive label. Current consumer robust context '
        'fallback training, owner Ampixa/user, review2026-11-10; controls/frames disposable24h '
        'after fitting/admission consumers close; retain compact configs/metrics/selected '
        'checkpoint. NOT on HF; source/MMS/PanPhon licences apply.',
        ['.venv-indicf5/bin/python -m scripts.build_full_context_controls'])
    (out / 'controls').mkdir()
    cache = {}
    needed = set()
    records, exclusions = [], []

    def status(phase, **extra):
        p = out / 'status.json'
        temp = p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2) + '\n')
        temp.replace(p)

    def item(job_id):
        if job_id not in cache:
            receipt = receipts[job_id]
            loaded = load_item(frames, receipt)
            with np.load(frames / receipt['features'], allow_pickle=False) as arrays:
                other = arrays['secondary_phone_log_probs'].astype(np.float32)
            word = wordbank[jobs[job_id]['word_id']]['word']
            indices = [i for i, value in enumerate(loaded['detail']['words']) if value['word'] == word]
            if len(indices) != 1:
                raise ValueError('Canonical source target word not uniquely aligned')
            ordinal = indices[0]
            source_word = loaded['detail']['words'][ordinal]
            pair = [
                {'detail': {'words': [source_word], 'phone_frame_seconds': loaded['detail']['phone_frame_seconds']}, 'phone_log_probs': loaded['phone_log_probs']},
                {'detail': {'words': [source_word], 'phone_frame_seconds': loaded['detail']['secondary_phone_frame_seconds']}, 'phone_log_probs': other},
            ]
            cache[job_id] = loaded, ordinal, source_word, pair
        return cache[job_id]

    try:
        status('loading_verified_dual_encoders')
        _, state = load(ROOT / 'reports/task_vowel_release_v31/model.pt')
        runtime = PhonePair(ROOT, state)
        table = NonSyllabicTable(ROOT / 'data/panphon_context_v58')
        ipa = json.loads((ROOT / 'data/swar_phone_plan_v18/ipa_map.json').read_text())
        (out / 'runtime.json').write_text(json.dumps({'phone_inventory': state['phone_inventory'],
            'secondary_phone_inventory': state['secondary_phone_inventory'], 'source_teacher_runtime_sha256': sha(plan / 'teacher/runtime.json'),
            'context_pools': [table.pool(vocabulary)[1] for vocabulary in runtime.vocabularies]}, ensure_ascii=False, indent=2) + '\n')
        def declared_template(word, vocabulary):
            return unknown_template(word, ipa, vocabulary, table) if args.context_evaluation else template(word, ipa, vocabulary)

        def observation(job_id):
            if job_id not in observed:
                job = jobs[job_id]
                _, _, _, probes = item(job_id)
                values, evidence = [], []
                for probe, vocabulary in enumerate(runtime.vocabularies):
                    compiled, variant, contrast = declared_template(wordbank[job['word_id']], vocabulary)
                    vector_values, detail_values = measure(compiled, variant, contrast, windows(probes[probe], 0))
                    values.append(vector_values.tolist())
                    evidence.append(detail_values)
                observed[job_id] = {'features': values[0], 'secondary_features': values[1], 'evidence': evidence[0], 'secondary_evidence': evidence[1]}
            return observed[job_id]

        allowed_parts = {'test'} if args.context_evaluation else {'train', 'development'}
        canonical = [job for job in jobs.values() if job['requested_vowel_change_class'] == 0 and job['split'] in allowed_parts]
        for number, job in enumerate(canonical):
            word = wordbank[job['word_id']]
            try:
                if word['word'] in excluded_words:
                    raise ValueError('Actual review/development word explicitly excluded from new fitting data')
                reference_id = tuples[job['word_id'], 1 - job['voice_ordinal'], 0]
                source_observed, reference_observed = observation(job['id']), observation(reference_id)
                if not all(native_observed(source_observed[field]) for field in ('features', 'secondary_features')):
                    raise ValueError('Expected source vowel not favored and observed by both probes')
                reference_valid = all(native_observed(reference_observed[field]) for field in ('features', 'secondary_features'))
                if args.self_reference_backoff:
                    if reference_valid:
                        raise ValueError('Qualified other-voice pair belongs to v60; no duplicate generation')
                    reference_id = job['id']
                elif not reference_valid:
                    raise ValueError('Expected other-voice reference vowel not favored and observed by both probes')
                if (not args.self_reference_backoff and jobs[reference_id]['speaker_id'] == job['speaker_id']) or jobs[reference_id]['text'] != job['text']:
                    raise ValueError('Reference conditioning voice/text independence failed')
                loaded, ordinal, original_word, _ = item(job['id'])
                _, _, _, references = item(reference_id)
                templates = [declared_template(word, vocabulary) for vocabulary in runtime.vocabularies]
                start, end = full_gap(loaded, ordinal, *templates[0], source_observed['evidence'])
                if not original_word['start'] * 16000 <= start < end <= original_word['end'] * 16000 or not .04 <= (end - start) / 16000 <= .3:
                    raise ValueError('Source full vowel region outside owned40..300ms interval')
                samples = audio(plan / 'teacher', receipts[job['id']])
                donor, donor_id, donor_receipt = None, None, None
                if word['kind'] == 'vowel_quality':
                    donor_id = tuples[job['word_id'], job['voice_ordinal'], 1]
                    other = observation(donor_id)
                    if any(other[field][1] <= 0 or other[field][9] <= -1.5 for field in ('features', 'secondary_features')):
                        raise ValueError('Same-query-voice edited vowel donor not observed/favored by both probes')
                    donor_item, donor_ordinal, _, _ = item(donor_id)
                    ds, de = full_gap(donor_item, donor_ordinal, *templates[0], other['evidence'], alternative=True)
                    donor = audio(plan / 'teacher', receipts[donor_id])[ds:de]
                    if not .04 <= len(donor) / 16000 <= .3 or not .5 <= len(donor) / (end - start) <= 2:
                        raise ValueError('Alternative full vowel outside40..300ms or0.5..2 duration ratio')
                    donor_receipt = {'job_id': donor_id, 'sha256': receipts[donor_id]['sha256'], 'start_sample': ds, 'end_sample': de}
                pair = []
                for label in (0, 1):
                    chunk = samples[start:end] if label == 0 else np.empty(0, np.float32) if donor is None else donor
                    edited = splice(samples, start, end, chunk)
                    mapped = map_word(original_word, start, end, len(chunk))
                    identity = hashlib.sha256(('full-context-v60:' + job['id'] + ':' + str(label)).encode()).hexdigest()[:24]
                    path = out / 'controls' / (identity + '.wav')
                    sf.write(path, edited, 16000, subtype='PCM_16')
                    emitted, rate = sf.read(path, dtype='float32')
                    if rate != 16000 or emitted.ndim != 1:
                        raise ValueError('Emitted full-region control format changed')
                    log_probs = runtime.encode(emitted)
                    modes = {}
                    for mode in (('strict',) if args.context_evaluation else MODES):
                        values, masks = [], []
                        for probe, vocabulary in enumerate(runtime.vocabularies):
                            compiled, variant, contrast, selected = context_template(templates[probe], ipa, vocabulary, table, mode)
                            query = {'detail': {'words': [mapped], 'phone_frame_seconds': len(emitted) / 16000 / len(log_probs[probe])}, 'phone_log_probs': log_probs[probe]}
                            q, qe = measure(compiled, variant, contrast, windows(query, 0))
                            rf, re = measure(compiled, variant, contrast, windows(references[probe], 0))
                            values.append(vector(q, rf, qe, re, word['kind']))
                            masks.append(selected)
                        modes[mode] = {'features': np.concatenate(values).astype(np.float32).tolist(), 'unknown_native_indices': masks}
                    pair.append({'word': word['word'], 'word_id': word['word_id'], 'split': job['split'], 'kind': word['kind'], 'label': label,
                        'job_id': job['id'], 'reference_job_id': reference_id, 'donor_job_id': donor_id,
                        'voice': job['speaker_id'], 'reference_voice': jobs[reference_id]['speaker_id'],
                        'reference_mode': 'self_reference_backoff' if args.self_reference_backoff else 'different_conditioning_voice',
                        'source_audio_sha256': receipts[job['id']]['sha256'], 'reference_audio_sha256': receipts[reference_id]['sha256'],
                        'donor': donor_receipt, 'audio': str(path.relative_to(out)), 'audio_bytes': path.stat().st_size, 'audio_sha256': sha(path),
                        'geometry': {'start_sample': start, 'end_sample': end, 'replacement_samples': len(chunk),
                            'original_word': {k: original_word[k] for k in ('start', 'end')}, 'mapped_word': {k: mapped[k] for k in ('start', 'end')}},
                        'modes': modes, 'actual_hindi_error_label': None})
                needed.update([job['id'], reference_id])
                if donor_id is not None:
                    needed.add(donor_id)
                for record in pair:
                    records.append(record)
                    with (out / 'items.jsonl').open('a') as f:
                        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            except ValueError as error:
                logging.info('Explicit full-region whole-pair exclusion %s: %s', job['id'], error)
                exclusions.append({'job_id': job['id'], 'word': word['word'], 'reason': str(error)})
            status('building_full_region_context_pairs', canonical_inputs=number + 1, planned=len(canonical), pairs=len(records) // 2, excluded=len(exclusions))
        (out / 'source_job_ids.json').write_text(json.dumps(sorted(needed), indent=2) + '\n')
        (out / 'exclusions.json').write_text(json.dumps(exclusions, ensure_ascii=False, indent=2) + '\n')
        committed = {record['audio'] for record in records}
        inventory = [{'path': str(path.relative_to(out)), 'bytes': path.stat().st_size, 'sha256': sha(path),
            'status': 'active_training_control' if str(path.relative_to(out)) in committed else 'disposable_uncommitted_pair_diagnostic'}
            for path in sorted((out / 'controls').glob('*.wav'))]
        (out / 'waveform_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
        report = {'status': 'complete_full_region_context_pairs', 'context_evaluation': args.context_evaluation, 'pairs': len(records) // 2,
            'word_families': len({row['word'] for row in records}), 'counts_by_partition_kind': dict(Counter(row['split'] + '/' + row['kind'] for row in records)),
            'source_jobs_require_content_screen': len(needed), 'excluded_inputs': len(exclusions), 'fitted': False,
            'actual_hindi_error_accuracy': None, 'production_changed': False}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        status(report['status'], pairs=report['pairs'])
        print(json.dumps(report))
    except Exception:
        status('failed_explicitly', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Full-vowel cross-reference control builder failed explicitly')
        raise
