"""Reviewed local-vowel controls with live alignment and retained abstentions."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import os
import traceback

import numpy as np
import pyworld
from scipy.interpolate import interp1d
import soundfile as sf

from hindi_detector.anchored_vowels import FEATURE_NAMES, measure
from hindi_detector.artifacts import write_group
from hindi_detector.aspirate_context import AspirateContextTable
from hindi_detector.context_templates import template
from hindi_detector.control_measurements import attempt
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_runtime import PhonePair
from hindi_detector.edit_geometry import map_word
from hindi_detector.pronunciation_sites import windows
from hindi_detector.task_vowel_detector import load
from hindi_detector.teacher_reference_cache import verified_paths
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.build_vowel_controls import splice
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT = Path(__file__).resolve().parents[1]


def world_versions(samples, start, end):
    """Matched WORLD reconstruction and half-duration focal vowel, pitch held."""
    signal = np.asarray(samples, np.float64)
    f0, times = pyworld.dio(signal, 16000, f0_floor=50., f0_ceil=650., frame_period=5.)
    f0 = pyworld.stonemask(signal, f0, times, 16000)
    spectrum = pyworld.cheaptrick(signal, f0, times, 16000)
    noise = pyworld.d4c(signal, f0, times, 16000)
    original = pyworld.synthesize(f0, spectrum, noise, 16000, frame_period=5.)
    if len(original) < len(samples):
        raise ValueError('WORLD reconstruction shorter than declared input')
    replacement = (end - start) // 2
    target_samples = len(samples) - (end - start) + replacement
    source_knots = np.array([0., start / 16000, end / 16000, len(samples) / 16000])
    target_knots = np.array([0., start / 16000, (start + replacement) / 16000, target_samples / 16000])
    target_times = np.minimum(np.arange(int(np.ceil(target_samples / 80)) + 1) * .005, target_samples / 16000)
    query = np.interp(target_times, target_knots, source_knots)
    query = np.clip(query, times[0], times[-1])
    nearest = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
    nearest -= np.abs(times[nearest - 1] - query) < np.abs(times[nearest] - query)
    sp = np.ascontiguousarray(interp1d(times, spectrum, axis=0)(query))
    ap = np.ascontiguousarray(interp1d(times, noise, axis=0)(query))
    compressed = pyworld.synthesize(np.ascontiguousarray(f0[nearest]), sp, ap, 16000, frame_period=5.)
    if len(compressed) < target_samples:
        raise ValueError('WORLD compression shorter than declared time map')
    return original[:len(samples)].astype(np.float32), compressed[:target_samples].astype(np.float32), replacement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--partition', action='append', choices=['train', 'development', 'test'])
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    selection = json.loads((args.plan / 'selection.json').read_text())
    jobs = {job['id']: job for job in selection['jobs']}
    words = {word['word_id']: word for word in selection['words']}
    receipts = {row['id']: row for row in read_rows(args.frames / 'items.jsonl')}
    parts = set(args.partition or ['train', 'development', 'test'])
    canonical = [job for job in jobs.values() if job['requested_vowel_change_class'] == 0 and job['split'] in parts]
    verified_paths(args.plan, canonical)
    if any(job['id'] not in receipts for job in canonical):
        raise ValueError('Requested partition has not finished source feature extraction')
    write_group(args.out, {'protocol.json': {
        'plan_sha256': sha(args.plan / 'selection.json'), 'frame_journal_prefix_sha256': sha(args.frames / 'items.jsonl'),
        'feature_names': ['primary_' + n for n in FEATURE_NAMES] + ['secondary_' + n for n in FEATURE_NAMES],
        'partitions': sorted(parts), 'source_eligibility': 'Deletion only; internal consonant neighbors; both exact local phone probes favor a present vowel, focal emission>-1.5, ownership>=0.5; two boundary estimates agree within20ms; shared region40..300ms inside source word and contains both focal spans.',
        'boundary': 'Intersection of both unedited-source anchor gaps, fully containing both focal spans; at most20ms disagreement. No selection by edited outcome.',
        'families': {'S_original': 'Untouched source', 'S0': 'Matched2ms focal reinsertion sham', 'P1': 'Full observed focal vowel removal',
            'S1': 'Identical-duration removal outside target word, with50ms separation', 'S_world': 'WORLD reconstruction with original duration/pitch',
            'P1w': 'Focal removal from matched WORLD reconstruction; vocoder balanced across labels',
            'P1u': 'Test-only removal of the source-probe union interval; boundary robustness',
            'S2': 'WORLD half-duration vowel with source pitch and spectral envelope; source region>=80ms so at least40ms retained'},
        'production_alignment': 'Every emitted waveform is re-aligned with the actual MMS path; exact mapped boundaries retained only for diagnosis.',
        'post_edit_policy': 'Every attempted family retained. Measurement failure is an abstain, counted as a positive miss at evaluation; no edited outcome filtering.',
        'source_content_gate': 'Source content receipts must pass separately before fitting; no edited-waveform ASR filtering.',
        'labels': 'Known physical presence/removal interventions only; not real Hindi-error gold.', 'classifier_scores_used': False},
        'source_frame_receipts.json': [receipts[job['id']] for job in canonical],
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in
            (Path(__file__), ROOT / 'hindi_detector/exact_phone_ctc.py', ROOT / 'hindi_detector/anchored_vowels.py', ROOT / 'hindi_detector/control_measurements.py')]},
        'Reviewed local deletion controls. controls/*.wav actual same-source original/sham/deletion/off-target/World/compression audio; '
        'items.jsonl every attempted row, label, audio SHA, source geometry, production alignment and24 local features or retained abstention. '
        'exclusions.json pre-edit/source-only exclusions; report.json independent source/lexeme counts and family outcome denominators; '
        'source_job_ids.json canonical source IDs requiring content screening; source_frame_receipts.json immutable selected-partition receipt snapshot while the full producer journal can still grow; runtime.json exact encoders/aligner. No source/voice/reference IDs '
        'are model features. Owner Ampixa/user, review2026-11-10; active consumer reviewed training/pilot-yield/fresh-test. Raw controls/frames '
        'disposable24h after these consumers close; retain compact metrics/configs/pins. Known duplicate raw teacher renders on L40/k2, no new '
        'checkpoint copy or HF upload. Inherited source/MMS/PanPhon rights apply; internal research. All after-edit failures remain visible.',
        ['python -m scripts.build_local_controls --plan PLAN --frames FRAMES --out NEW [--partition train]'])
    (args.out / 'controls').mkdir()
    def status(phase, **extra):
        p = args.out / 'status.json'
        temporary = p.with_suffix('.partial')
        temporary.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2) + '\n')
        temporary.replace(p)
    try:
        status('loading_verified_production_runtime')
        _, state = load(ROOT / 'reports/task_vowel_release_v31/model.pt')
        engine = WordEvidenceEngine(ROOT, device='mps', pitch_method='dio', retain_frame_evidence=True)
        runtime = PhonePair(ROOT, state, primary_engine=engine)
        table = AspirateContextTable(ROOT / 'data/panphon_context_v58')
        ipa = json.loads((ROOT / 'data/swar_phone_plan_v18/ipa_map.json').read_text())
        (args.out / 'runtime.json').write_text(json.dumps({'phone_inventories': [state['phone_inventory'], state['secondary_phone_inventory']],
            'aligner_inventory': engine.aligner_inventory, 'world_version': pyworld.__version__, 'raw_teacher_runtime_sha256': sha(args.plan / 'teacher/runtime.json')}, indent=2) + '\n')
        records, excluded, source_ids = [], [], []
        for number, job in enumerate(canonical):
            word = words[job['word_id']]
            # All eligibility exceptions end before a changed waveform exists.
            try:
                if word['kind'] != 'vowel_deletion':
                    raise ValueError('Separate quality head; not a deletion control')
                receipt = receipts[job['id']]
                item = load_item(args.frames, receipt)
                ordinals = [i for i, value in enumerate(item['detail']['words']) if value['word'] == word['word']]
                if len(ordinals) != 1:
                    raise ValueError('Canonical target is not uniquely aligned')
                ordinal = ordinals[0]
                original_word = item['detail']['words'][ordinal]
                with np.load(args.frames / receipt['features'], allow_pickle=False) as arrays:
                    raw = [arrays['phone_log_probs'].astype(float), arrays['secondary_phone_log_probs'].astype(float)]
                templates = [template(word, ipa, vocabulary, table) for vocabulary in runtime.vocabularies]
                samples = audio(args.plan / 'teacher', receipt)
                gaps, peaks, source_features = [], [], []
                for probe in (0, 1):
                    dt = len(samples) / 16000 / len(raw[probe])
                    measurement = {'detail': {'words': [original_word], 'phone_frame_seconds': dt}, 'phone_log_probs': raw[probe]}
                    features, evidence = measure(*templates[probe], windows(measurement, 0))
                    source_features.extend(features.tolist())
                    if features[2] >= 0 or features[5] <= -1.5 or features[6] < .5:
                        raise ValueError('Source focal vowel not locally favored/owned by both exact phone probes')
                    central = evidence['windows'][1]
                    span = central['native_focal_span']
                    gaps.append((int(central['gap_start_frame'] * dt * 16000), int(central['gap_end_frame'] * dt * 16000)))
                    peaks.append((int((central['start_frame'] + span['start_frame']) * dt * 16000), int((central['start_frame'] + span['end_frame']) * dt * 16000)))
                if max(abs(gaps[0][i] - gaps[1][i]) for i in (0, 1)) > 320:
                    raise ValueError('Unedited-source anchor boundaries disagree by more than20ms')
                start, end = max(g[0] for g in gaps), min(g[1] for g in gaps)
                if any(not start <= a < b <= end for a, b in peaks):
                    raise ValueError('Conservative shared gap does not fully contain both source focal spans')
                if not original_word['start'] * 16000 <= start < end <= original_word['end'] * 16000 or not .04 <= (end - start) / 16000 <= .3:
                    raise ValueError('Full owned source vowel outside40..300ms or source word')
            except ValueError as error:
                excluded.append({'job_id': job['id'], 'word': word['word'], 'phase': 'pre_edit_source_eligibility', 'reason': str(error)})
                status('building_local_controls', sources=number + 1, total=len(canonical), eligible=len(source_ids), rows=len(records))
                continue
            source_ids.append(job['id'])
            length = end - start
            off_end = int(original_word['start'] * 16000) - 800
            off_start = off_end - length
            families = ['S_original', 'S0', 'P1']
            if 0 < off_start < off_end < len(samples):
                families.append('S1')
            families.extend(['S_world','P1w'])
            if length >= 1280:
                families.append('S2')
            if job['split']=='test':families.append('P1u')
            world_cache = None
            for family in families:
                record = {'pair_id': job['id'], 'job_id': job['id'], 'word_id': word['word_id'], 'word': word['word'],
                    'phone_collision_group': word.get('phone_collision_group', word['word_id']), 'voice': job['speaker_id'],
                    'split': job['split'], 'family': family, 'label': int(family in {'P1','P1w','P1u'}), 'source_audio_sha256': receipt['sha256'],
                    'source_features': source_features,
                    'source_region': {'start_sample': start, 'end_sample': end}, 'actual_hindi_error_label': None}
                def evaluate():
                    nonlocal world_cache
                    a, b, replacement = start, end, length
                    if family == 'S_original':
                        emitted = samples.copy()
                    elif family == 'S0':
                        emitted = splice(samples, a, b, samples[a:b])
                    elif family == 'P1':
                        emitted, replacement = splice(samples, a, b, np.empty(0, np.float32)), 0
                    elif family == 'P1u':
                        a,b=min(g[0] for g in gaps+peaks),max(g[1] for g in gaps+peaks)
                        if not original_word['start']*16000<=a<b<=original_word['end']*16000:
                            raise ValueError('Source-only union interval extends beyond original word')
                        emitted,replacement=splice(samples,a,b,np.empty(0,np.float32)),0
                    elif family == 'S1':
                        a, b, replacement = off_start, off_end, 0
                        emitted = splice(samples, a, b, np.empty(0, np.float32))
                    else:
                        if world_cache is None:
                            world_cache = world_versions(samples, start, end)
                        if family=='P1w':
                            emitted=splice(world_cache[0],start,end,np.empty(0,np.float32));replacement=0
                        else:
                            emitted = world_cache[0] if family == 'S_world' else world_cache[1]
                            replacement = length if family == 'S_world' else world_cache[2]
                    identity = hashlib.sha256((job['id'] + ':local69:' + family).encode()).hexdigest()[:24]
                    path = args.out / 'controls' / (identity + '.wav')
                    sf.write(path, emitted, 16000, subtype='PCM_16')
                    wave, rate = sf.read(path, dtype='float32')
                    if rate != 16000 or wave.ndim != 1:
                        raise ValueError('Actual local-control waveform format differs')
                    record.update(audio=str(path.relative_to(args.out)), audio_sha256=sha(path), audio_bytes=path.stat().st_size,
                        mapped_word=map_word(original_word, a, b, replacement),
                        edit={'start_sample': a, 'end_sample': b, 'replacement_samples': replacement})
                    detail, arrays = engine.analyze(wave, job['text'])
                    aligned = detail['words'][ordinal]
                    if aligned['word'] != word['word']:
                        raise ValueError('Production-aligned target identity differs')
                    log_probs = [arrays['phone_log_probs'], runtime.encode_one(wave, 1)]
                    features, evidence = [], []
                    for probe in (0, 1):
                        query = {'detail': {'words': [aligned], 'phone_frame_seconds': len(wave) / 16000 / len(log_probs[probe])}, 'phone_log_probs': log_probs[probe]}
                        vector, local = measure(*templates[probe], windows(query, 0))
                        features.extend(vector.tolist())
                        evidence.append(local)
                    return {'features': features, 'evidence': evidence,
                            'production_word': {key: aligned[key] for key in ('start', 'end', 'alignment_score')}}
                outcome = attempt(record, evaluate)
                records.append(outcome)
                with (args.out / 'items.jsonl').open('a') as stream:
                    stream.write(json.dumps(outcome, ensure_ascii=False, allow_nan=False) + '\n')
            status('building_local_controls', sources=number + 1, total=len(canonical), eligible=len(source_ids), rows=len(records))
        (args.out / 'exclusions.json').write_text(json.dumps(excluded, ensure_ascii=False, indent=2) + '\n')
        (args.out / 'source_job_ids.json').write_text(json.dumps(source_ids, indent=2) + '\n')
        report = {'status': 'complete_reviewed_local_controls', 'canonical_requests': len(canonical), 'eligible_sources': len(source_ids),
            'eligible_lexemes': len({row['word'] for row in records}), 'pre_edit_exclusions': len(excluded),
            'family_outcomes': dict(Counter(row['family'] + '/' + row['status'] for row in records)),
            'label_outcomes': dict(Counter(str(row['label']) + '/' + row['status'] for row in records)),
            'items_sha256': sha(args.out / 'items.jsonl') if records else None,
            'post_edit_abstentions_retained': True, 'fitted': False, 'actual_hindi_error_accuracy': None}
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        status(report['status'], eligible=report['eligible_sources'], rows=len(records))
        print(json.dumps(report), flush=True)
    except Exception:
        status('failed_explicitly', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Reviewed local-control construction failed explicitly; retain attempted outcomes, do not fit')
        raise
