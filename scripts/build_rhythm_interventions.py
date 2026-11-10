"""Causal word-duration controls with measured MMS spans and phone stability."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import traceback

import numpy as np
import pyworld
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.local_features import PROSODY_NAMES
from hindi_detector.rhythm_transfer import duration_map, retime_parameters
from hindi_detector.structured_detector import context
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.build_prosody_interventions import detail
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def phone_distance(a, b):
    a, b = a.split(), b.split()
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recovery-v52', action='store_true')
    args = parser.parse_args()
    plan = ROOT / 'data/prosody_transfer_v49'
    original = ROOT / 'reports/structured_evidence_v15'
    extra = ROOT / 'reports/prosody_native_donor_features_v49'
    out = ROOT / ('reports/rhythm_interventions_v52' if args.recovery_v52 else 'reports/rhythm_interventions_v51')
    minimum = .03 if args.recovery_v52 else .08
    if out.exists():
        raise FileExistsError(out)
    accepted = set(json.loads((ROOT / 'reports/prosody_native_donor_content_v49/result.json').read_text())['accepted_ids'])
    third = {row['id']: row for row in read_rows(extra / 'items.jsonl')}
    selection = json.loads((plan / 'transfer_plan.json').read_text())
    write_group(out, {'protocol.json': {'mode': 'relative_word_duration_only',
        'partitions': {'train': 100, 'development': 24, 'reserved_evaluation': 40},
        'inputs': 'Same frozen164-word/conditioning-group partitions as pitch experiment; no rhythm model fitted or reserved rhythm scores inspected before this protocol.',
        'prior_exposure': 'The word sets previously appeared in v49 pitch-only evaluation. They remain held out from rhythm fitting, not globally untouched data.',
        'intervention': 'Redistribute Native word durations proportional to donor word lengths; preserve Native absolute pauses, total utterance length and all Native spectral/F0/aperiodicity values under monotone time warp.',
        'admissibility': 'Local duration ratio.4..2.5, explicit minimum word' + str(round(minimum * 1000)) + 'ms; finite outputs; lexical Native donor ASR pass; identical aligned words.',
        'protocol_recovery': ('v51 failed before any model fitting:58train/13dev/21reserved below preset60/15/25 quotas. Metadata audit identified natural short function words such as म/यो/र below80ms. New30ms floor retains six WORLD frames, same stretch bounds; exact last-word-at-duration boundary fixed. No v51 checkpoint or reserved rhythm scores existed.' if args.recovery_v52 else None),
        'measurement': 'MMS_FA re-aligns emitted WAV; only measured duration_relative,pause_before_relative,pause_after_relative with+/-2 word context feed model, not intended warp landmarks.',
        'phone_audit': 'Free phone edit distance relative to original Native compared to sham baseline; exact-stability subset reported separately, never called human gold.',
        'labels': 'Physical Hindi-checkpoint versus Native-donor word rhythm, not unacceptable Hindi-error gold.',
        'model': '40->8->1 MLP with training-only normalization; minimum dev BCE of300 epochs; high=max(.5,max dev native+.00001), frozen before reserved scoring.',
        'nuisance': 'Reserved sham +/-3 semitones and gain.4, same duration intent. Extra flags over WORLD sham explicitly reported.',
        'source_plan_sha256': sha(plan / 'transfer_plan.json')},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
            for p in (Path(__file__), ROOT / 'hindi_detector/rhythm_transfer.py', ROOT / 'hindi_detector/word_evidence.py')]},
        'Word-rhythm causal controls. protocol.json frozen before synthesis/score fitting. '
        'audio/*.wav Native-spectrum monotone word-duration interventions and WORLD shams, '
        'features/*.npz measured40-dimensional contextual rhythm pair and29-column measured '
        'query/reference matrices. details/*.json actual MMS word boundaries, free phones and '
        'prosody per output. rows.jsonl pins source/output SHA/bytes, intended-vs-measured '
        'duration and phone-edit-over-sham audit, explicit exclusions. runtime.json exact frozen '
        'acoustic identities, status/report actual progress. No donor pitch/spectrum imported. '
        'No human accent labels or live model changes. Consumer rhythm model admission test; '
        'review2026-11-10, raw audio/frames expire24h after consumer closes, retain compact '
        'metrics/configs/selected state. No HF upload; inherited MMS/probe/source rights.',
        ['.venv-indicf5/bin/python -m scripts.build_rhythm_interventions'])
    for name in ('audio', 'features', 'details'):
        (out / name).mkdir()
    counts = Counter()
    engine = WordEvidenceEngine(ROOT, device='mps', pitch_method='dio')
    (out / 'runtime.json').write_text(json.dumps({'alignment': engine.aligner_inventory, 'phone_probe': engine.phone_inventory,
        'vocabulary': engine.vocab, 'device': 'mps', 'pitch_method': 'dio'}, indent=2) + '\n')
    try:
        for item in selection:
            record = {k: item[k] for k in ('id', 'word', 'split', 'text', 'word_ordinal', 'conditioning_groups')}
            try:
                row = third[item['third_native_donor_job']]
                if row['id'] not in accepted or row['status'] != 'ok_rendered_phone_frames':
                    raise ValueError('Third Native donor failed content/feature qualification')
                cd, rd, hd = [detail(original, item[key]) for key in ('carrier', 'comparison_reference', 'hindi_prosody_donor')]
                nd = detail(extra, row)
                words = [word['word'] for word in cd['words']]
                if any([word['word'] for word in d['words']] != words for d in (rd, hd, nd)):
                    raise ValueError('Same-text word identities differ')
                ordinal = item['word_ordinal']
                carrier_row = item['carrier']['row']
                samples = audio(Path(carrier_row['source_pack']), carrier_row).astype(np.float64)
                f0, times = pyworld.dio(samples, 16000, f0_floor=50., f0_ceil=650., frame_period=5.)
                f0 = pyworld.stonemask(samples, f0, times, 16000)
                sp, ap = pyworld.cheaptrick(samples, f0, times, 16000), pyworld.d4c(samples, f0, times, 16000)
                reference = np.array([[w['prosody'][name] for name in PROSODY_NAMES] for w in rd['words']], np.float32)
                seconds = len(samples) / 16000
                conditions = [('sham', cd, 1., 1., 0), ('native_transfer', nd, 1., 1., 0), ('hindi_transfer', hd, 1., 1., 1)]
                if item['split'] == 'reserved_evaluation':
                    conditions += [('sham_pitch_plus3', cd, np.exp2(3 / 12), 1., 0),
                                   ('sham_pitch_minus3', cd, np.exp2(-3 / 12), 1., 0), ('sham_gain', cd, 1., .4, 0)]
                outputs = []
                for name, donor, pitch_factor, gain, label in conditions:
                    source_knots, target_knots, intended, ratios = duration_map(cd['words'], donor['words'], seconds, minimum_word_seconds=minimum)
                    pitch, spectral, noise = retime_parameters(times, f0, sp, ap, source_knots, target_knots)
                    wave = pyworld.synthesize(pitch * pitch_factor, spectral, noise, 16000, frame_period=5.)[:len(samples)] * gain
                    if len(wave) != len(samples) or not np.isfinite(wave).all() or np.max(np.abs(wave)) > 1.5:
                        raise ValueError('Invalid emitted duration-control waveform')
                    wp = out / 'audio' / (item['id'] + '_' + name + '.wav')
                    sf.write(wp, wave, 16000, subtype='FLOAT')
                    emitted, rate = sf.read(wp, dtype='float32')
                    if rate != 16000:
                        raise ValueError('Emitted control sample rate changed')
                    measured, _ = engine.analyze(emitted, item['text'])
                    if [word['word'] for word in measured['words']] != words:
                        raise ValueError('Emission re-alignment lost word identity')
                    q = np.array([[w['prosody'][column] for column in PROSODY_NAMES] for w in measured['words']], np.float32)
                    delta = context(q[:, :3])[ordinal] - context(reference[:, :3])[ordinal]
                    x = np.concatenate([delta, np.abs(delta)]).astype(np.float32)
                    if x.shape != (40,) or not np.isfinite(x).all():
                        raise ValueError('Measured contextual rhythm dimensions differ')
                    fp = out / 'features' / (item['id'] + '_' + name + '.npz')
                    np.savez_compressed(fp, paired_rhythm=x, query_prosody=q, reference_prosody=reference)
                    dp = out / 'details' / (item['id'] + '_' + name + '.json')
                    dp.write_text(json.dumps(measured, ensure_ascii=False, allow_nan=False) + '\n')
                    actual = measured['words'][ordinal]
                    outputs.append({'condition': name, 'source_style_label': label,
                        'audio': str(wp.relative_to(out)), 'audio_sha256': sha(wp), 'audio_bytes': wp.stat().st_size,
                        'features': str(fp.relative_to(out)), 'features_sha256': sha(fp), 'features_bytes': fp.stat().st_size,
                        'detail': str(dp.relative_to(out)), 'detail_sha256': sha(dp), 'detail_bytes': dp.stat().st_size,
                        'actual_word_duration_seconds': actual['end'] - actual['start'],
                        'intended_word_duration_seconds': float(intended[ordinal, 1] - intended[ordinal, 0]),
                        'focal_word_stretch': float(ratios[ordinal]), 'phone_sequence': actual['phone_probe_sequence'],
                        'phone_edit_distance_from_carrier': phone_distance(actual['phone_probe_sequence'], cd['words'][ordinal]['phone_probe_sequence']),
                        'actual_hindi_error_label': None})
                for output in outputs:
                    output['additional_phone_edits_over_sham'] = output['phone_edit_distance_from_carrier'] - outputs[0]['phone_edit_distance_from_carrier']
                record.update(status='ok_rhythm_interventions', outputs=outputs, intervention_kind='rhythm_only',
                              source_sha256={'carrier': carrier_row['sha256'], 'native_donor': row['sha256'],
                                             'hindi_donor': item['hindi_prosody_donor']['row']['sha256']})
            except ValueError:
                logging.exception('Explicit rhythm-control exclusion %s', item['id'])
                record.update(status='excluded_rhythm_intervention', error=traceback.format_exc())
            with (out / 'rows.jsonl').open('a') as f:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            counts[record['status']] += 1
            temp = out / 'status.partial'
            temp.write_text(json.dumps({'phase': 'building_word_rhythm_controls', 'counts': dict(counts),
                                       'updated_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
            temp.replace(out / 'status.json')
        report = {'status': 'complete_word_rhythm_controls', 'counts': dict(counts), 'actual_hindi_error_accuracy': None}
        for name in ('report.json', 'status.json'):
            (out / name).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
    except Exception:
        (out / 'status.json').write_text(json.dumps({'phase': 'failed_explicitly', 'error': traceback.format_exc()}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Physical word-rhythm experiment failed explicitly')
        raise
