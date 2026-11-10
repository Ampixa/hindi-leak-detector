"""Training-only diagnostic: physical retained interval versus inferred anchors."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.anchored_detector import load, score
from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_runtime import PhonePair
from hindi_detector.phone_lattice import VOWELS
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'reports/short_vowel_localization_v75'
    if out.exists():
        raise FileExistsError(out)
    controls = ROOT / 'reports/native_main_controls_train_v71'
    rows = read_rows(controls / 'items.jsonl')
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('This diagnostic must never read a held-out control partition')
    accepted = set(json.loads((ROOT / 'reports/native_main_content_train_v71/result.json').read_text())['accepted_ids'])
    rows = [r for r in rows if r['job_id'] in accepted and r['status'] == 'measured']
    network, state = load(ROOT / 'reports/anchored_training_v71/model.pt')
    short = [r for r in rows if r['family'] == 'S2']
    for r in short:
        r['audit_score'] = float(score(network, state, np.array(r['features'])[None])[0])
    # Purposefully hard training cases plus a fixed random sample: diagnostic,
    # not an unbiased estimate or another operating-point calibration.
    selected, groups = [], set()
    for r in sorted(short, key=lambda r: (-r['audit_score'], r['pair_id'])):
        if r['phone_collision_group'] not in groups:
            selected.append(r); groups.add(r['phone_collision_group'])
        if len(selected) == 12:
            break
    rng = np.random.default_rng(750)
    for at in rng.permutation(len(short)):
        r = short[at]
        if r['phone_collision_group'] not in groups:
            selected.append(r); groups.add(r['phone_collision_group'])
        if len(selected) == 24:
            break
    write_group(out, {'protocol.json': {'partition': 'train only', 'model_sha256': sha(ROOT / 'reports/anchored_training_v71/model.pt'),
        'control_items_sha256': sha(controls / 'items.jsonl'), 'selected_sources': [r['pair_id'] for r in selected],
        'selection': '12 highest-scoring S2 cases from distinct collision groups, then12 additional distinct groups with seed750. Hard-case diagnostic, not generalization accuracy.',
        'comparison': 'Matched WORLD original and shortened-but-retained vowel. Oracle interval is the actual documented edit geometry, for diagnosis only; never an inference feature.',
        'frame_geometry': 'Audio-duration/frame-count bins, as in the frozen feature runtime; bins intersecting the actual retained interval. Not exact receptive-field localization.',
        'interpretation_thresholds': {'encoder_recovers_vowel_in_oracle': .6, 'low_inferred_gap_peak': .1, 'low_oracle_peak': .2},
        'go_no_go': 'If low inferred-gap peak with oracle vowel peak>=.6, inspect localization. If shortened oracle peak<.2 while matched WORLD original>=.6, inspect encoder short-vowel sensitivity. These are diagnostic hypotheses, not human phonetic labels.'}},
        'Train-only short-vowel localization versus phone-resolution audit after rejected frozen candidate. items.jsonl stores source/family identities, retained-interval geometry, phonetic peaks, actual anchor geometry and explicit diagnostics; report.json summarizes both probes. '
        'No new audio/encoder/model copies; no fit or threshold change. Owner Ampixa/user; review2026-11-10. Active consumer decide one next measurement experiment. NOT on HF, inherited research rights. Retain compact conclusions, raw diagnostic rows expire24h after consumer closes.',
        ['.venv-indicf5/bin/python -m scripts.audit_short_vowel_localization'])
    pair = PhonePair(ROOT, state)
    measured = []
    for chosen in selected:
        for family in ('S_world', 'S2'):
            matches = [r for r in rows if r['pair_id'] == chosen['pair_id'] and r['family'] == family]
            if len(matches) != 1:
                raise ValueError('Matched original/shortened training pair missing')
            r = matches[0]
            wave = audio(controls, {'audio': r['audio'], 'sha256': r['audio_sha256']})
            start = r['edit']['start_sample'] / 16000
            end = start + r['edit']['replacement_samples'] / 16000
            if not 0 <= start < end <= len(wave) / 16000:
                raise ValueError('Retained vowel interval invalid')
            evidence = []
            for probe, lp in enumerate(pair.encode(wave)):
                dt = len(wave) / 16000 / len(lp)
                a, b = int(np.floor(start / dt)), int(np.ceil(end / dt))
                vowel_ids = sorted(pair.vocabularies[probe][v] for v in VOWELS if v in pair.vocabularies[probe])
                peak = float(np.exp(lp[a:b, vowel_ids]).sum(1).max())
                windows = r['evidence'][probe]['windows']
                local_peak = float(np.mean([w['gap_vowel_peak'] for w in windows]))
                mapped = []
                for w in windows:
                    span = w['native_focal_span']
                    fa = None if span is None else (w['start_frame'] + span['start_frame']) * dt
                    fb = None if span is None else (w['start_frame'] + span['end_frame']) * dt
                    mapped.append({'padding': w['padding_seconds'], 'gap_start': w['gap_start_frame'] * dt,
                        'gap_end': w['gap_end_frame'] * dt, 'native_focal_start': fa, 'native_focal_end': fb,
                        'native_focal_inside_actual_retained_interval': False if span is None else max(0., min(end, fb) - max(start, fa)) > 0})
                evidence.append({'probe': probe, 'oracle_interval_vowel_peak': peak, 'inferred_gap_vowel_peak': local_peak,
                                 'localization_recovery_signal': peak >= .6 and local_peak < .1, 'windows': mapped})
            result = {'pair_id': r['pair_id'], 'word': r['word'], 'voice': r['voice'], 'family': family,
                'selection_group': 'highest12' if len(measured) < 24 else 'random12', 'audio_sha256': r['audio_sha256'],
                'source_audio_sha256': r['source_audio_sha256'],
                'retained_interval': [start, end], 'retained_seconds': end - start,
                'clip_fraction': float(np.mean(np.abs(wave) >= .999)), 'evidence': evidence}
            measured.append(result)
            with (out / 'items.jsonl').open('a') as stream:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
    summary = []
    for probe in (0, 1):
        shortened = [r for r in measured if r['family'] == 'S2']
        original = {r['pair_id']: r for r in measured if r['family'] == 'S_world'}
        lost = sum(r['evidence'][probe]['oracle_interval_vowel_peak'] < .2 and
                   original[r['pair_id']]['evidence'][probe]['oracle_interval_vowel_peak'] >= .6 for r in shortened)
        summary.append({'probe': probe, 'sources': len(shortened),
                        'localization_recovery_signals': sum(r['evidence'][probe]['localization_recovery_signal'] for r in shortened),
                        'shortening_loses_encoder_vowel_signal': lost,
                        'shortened_oracle_peak_median': float(np.median([r['evidence'][probe]['oracle_interval_vowel_peak'] for r in shortened]))})
    report = {'status': 'complete_training_only_short_vowel_audit', 'sources': len(selected), 'rows': len(measured),
              'probes': summary, 'maximum_clip_fraction': max(r['clip_fraction'] for r in measured),
              'actual_hindi_error_accuracy': None, 'model_changed': False, 'items_sha256': sha(out / 'items.jsonl')}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Training-only short-vowel audit failed explicitly; no model change')
        raise
