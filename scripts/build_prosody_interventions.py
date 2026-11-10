"""Render matched pitch-only interventions; measure actual outputs, not target curves."""
from __future__ import annotations

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
from hindi_detector.prosody_transfer import measure_waveform, pair_prosody, transfer_pitch
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def detail(directory, row):
    path = (directory / row['detail']).resolve(strict=True)
    if not path.is_relative_to(directory.resolve()) or sha(path) != row['detail_sha256']:
        raise ValueError('Frozen measured word detail changed')
    return json.loads(path.read_text())


def main():
    plan = ROOT / 'data/prosody_transfer_v49'
    original = ROOT / 'reports/structured_evidence_v15'
    extra = ROOT / 'reports/prosody_native_donor_features_v49'
    out = ROOT / 'reports/prosody_interventions_v49'
    if out.exists():
        raise FileExistsError(out)
    accepted = set(json.loads((ROOT / 'reports/prosody_native_donor_content_v49/result.json').read_text())['accepted_ids'])
    third = {row['id']: row for row in read_rows(extra / 'items.jsonl')}
    selection = json.loads((plan / 'transfer_plan.json').read_text())
    protocol = json.loads((plan / 'protocol.json').read_text())
    if protocol['partitions'] != {'train': 100, 'development': 24, 'reserved_evaluation': 40}:
        raise ValueError('Frozen experimental partitions changed')
    write_group(out, {'inputs.json': {'transfer_plan_sha256': sha(plan / 'transfer_plan.json'),
        'protocol_sha256': sha(plan / 'protocol.json'), 'extra_features_sha256': sha(extra / 'items.jsonl'),
        'content_gate_sha256': sha(ROOT / 'reports/prosody_native_donor_content_v49/result.json')},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
            for p in (Path(__file__), ROOT / 'hindi_detector/prosody_transfer.py', ROOT / 'hindi_detector/local_features.py')]},
        'Matched WORLD pitch interventions for frozen164 distinct word contexts. Same Native '
        'carrier spectral envelope/aperiodicity/voicing mask/spans in each condition; only focal '
        'F0 changes. Third-Native donors must pass isolated lexical ASR. rows.jsonl records source '
        'SHA, actual WAV/features SHA, split, eligibility and measured contour evidence. '
        'audio/*.wav actual16k float outputs; features/*.npz actual remeasured query/reference '
        'prosody and300-dimensional contextual pair. status.json/report.json explicit progress '
        'and exclusions. Carrier/donor source recordings remain in original documented packs; '
        'no spectrum/phonetic substitution. Source-style labels, not Hindi-error gold. '
        'Active consumer causal-prosody training and withheld-word evaluation; owner Ampixa/user, '
        'review2026-11-10, redundant WAVs/frames expire24h after consumer closes. No HF upload; '
        'inherited licences. Original scripts retained in source_snapshot.json.',
        ['.venv-indicf5/bin/python -m scripts.build_prosody_interventions'])
    for name in ('audio', 'features'):
        (out / name).mkdir()
    counts = Counter()

    def commit(row):
        with (out / 'rows.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        counts[row['status']] += 1
        temp = out / 'status.partial'
        temp.write_text(json.dumps({'phase': 'building_matched_pitch_controls', 'counts': dict(counts),
                                   'updated_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
        temp.replace(out / 'status.json')

    try:
        for item in selection:
            record = {'id': item['id'], 'word': item['word'], 'split': item['split'],
                      'text': item['text'], 'word_ordinal': item['word_ordinal'],
                      'conditioning_groups': item['conditioning_groups']}
            try:
                row = third[item['third_native_donor_job']]
                if row['id'] not in accepted or row['status'] != 'ok_rendered_phone_frames':
                    raise ValueError('Third Native donor content/features not qualified')
                cd = detail(original, item['carrier'])
                rd = detail(original, item['comparison_reference'])
                hd = detail(original, item['hindi_prosody_donor'])
                nd = detail(extra, row)
                words = [word['word'] for word in cd['words']]
                if any([word['word'] for word in d['words']] != words for d in (rd, hd, nd)):
                    raise ValueError('Canonical same-text aligned word identity differs')
                ordinal = item['word_ordinal']
                if words[ordinal] != item['word']:
                    raise ValueError('Frozen focal word identity differs')
                cw = cd['words'][ordinal]
                for donor in (hd, nd):
                    dw = donor['words'][ordinal]
                    if not .15 <= dw['end'] - dw['start'] <= 2.5 or dw['prosody']['voiced_fraction'] < .3:
                        raise ValueError('Donor focal word does not meet frozen voiced-span rule')
                carrier_row = item['carrier']['row']
                samples = audio(Path(carrier_row['source_pack']), carrier_row).astype(np.float64)
                f0, times = pyworld.dio(samples, 16000, f0_floor=50., f0_ceil=650., frame_period=5.)
                f0 = pyworld.stonemask(samples, f0, times, 16000)
                spectrum = pyworld.cheaptrick(samples, f0, times, 16000)
                aperiodicity = pyworld.d4c(samples, f0, times, 16000)
                nf0, nr = transfer_pitch(times, f0, nd['tracks'], cw, nd['words'][ordinal])
                hf0, hr = transfer_pitch(times, f0, hd['tracks'], cw, hd['words'][ordinal])
                if not all(np.isfinite(a).all() for a in (spectrum, aperiodicity)):
                    raise ValueError('Nonfinite Native spectral decomposition')
                reference = np.array([[word['prosody'][name] for name in PROSODY_NAMES] for word in rd['words']], np.float32)
                conditions = [('sham', f0, 1., 0), ('native_transfer', nf0, 1., 0), ('hindi_transfer', hf0, 1., 1)]
                if item['split'] == 'reserved_evaluation':
                    conditions += [('sham_pitch_plus3', f0 * np.exp2(3 / 12), 1., 0),
                                   ('sham_pitch_minus3', f0 * np.exp2(-3 / 12), 1., 0), ('sham_gain', f0, .4, 0)]
                outputs = []
                for name, pitch, gain, label in conditions:
                    wave = pyworld.synthesize(pitch, spectrum, aperiodicity, 16000, frame_period=5.)[:len(samples)] * gain
                    if len(wave) != len(samples):
                        raise ValueError('WORLD output length differs from fixed carrier')
                    wp = out / 'audio' / (item['id'] + '_' + name + '.wav')
                    sf.write(wp, wave, 16000, subtype='FLOAT')
                    emitted, rate = sf.read(wp, dtype='float64')
                    if rate != 16000 or not np.isfinite(emitted).all() or np.max(np.abs(emitted)) > 1.5:
                        raise ValueError('Invalid emitted waveform/rate/range')
                    measured, measured_f0, measured_times = measure_waveform(emitted, cd['words'], pyworld)
                    x = pair_prosody(measured, reference, ordinal)
                    fp = out / 'features' / (item['id'] + '_' + name + '.npz')
                    np.savez_compressed(fp, paired_prosody=x, query_prosody=measured, reference_prosody=reference,
                                        measured_f0=measured_f0, measured_times=measured_times)
                    outputs.append({'condition': name, 'source_style_label': label, 'audio': str(wp.relative_to(out)),
                        'audio_sha256': sha(wp), 'audio_bytes': wp.stat().st_size, 'features': str(fp.relative_to(out)),
                        'features_sha256': sha(fp), 'features_bytes': fp.stat().st_size,
                        'pitch_available': bool(measured[ordinal, 4]),
                        'measured_pitch_change_st': float(measured[ordinal, 9]),
                        'actual_hindi_error_label': None})
                if any(not output['pitch_available'] for output in outputs):
                    raise ValueError('Emitted focal word lacks measured pitch')
                record.update(status='ok_pitch_interventions', outputs=outputs, transfer_receipts={'native': nr, 'hindi': hr},
                    source_sha256={'carrier': carrier_row['sha256'], 'native_donor': row['sha256'],
                                   'hindi_donor': item['hindi_prosody_donor']['row']['sha256']},
                    output_pitch_change_st_difference=float(outputs[2]['measured_pitch_change_st'] - outputs[1]['measured_pitch_change_st']))
            except ValueError:
                logging.exception('Explicit pitch-control eligibility exclusion %s', item['id'])
                record.update(status='excluded_pitch_intervention', error=traceback.format_exc())
            commit(record)
        report = {'status': 'complete_pitch_interventions', 'counts': dict(counts),
                  'actual_hindi_error_accuracy': None, 'reserved_scores_opened': False}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'status.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
    except Exception:
        (out / 'status.json').write_text(json.dumps({'phase': 'failed_explicitly', 'error': traceback.format_exc()}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Matched pitch intervention generation failed explicitly')
        raise
