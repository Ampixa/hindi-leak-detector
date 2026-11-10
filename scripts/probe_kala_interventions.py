"""Probe one-site renderer interventions; requested phones are not human gold."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from kala_tts import _infer, _postproc

from hindi_detector.artifacts import write_group
from hindi_detector.candidates import schwa_slots

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/schwa_interventions_pilot_v7'
LOG = logging.getLogger(__name__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not (OUT / 'README.md').is_file() or (OUT / 'result').exists():
        raise ValueError('Need a documented fresh intervention pilot')
    model = ROOT / 'models/kala_controls'
    inventory = json.loads((model / 'inventory.json').read_text())
    for member in inventory['files']:
        path = (model / member['path']).resolve(strict=True)
        if not path.is_relative_to(model.resolve()) or sha(path) != member['sha256']:
            raise ValueError('Pinned renderer file checksum mismatch')
    environment = json.loads((ROOT / 'runtime/kala_controls_environment.json').read_text())
    if importlib.metadata.version('kala-tts') != '0.1.4':
        raise ValueError('Native renderer helper version changed')
    for row in environment['native_sources']:
        if sha(Path(row['path'])) != row['sha256']:
            raise ValueError('Native renderer helper changed after inspection')
    config = json.loads((model / 'config.json').read_text())
    speakers = json.loads((model / 'speaker_id_map.json').read_text())
    if config['audio']['sample_rate'] != _infer.SAMPLE_RATE or speakers['kala'] != 2 or speakers['barsha'] != 1:
        raise ValueError('Inspected model speaker/rate mapping changed')
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model / 'real_nepali_v02_kala.fp32.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    names = [r.name for r in session.get_outputs()]
    if {r.name for r in session.get_inputs()} != {'input', 'input_lengths', 'scales', 'sid'} or set(names) != {'output', 'durations'}:
        raise ValueError('Actual ONNX interface differs from inspected runtime metadata')
    candidates = json.loads((ROOT / 'data/schwa_candidates_v1/candidates.json').read_text())
    chosen = [r for r in candidates if r['word'] in {'कमला', 'कर्म'}]
    if len(chosen) != 2 or any(len(r['sites']) != 1 for r in chosen):
        raise ValueError('Inspected pilot needs exactly two one-site contrasts')
    receipts = []
    for candidate in chosen:
        skeleton, counts = schwa_slots(candidate['target_phones'])
        alternate_skeleton, alternate_counts = schwa_slots(candidate['hindi_schwa_hypothesis'])
        changes = [i for i, (a, b) in enumerate(zip(counts, alternate_counts, strict=True)) if a != b]
        if skeleton != alternate_skeleton or changes != [candidate['sites'][0]['gap_index']]:
            raise ValueError('Pilot contrast changes more than the declared vowel-count site')
        for speaker in ('kala', 'barsha'):
            for condition, phones in [('lexicon_path', candidate['target_phones']), ('site_count_alternative', candidate['hindi_schwa_hypothesis'])]:
                ids = _infer._phones_to_ids(phones, config['phoneme_id_map'])
                feed = {'input': np.array([ids], dtype=np.int64), 'input_lengths': np.array([len(ids)], dtype=np.int64),
                        'scales': np.array([0., 1., 0.], dtype=np.float32), 'sid': np.array([speakers[speaker]], dtype=np.int64)}
                values = dict(zip(names, session.run(None, feed), strict=True))
                repeated = dict(zip(names, session.run(None, feed), strict=True))
                if any(not np.array_equal(values[k], repeated[k]) for k in names):
                    raise ValueError('Zero-noise repeated renderer input is not deterministic')
                wave = values['output'].reshape(-1).astype('float32')
                durations = np.asarray(values['durations'])
                if not np.isfinite(wave).all() or not np.isfinite(durations).all() or np.max(np.abs(wave)) < 1e-5:
                    raise ValueError('Renderer produced invalid/silent output')
                waveform = _postproc.postprocess(wave, _infer.SAMPLE_RATE)
                filename = candidate['candidate_id'] + '_' + speaker + '_' + condition + '.wav'
                path = OUT / filename
                if path.exists():
                    raise FileExistsError('Inspect partial pilot before rerunning')
                path.write_bytes(_infer._audio_to_wav_bytes(waveform))
                record = {'word': candidate['word'], 'candidate_id': candidate['candidate_id'], 'site': candidate['sites'][0],
                          'condition': condition, 'speaker': speaker, 'speaker_id': speakers[speaker],
                          'requested_phones': phones, 'native_encoded_input_ids': ids,
                          'scales': [0., 1., 0.], 'deterministic_repeat_verified': True,
                          'audio': filename, 'bytes': path.stat().st_size, 'sha256': sha(path),
                          'sample_rate': _infer.SAMPLE_RATE, 'audio_frames': len(waveform),
                          'seconds': len(waveform) / _infer.SAMPLE_RATE,
                          'actual_output_shapes': {k: list(v.shape) for k, v in values.items()},
                          'actual_duration_output': durations.tolist(), 'duration_output_sum': float(durations.sum()),
                          'audio_frames_per_duration_unit': len(waveform) / float(durations.sum()) if durations.sum() else None,
                          'raw_peak': float(np.max(np.abs(wave))),
                          'human_pronunciation_gold': False, 'actual_hindi_leakage_label': None}
                receipts.append(record)
                with (OUT / 'items.jsonl').open('a') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                LOG.info('Rendered deterministic one-site control %s', filename)
    sources = [{'path': str(p), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in [Path(__file__), Path(_infer.__file__), Path(_postproc.__file__), ROOT / 'hindi_detector/candidates.py']]
    report = {'status': 'phoneme_intervention_pilot_rendered_not_validated_error_labels',
              'clips': len(receipts), 'paired_word_voice_contrasts': len(receipts) // 2,
              'same_voices_and_decoder_across_conditions': True, 'noise_scales_zero_for_determinism': True,
              'model_inventory_sha256': sha(model / 'inventory.json'), 'created_utc': datetime.now(timezone.utc).isoformat(),
              'limits': ['Both sides use the same voice/model and differ only in the requested one-site phone count.',
                         'Requested phonemes do not prove what the renderer acoustically realized.',
                         'The alternate is a candidate site control, not automatically a Hindi error or native-invalid word.',
                         'Duration output is inventoried as observed, not yet treated as human-verified phone alignment.',
                         'Zero-noise intervention controls are auxiliary data; Pocket remains the preferred TTS comparison.']}
    write_group(OUT / 'result', {'report.json': report, 'receipts.json': receipts, 'source_snapshot.json': sources},
                'Inspect whether a pinned same-voice renderer supports reproducible schwa-count interventions and exposes timing.',
                ['Parent items.jsonl lists every WAV by size/SHA', 'Model HF revision45b7d3052a0ad01d5741213ff041e65db0829f5a',
                 '.venv-kala-controls/bin/python -m scripts.probe_kala_interventions'])
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('One-site control pilot failed; do not infer acoustic or accent labels')
        raise
