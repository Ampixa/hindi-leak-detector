"""Verified auxiliary renderer; phoneme requests never become human error gold."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


class KalaControlRenderer:
    def __init__(self, root: Path):
        import onnxruntime as ort
        from kala_tts import _infer, _postproc

        self.infer, self.postprocess = _infer, _postproc.postprocess
        model = root / 'models/kala_controls'
        inventory = json.loads((model / 'inventory.json').read_text())
        for row in inventory['files']:
            path = (model / row['path']).resolve(strict=True)
            if not path.is_relative_to(model.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Pinned control renderer checksum mismatch')
        environment = json.loads((root / 'runtime/kala_controls_environment.json').read_text())
        if importlib.metadata.version('kala-tts') != '0.1.4':
            raise ValueError('Native control helper version changed')
        for row in environment['native_sources']:
            if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Native helper source changed')
        self.config = json.loads((model / 'config.json').read_text())
        self.speakers = json.loads((model / 'speaker_id_map.json').read_text())
        self.rate = self.config['audio']['sample_rate']
        if self.rate != _infer.SAMPLE_RATE:
            raise ValueError('Native helper/config sample rate mismatch')
        options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model / 'real_nepali_v02_kala.fp32.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        self.output_names = [r.name for r in self.session.get_outputs()]
        if {r.name for r in self.session.get_inputs()} != {'input', 'input_lengths', 'scales', 'sid'} or set(self.output_names) != {'output', 'durations'}:
            raise ValueError('Verified native ONNX interface changed')

    def render(self, phones: list[str], speaker: str, length_scale: float) -> tuple[bytes, dict]:
        if speaker == 'algenib' or speaker not in self.speakers:
            raise ValueError('Only the five public human-source renderer voices are used')
        if not np.isfinite(length_scale) or not .5 <= length_scale <= 2:
            raise ValueError('Explicit bounded positive length scale required')
        if any(p in {'.', '|', '_', '^', '$', '#'} for p in phones):
            raise ValueError('Control requests must be explicit lexical phones without hidden separators/specials')
        ids = self.infer._phones_to_ids(phones, self.config['phoneme_id_map'])
        if len(ids) != len(phones) + 2:
            raise ValueError('Observed one-ID-per-phone encoding changed')
        feed = {'input': np.array([ids], dtype=np.int64), 'input_lengths': np.array([len(ids)], dtype=np.int64),
                'scales': np.array([0., length_scale, 0.], dtype=np.float32), 'sid': np.array([self.speakers[speaker]], dtype=np.int64)}
        values = dict(zip(self.output_names, self.session.run(None, feed), strict=True))
        wave = np.asarray(values['output']).reshape(-1).astype('float32')
        durations = np.asarray(values['durations'])
        if durations.shape != (1, len(ids)) or not np.isfinite(durations).all() or np.any(durations < 0):
            raise ValueError('Invalid observed renderer duration output')
        if not np.isfinite(wave).all() or not .2 <= len(wave) / self.rate <= 15 or np.max(np.abs(wave)) < 1e-5:
            raise ValueError('Invalid or too-short/long control waveform')
        steps = np.ceil(durations[0]).astype(int)
        if len(wave) != int(steps.sum()) * 256:
            raise ValueError('Observed ceiling-duration256-sample correspondence failed; do not invent timing')
        processed = self.postprocess(wave, self.rate)
        return self.infer._audio_to_wav_bytes(processed), {
            'native_encoded_input_ids': ids, 'scales': [0., length_scale, 0.],
            'sample_rate': self.rate, 'audio_frames': len(processed), 'seconds': len(processed) / self.rate,
            'raw_peak': float(np.max(np.abs(wave))), 'duration_output': durations.tolist(),
            'ceil_duration_steps': steps.tolist(), 'observed_samples_per_step': 256,
            'timing_meaning': 'Renderer duration correspondence only; not human-verified acoustic phone boundaries.'}
