"""Render the frozen v80 sentence plan with HF Aarjan/swar (podvoices_ema.pt) through Swar's own text frontend.

Model text = Swar app `to_model_text(text, 'chars')` on both prompt transcript and sentence, as its cloner does.
Sampler settings equal the v80 IndicF5 run (nfe32/cfg2/sway-1/speed1, no silence removal) so systems stay comparable.
Receipts keep the original plan text; the frontend rewrite is recorded per row.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import soundfile as sf
import torch

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import QualityError, sha, validate_wave

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'data/cross_system_sentences_v80'
OUT = ROOT / 'data/cross_system_swar_v84'
SYSTEM = 'swar_podvoices_ema'
SWAR = {'repo': 'Aarjan/swar', 'revision': '8ca6f1bbc9da567d92158822e24673c2084e7e86', 'root': ROOT / 'models/swar_8ca6f1b',
        'sha256': 'b932ea211288795a4a07749b84b51622b2c181cd29343f56f2a5c85598875b53'}
INDICF5 = ROOT / 'models/indicf5'


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class SwarEngine:
    def __init__(self, settings):
        from safetensors.torch import load_file
        from vocos import Vocos
        root = SWAR['root']
        if sha(root / 'podvoices_ema.pt') != SWAR['sha256']:
            raise ValueError('Swar checkpoint differs from pinned HF revision')
        if sha(root / 'vocab.txt') != sha(INDICF5 / 'checkpoints/vocab.txt'):
            raise ValueError('Swar vocab differs from IndicF5 vocab')
        if not torch.backends.mps.is_available():
            raise ValueError('Explicit MPS unavailable')
        sys.path.insert(0, str(root / 'app'))
        from pipeline.clone import to_model_text
        sys.path.insert(0, str(INDICF5))
        from f5_tts.model import CFM, DiT
        from f5_tts.model.utils import get_tokenizer, seed_everything
        from f5_tts.infer.utils_infer import infer_batch_process
        vocab, size = get_tokenizer(str(root / 'vocab.txt'), 'custom')
        model = CFM(transformer=DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4, text_num_embeds=size, mel_dim=100),
            mel_spec_kwargs={'n_fft': 1024, 'hop_length': 256, 'win_length': 1024, 'n_mel_channels': 100, 'target_sample_rate': 24000, 'mel_spec_type': 'vocos'},
            odeint_kwargs={'method': 'euler'}, vocab_char_map=vocab)
        state = torch.load(root / 'podvoices_ema.pt', map_location='cpu', weights_only=False)['ema_model_state_dict']
        model.load_state_dict({k.removeprefix('ema_model.'): v for k, v in state.items() if k.startswith('ema_model.')}, strict=True)
        # Vocoder: IndicF5-bundled weights, verified bit-identical to charactr/vocos-mel-24khz@0feb3fd that Swar's F5TTS API loads.
        vocoder = Vocos.from_hparams(str(INDICF5 / 'vocos/config.yaml'))
        prefix = 'vocoder._orig_mod.'
        vocoder.load_state_dict({k.removeprefix(prefix): v for k, v in load_file(str(INDICF5 / 'model.safetensors'), device='cpu').items() if k.startswith(prefix)}, strict=True)
        self.model = model.float().eval().to('mps'); self.vocoder = vocoder.eval().to('mps')
        self.frontend, self.seed, self.infer, self.settings = to_model_text, seed_everything, infer_batch_process, settings
        self.runtime = {'system': SYSTEM, 'repo': SWAR['repo'], 'revision': SWAR['revision'], 'weights_sha256': SWAR['sha256'],
            'frontend': 'app/pipeline/clone.py to_model_text(chars), SCHWA_HOLD/NEPAL_ACCENT defaults', 'frontend_sha256': sha(root / 'app/pipeline/clone.py'),
            'settings': settings, 'remove_silence': False, 'torch': torch.__version__, 'strict_state_loading': True, 'fp32': True}

    def generate(self, job):
        wave, rate = sf.read(job['reference_audio'], dtype='float32', always_2d=True)
        if wave.shape[1] != 1 or not np.isfinite(wave).all():
            raise ValueError('Invalid shared prompt')
        reference_text, text = self.frontend(job['reference_text'], 'chars'), self.frontend(job['text'], 'chars')
        self.seed(job['seed']); s = self.settings
        audio, rate, _ = self.infer((torch.from_numpy(wave.T), rate), reference_text, [text], self.model, self.vocoder,
            mel_spec_type='vocos', target_rms=.1, cross_fade_duration=.15, nfe_step=s['nfe_step'], cfg_strength=s['cfg_strength'],
            sway_sampling_coef=s['sway_sampling_coef'], speed=s['speed'], fix_duration=None, device='mps')
        return audio, rate, {'model_text': text, 'model_reference_text': reference_text}


def main():
    selection = json.loads((PLAN / 'selection.json').read_text())
    settings = json.loads((PLAN / 'model_pair.json').read_text())['systems']['indicf5_stock']['inference']
    jobs = [{**j, 'reference_audio': str((PLAN / j['reference_audio']).resolve(strict=True))} for j in selection['jobs']]
    for job in jobs:
        if sha(Path(job['reference_audio'])) != job['reference_sha256']:
            raise ValueError('Shared prompt changed: ' + job['id'])
    if OUT.exists():
        raise FileExistsError(OUT)
    source = Path(__file__)
    write_group(OUT, {'lock.json': {'selection_sha256': sha(PLAN / 'selection.json'), 'system': SYSTEM, 'driver_sha256': sha(source),
            'swar': {k: str(v) for k, v in SWAR.items()}, 'settings': settings}},
        f'Swar ({SWAR["repo"]}@{SWAR["revision"][:7]} podvoices_ema.pt) renderings of the frozen v80 plan: 26 texts x 4 shared prompts. '
        'items.jsonl receipts (plan text, Swar frontend model text, WAV SHA/bytes/duration or explicit quality rejection); audio/*.wav 24k PCM16; runtime.json; run.json. '
        'Weights CC BY-SA 4.0; audio NOT on HF. Created by Claude (claude-opus-5-5) Claude Code session. Status active; consumer reports/cross_system_swar_v84; '
        'review 2026-11-17; delete audio within 24h after that consumer closes, keep receipts. Reproduce by rerunning the command.',
        ['.venv-indicf5/bin/python -m scripts.run_swar_cross_system'])
    (OUT / 'audio').mkdir()
    counts = Counter()

    def status(phase, **extra):
        save(OUT / 'run.json', {'status': phase, 'pid': os.getpid(), 'system': SYSTEM, 'updated_utc': datetime.now(timezone.utc).isoformat(),
            'planned_clips': len(jobs), 'processed_clips': sum(counts.values()), 'counts': dict(counts), **extra})
    try:
        status('loading_explicit_verified_model')
        engine = SwarEngine(settings)
        save(OUT / 'runtime.json', engine.runtime)
        for job in jobs:
            if shutil.disk_usage(OUT).free < 20 * 1024**3:
                raise RuntimeError('Preserve 20 GiB free')
            status('generating', current_job=job['id']); start = time.monotonic()
            row = {k: job[k] for k in ('id', 'text', 'text_id', 'text_kind', 'target_words', 'speaker_id', 'split', 'reference_sha256', 'source_reference_sha256')}
            row.update(system=SYSTEM, clip_id=hashlib.sha256((SYSTEM + ':' + job['id']).encode()).hexdigest()[:24], automatic_leakage_label=None)
            try:
                wave, rate, detail = engine.generate(job); validate_wave(wave, rate)
                path = OUT / 'audio' / (job['id'] + '.wav')
                sf.write(path, wave, rate, subtype='PCM_16')
                row.update(status='ok_unreviewed', audio=str(path.relative_to(OUT)), sha256=sha(path), bytes=path.stat().st_size,
                    seconds=len(wave) / rate, clipping_fraction_before_pcm=float(np.mean(np.abs(wave) >= 1)), generation_detail=detail)
            except QualityError:
                LOG.exception('Explicit waveform-quality rejection %s', job['id']); row.update(status='failed_waveform_quality', error=traceback.format_exc())
            row['generation_wall_seconds'] = time.monotonic() - start
            with (OUT / 'items.jsonl').open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); handle.flush(); os.fsync(handle.fileno())
            counts[row['status']] += 1
            LOG.info('%s %s/%s', SYSTEM, sum(counts.values()), len(jobs))
        status('complete_corrected_audio')
    except Exception:
        status('failed_explicitly', error=traceback.format_exc()); raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Swar synthesis stopped; no silent fallback')
        raise
