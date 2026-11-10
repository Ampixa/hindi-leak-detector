"""Thin, strict adapter around the inspected OpenVoice converter core."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


class VoiceControlConverter:
    def __init__(self, root: Path):
        import librosa
        import torch

        source, directory = root / '.sources/openvoice', root / 'models/openvoice_v2'
        self.inventory = json.loads((directory / 'inventory.json').read_text())
        for entry in self.inventory['files']:
            path = (source if entry['directory'] == 'source' else directory) / entry['path']
            if path.stat().st_size != entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Voice-control model/source inventory mismatch')
        sys.path.insert(0, str(source))
        from openvoice.models import SynthesizerTrn
        from openvoice.mel_processing import spectrogram_torch

        self.torch, self.librosa, self.spectrogram_torch = torch, librosa, spectrogram_torch
        torch.set_num_threads(2)
        self.config = json.loads((directory / 'config.json').read_text())
        data = self.config['data']
        if self.config['_version_'] != 'v2' or data['n_speakers'] != 0:
            raise ValueError('Unexpected converter configuration')
        self.model = SynthesizerTrn(len(self.config.get('symbols', [])), data['filter_length'] // 2 + 1,
                                   n_speakers=data['n_speakers'], **self.config['model']).eval()
        state = torch.load(directory / 'checkpoint.pth', map_location='cpu', weights_only=True)
        self.model.load_state_dict(state['model'], strict=True)

    def spectrogram(self, path: Path):
        data = self.config['data']
        if sf.info(path).channels != 1:
            raise ValueError('Voice controls require explicit mono reference/source audio')
        samples, _ = self.librosa.load(path, sr=data['sampling_rate'], mono=True)
        if not np.isfinite(samples).all() or len(samples) < data['sampling_rate'] // 2:
            raise ValueError('Voice conversion needs finite speech >=0.5s')
        tensor = self.torch.from_numpy(samples).float()[None]
        return self.spectrogram_torch(tensor, data['filter_length'], data['sampling_rate'],
                                     data['hop_length'], data['win_length'], center=False)

    def style(self, path: Path):
        with self.torch.inference_mode():
            return self.model.ref_enc(self.spectrogram(path).transpose(1, 2)).unsqueeze(-1).detach()

    def convert(self, path: Path, source_style, target_style):
        self.torch.manual_seed(0)
        spec = self.spectrogram(path)
        length = self.torch.tensor([spec.shape[-1]], dtype=self.torch.long)
        with self.torch.inference_mode():
            result = self.model.voice_conversion(spec, length, sid_src=source_style, sid_tgt=target_style, tau=.3)[0][0, 0]
        wave = result.cpu().float().numpy()
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError('Nonfinite/empty converted speech')
        return wave, self.config['data']['sampling_rate']
