"""Pinned research-only Allosaurus encoder; never a native-pronunciation oracle."""
from __future__ import annotations

from argparse import Namespace
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from hindi_detector.corrected_synthesis import sha
from hindi_detector.ctc import validate_log_probs


class IndependentPhone:
    def __init__(self, root: Path):
        directory = root / 'runtime/independent_phone_v64'
        inventory = json.loads((directory / 'inventory.json').read_text())
        protected = [row for row in inventory if row['path'].startswith(('source/', 'models/'))]
        for row in protected:
            path = (directory / row['path']).resolve(strict=True)
            if not path.is_relative_to(directory.resolve()) or sha(path) != row['sha256']:
                raise ValueError('Pinned independent encoder source/model bytes changed: ' + row['path'])
        sources = [p for p in (directory / 'source').iterdir() if (p / 'allosaurus').is_dir()]
        if len(sources) != 1:
            raise ValueError('Independent encoder source root is ambiguous')
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(directory / 'deps'), str(sources[0])]
        package = importlib.import_module('allosaurus')
        if not Path(package.__file__).resolve().is_relative_to(sources[0].resolve()):
            raise ValueError('Unexpected globally installed Allosaurus package')
        pm_module = importlib.import_module('allosaurus.pm.factory')
        am_module = importlib.import_module('allosaurus.am.factory')
        self.Audio = importlib.import_module('allosaurus.audio').Audio
        read_unit = importlib.import_module('allosaurus.lm.unit').read_unit
        model = directory / 'models/uni2005'
        config = Namespace(model='uni2005', device_id=-1, lang='ipa', approximate=False, prior=None)
        self.pm = pm_module.read_pm(model, config)
        self.am = am_module.read_am(model, config).eval()
        units = read_unit(model / 'phone.txt').unit_to_id
        self.original_vocabulary = units
        if sorted(units.values()) != list(range(self.am.phone_size)) or units['<blk>'] != 0:
            raise ValueError('Observed independent phone inventory/model output mismatch')
        # Project measurement code reserves IDs1..3. Add exactly zero-probability
        # columns, preserving every original phone/blank probability and ID map.
        self.vocabulary = {'<blk>': 0, '<audit_reserved1>': 1, '<audit_reserved2>': 2, '<audit_reserved3>': 3,
                           **{token: index + 3 for token, index in units.items() if index}}
        self.frame_seconds = self.pm.config.window_shift * self.pm.config.feature_window
        self.metadata = {'encoder': 'Allosaurus uni2005', 'source_revision': 'd9f1adaf47d3b3765b41f4177da62a051516d636',
                         'model_sha256': sha(model / 'model.pt'), 'vocabulary': self.vocabulary,
                         'frame_seconds': self.frame_seconds, 'dither_seed': 6400,
                         'language_inventory_mask': None, 'native_phone_accuracy': None,
                         'licence': 'Source GPL-3.0; checkpoint-specific licence not supplied in release; no redistribution.'}

    def encode(self, samples, rate=16000):
        x = np.asarray(samples)
        if x.ndim != 1 or len(x) < rate * .1 or not np.isfinite(x).all() or rate <= 0:
            raise ValueError('Independent phone encoder requires finite mono audio of at least100ms')
        if np.max(np.abs(x)) > 1.001:
            raise ValueError('Expected normalized floating-point audio, not int16 magnitudes')
        np.random.seed(6400)
        signal = np.round(np.clip(x, -1, 1) * 32767).astype(np.int16)
        features = self.pm.compute(self.Audio(signal, rate))
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError('Independent encoder feature extraction failed')
        with torch.inference_mode():
            logits = self.am(torch.from_numpy(features[None]), torch.tensor([len(features)]))[0]
            values = logits.log_softmax(-1).cpu().numpy().astype(np.float64)
        padded = np.full((len(values), values.shape[1] + 3), -np.inf, np.float64)
        padded[:, 0] = values[:, 0]
        padded[:, 4:] = values[:, 1:]
        return validate_log_probs(padded, 0)

    def decode(self, values):
        inverse = {index: token for token, index in self.vocabulary.items()}
        ids = values.argmax(1)
        return [{'phone': inverse[int(index)], 'start': frame * self.frame_seconds,
                 'model_window_seconds': .045}
                for frame, index in enumerate(ids)
                if index > 3 and (frame == 0 or index != ids[frame - 1])]
