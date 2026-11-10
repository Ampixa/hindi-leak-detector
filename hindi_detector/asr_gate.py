"""Pinned CPU ASR content checks, distinct from accent/phone supervision."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from hindi_detector.content_quality import normalize, transcript_metrics

RECIPE = {'version': 'content_gate_v11', 'conformer_cer': .15, 'conformer_wer': .5,
          'tight_cer': .08, 'agreed_roman_cer': .18, 'roman_wer': .7,
          'reject_cer': .35, 'reject_roman_cer': .3, 'reject_wer': .8}


def content_decision(evidence: dict) -> str:
    c, z = evidence['conformer_ne'], evidence['zipformer_ne']
    if c['cer'] <= RECIPE['tight_cer'] or c['cer'] <= RECIPE['conformer_cer'] and c['wer'] <= RECIPE['conformer_wer']:
        return 'pass'
    if c['roman_cer'] <= RECIPE['agreed_roman_cer'] and z['roman_cer'] <= RECIPE['agreed_roman_cer'] and min(c['wer'], z['wer']) <= RECIPE['roman_wer']:
        return 'pass'
    if all(e['cer'] > RECIPE['reject_cer'] and e['roman_cer'] > RECIPE['reject_roman_cer'] and e['wer'] > RECIPE['reject_wer'] for e in (c, z)):
        return 'reject'
    return 'quarantine'


class CpuContentGate:
    def __init__(self, root: Path, *, threads: int = 2):
        import sherpa_onnx
        import uroman

        self.romanizer = uroman.Uroman()
        source = Path('/Users/k2/kura_hat/multiuser_v1/source/kura_hat/asr.py')
        model = Path('/Volumes/T9/asr_a2/exports/long1c_step95000_full')
        pins = None
        for node in ast.parse(source.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FULL_CONTEXT_SHA256' for t in node.targets):
                pins = ast.literal_eval(node.value)
        if not isinstance(pins, dict) or set(pins) != {'encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt'}:
            raise ValueError('Actual deployed Zipformer pin schema changed')
        for name, digest in pins.items():
            if hashlib.sha256((model / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Actual deployed Zipformer bytes changed')
        self.zip = sherpa_onnx.OfflineRecognizer.from_transducer(encoder=str(model / 'encoder.int8.onnx'),
            decoder=str(model / 'decoder.int8.onnx'), joiner=str(model / 'joiner.int8.onnx'), tokens=str(model / 'tokens.txt'),
            num_threads=threads, sample_rate=16000, feature_dim=80, decoding_method='modified_beam_search', max_active_paths=4)
        directory = root / 'models/indicconformer_gate'
        inventory = json.loads((directory / 'inventory.json').read_text())
        for member in inventory['files']:
            if hashlib.sha256((directory / member['path']).read_bytes()).hexdigest() != member['sha256']:
                raise ValueError('Public Conformer export identity changed')
        self.conformer = {lang: sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=str(directory / lang / 'model.int8.onnx'),
            tokens=str(directory / 'tokens.txt'), num_threads=threads, decoding_method='greedy_search') for lang in ('hi', 'ne')}
        self.runtime = {'zipformer_path': str(model), 'zipformer_pins': pins,
                        'zipformer_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                        'conformer_repo': inventory['repo'], 'conformer_revision': inventory['revision'],
                        'conformer_inventory_sha256': hashlib.sha256((directory / 'inventory.json').read_bytes()).hexdigest(),
                        'threads_per_recognizer': threads, 'recipe': RECIPE,
                        'language_policy': 'All planned text is Nepali, even for Hindi-checkpoint speech. Nepali ASR decoders check intended lexical content; Hindi decoder is retained for Hindi-source audit.'}

    @staticmethod
    def decode(recognizer, samples) -> str:
        stream = recognizer.create_stream(); stream.accept_waveform(16000, samples); recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def score(self, samples, text: str, system: str) -> dict:
        hypotheses = {'zipformer_ne': self.decode(self.zip, samples), 'conformer_ne': self.decode(self.conformer['ne'], samples)}
        if system in {'pocket_hindi_parent', 'indicf5_stock'}:
            hypotheses['conformer_hi_audit'] = self.decode(self.conformer['hi'], samples)
        rroman = normalize(self.romanizer.romanize_string(text, lcode='nep'))
        measured = {}
        for name, hypothesis in hypotheses.items():
            hroman = normalize(self.romanizer.romanize_string(hypothesis, lcode='nep'))
            metrics = transcript_metrics(text, hypothesis)
            metrics.update(text=hypothesis, roman_cer=transcript_metrics(rroman, hroman)['cer'])
            measured[name] = metrics
        return {'hypotheses': measured, 'decision': content_decision(measured), 'actual_hindi_error_label': None}
