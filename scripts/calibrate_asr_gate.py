"""Training-only actual paired-TTS ASR calibration with explicit model identities."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.content_quality import transcript_metrics

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
ZIP_SOURCE = Path('/Users/k2/kura_hat/multiuser_v1/source/kura_hat/asr.py')
ZIP_MODEL = Path('/Volumes/T9/asr_a2/exports/long1c_step95000_full')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audio(pack: Path, row: dict) -> np.ndarray:
    path = (pack / row['audio']).resolve(strict=True)
    if not path.is_relative_to(pack.resolve()) or sha(path) != row['sha256']:
        raise ValueError('ASR input path/checksum differs from paired generation')
    wave, rate = sf.read(path, dtype='float32', always_2d=True)
    if wave.shape[1] != 1 or not np.isfinite(wave).all() or not .2 <= len(wave) / rate <= 30:
        raise ValueError('Need actual finite .2–30s mono TTS for ASR calibration')
    divisor = int(np.gcd(rate, 16000))
    return resample_poly(wave[:, 0], 16000 // divisor, rate // divisor).astype('float32') if rate != 16000 else wave[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--engine', choices=['zipformer', 'whisper', 'conformer'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in (args.pack / 'items.jsonl').read_text().splitlines()]
    if not rows or any(r['split'] != 'train' or r['status'] != 'ok_unreviewed' for r in rows):
        raise ValueError('ASR calibration must use actual successful training-only generated clips')
    source = Path(__file__)
    write_group(args.out, {'inputs.json': {'pack': str(args.pack.resolve()), 'receipt_sha256': sha(args.pack / 'items.jsonl'), 'engine': args.engine, 'clips': len(rows)},
                          'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in (source, ROOT / 'hindi_detector/content_quality.py')]},
                'Measure actual Hindi/Nepali paired training clips with the user Zipformer or public Hindi-capable Whisper, before bulk filtering. hypotheses.jsonl holds per-clip decoder/WER/CER/timing; runtime.json exact runtime/model recipe; result/ has final metrics. ASR output is content evidence, never accent gold.',
                [str(args.pack.resolve()), 'Run same source with --engine zipformer in deployed hat runtime or --engine whisper in task word runtime; --out NEW'])
    if args.engine == 'zipformer':
        import sherpa_onnx
        pins = None
        for node in ast.parse(ZIP_SOURCE.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FULL_CONTEXT_SHA256' for t in node.targets):
                pins = ast.literal_eval(node.value)
        expected_names = {'encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt'}
        if not isinstance(pins, dict) or set(pins) != expected_names:
            raise ValueError('Deployed Zipformer pin schema differs')
        for name, digest in pins.items():
            if sha(ZIP_MODEL / name) != digest:
                raise ValueError('Deployed Zipformer model bytes changed')
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(ZIP_MODEL / 'encoder.int8.onnx'), decoder=str(ZIP_MODEL / 'decoder.int8.onnx'),
            joiner=str(ZIP_MODEL / 'joiner.int8.onnx'), tokens=str(ZIP_MODEL / 'tokens.txt'),
            num_threads=2, sample_rate=16000, feature_dim=80, decoding_method='modified_beam_search', max_active_paths=4)
        runtime = {'engine': args.engine, 'model_path': str(ZIP_MODEL), 'pins': pins, 'original_source_sha256': sha(ZIP_SOURCE),
                   'decoder': 'offline full-context modified_beam_search4, no VAD for already isolated TTS', 'threads': 2}

        def decode(samples):
            stream = recognizer.create_stream(); stream.accept_waveform(16000, samples); recognizer.decode_stream(stream)
            return {'ne': stream.result.text.strip()}
    elif args.engine == 'conformer':
        import sherpa_onnx
        directory = ROOT / 'models/indicconformer_gate'
        inventory = json.loads((directory / 'inventory.json').read_text())
        for member in inventory['files']:
            if sha(directory / member['path']) != member['sha256']:
                raise ValueError('Hindi/Nepali Conformer export identity changed')
        recognizers = {lang: sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=str(directory / lang / 'model.int8.onnx'),
                       tokens=str(directory / 'tokens.txt'), num_threads=2, decoding_method='greedy_search') for lang in ('hi', 'ne')}
        runtime = {'engine': args.engine, 'repo': inventory['repo'], 'revision': inventory['revision'],
                   'inventory_sha256': sha(directory / 'inventory.json'), 'device': 'cpu', 'threads': 2, 'decoder': 'nemo_ctc_greedy',
                   'languages_tested': ['hi', 'ne']}

        def decode(samples):
            output = {}
            for lang, recognizer in recognizers.items():
                stream = recognizer.create_stream(); stream.accept_waveform(16000, samples); recognizer.decode_stream(stream)
                output[lang] = stream.result.text.strip()
            return output
    else:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        torch.set_num_threads(2)
        model_path = ROOT / 'models/hindi_asr_whisper_turbo'
        inventory = json.loads((model_path / 'inventory.json').read_text())
        for member in inventory['files']:
            if sha(model_path / member['path']) != member['sha256']:
                raise ValueError('Hindi-capable ASR bytes changed')
        if not torch.backends.mps.is_available():
            raise RuntimeError('Expected available k2 MPS for Hindi ASR calibration')
        processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16, attn_implementation='sdpa').eval().to('mps')
        runtime = {'engine': args.engine, 'repo': inventory['repo'], 'revision': inventory['revision'],
                   'inventory_sha256': sha(model_path / 'inventory.json'), 'device': 'mps', 'dtype': 'float16',
                   'languages_tested': ['hi', 'ne'], 'task': 'transcribe', 'max_new_tokens': 256, 'reference_text_prompt': False}

        def decode(samples):
            batch = processor(samples, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
            values = batch.input_features.to('mps', dtype=torch.float16); mask = batch.attention_mask.to('mps')
            output = {}
            with torch.inference_mode():
                for language in ('hi', 'ne'):
                    tokens = model.generate(values, attention_mask=mask, language=language, task='transcribe', do_sample=False, max_new_tokens=256, return_timestamps=False)
                    output[language] = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            return output
    (args.out / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    results = []
    for row in rows:
        samples = audio(args.pack, row); start = time.monotonic(); hypotheses = decode(samples)
        record = {'clip_id': row['clip_id'], 'system': row['system'], 'audio_sha256': row['sha256'], 'text': row['text'],
                  'hypotheses': {lang: {'text': text, **transcript_metrics(row['text'], text)} for lang, text in hypotheses.items()},
                  'wall_seconds': time.monotonic() - start, 'seconds': len(samples) / 16000,
                  'actual_hindi_error_label': None, 'observed_utc': datetime.now(timezone.utc).isoformat()}
        with (args.out / 'hypotheses.jsonl').open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        results.append(record); LOG.info('ASR calibration %s/%s %s', len(results), len(rows), row['system'])
    groups = {}
    for system in sorted({r['system'] for r in results}):
        group = [r for r in results if r['system'] == system]
        groups[system] = {lang: {'clips': len(group), 'corpus_wer': sum(r['hypotheses'][lang]['word_errors'] for r in group) / sum(r['hypotheses'][lang]['reference_words'] for r in group),
                               'corpus_cer': sum(r['hypotheses'][lang]['character_errors'] for r in group) / sum(r['hypotheses'][lang]['reference_characters'] for r in group)} for lang in results[0]['hypotheses']}
    write_group(args.out / 'result', {'report.json': {'engine': args.engine, 'clips': len(results), 'groups': groups,
                   'wall_seconds': sum(r['wall_seconds'] for r in results), 'audio_seconds': sum(r['seconds'] for r in results), 'actual_hindi_accuracy': None}},
                'Training-only ASR content-filter calibration; preserves language/source-specific transcript errors and runtime.', ['Parent hypotheses.jsonl and runtime.json'])
    print(json.dumps(groups))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Training-only ASR calibration failed; no guessed transcripts or false success')
        raise
