"""Lexical-content check in the isolated CPU runtime, not an accent classifier."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.asr_gate import CpuContentGate
from hindi_detector.content_quality import normalize, transcript_metrics
from hindi_detector.corrected_synthesis import sha
from scripts.calibrate_asr_gate import audio

ROOT = Path(__file__).resolve().parents[1]


def check(gate, samples, text):
    result = gate.score(samples, text, 'unknown_nepali_tts_query')
    hindi = gate.decode(gate.conformer['hi'], samples)
    reference_roman = normalize(gate.romanizer.romanize_string(text, lcode='nep'))
    hindi_roman = normalize(gate.romanizer.romanize_string(hindi, lcode='nep'))
    result['hypotheses']['conformer_hi_audit'] = {
        **transcript_metrics(text, hindi), 'text': hindi,
        'roman_cer': transcript_metrics(reference_roman, hindi_roman)['cer']}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--stdout-only', action='store_true', help='Return the full JSON receipt to a parent CLI without retaining child files')
    args = parser.parse_args()
    if bool(args.out) == args.stdout_only or (args.out is not None and args.out.exists()) or not args.text.strip():
        raise ValueError('Choose --out NEW or --stdout-only, with nonempty intended text')
    path = args.query.resolve(strict=True)
    digest = sha(path)
    samples = audio(path.parent, {'audio': path.name, 'sha256': digest})
    gate = CpuContentGate(ROOT, threads=2)
    # Hindi decoding is an additional transcription audit. It does not supply
    # source identity or a Hindi-error label, or change the frozen content rule.
    result = check(gate, samples, args.text)
    payload = {'query': str(path), 'sha256': digest, 'text': args.text,
        'runtime': gate.runtime, **result, 'policy': 'Nepali decoders check intended Nepali lexical content; Hindi decoder is an additional audit. Reject is only both-Nepali-decoders gross mismatch. Quarantine remains explicit. ASR never supplies a Hindi pronunciation label.'}
    if args.stdout_only:
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        return
    write_group(args.out, {'result.json': payload,
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
            for p in (Path(__file__), ROOT / 'hindi_detector/asr_gate.py')]},
        'Query lexical-content check before phonetic inference. result.json exact query SHA/text, '
        'Native Zipformer/IndicConformer and Hindi IndicConformer hypotheses, WER/CER/roman-CER, '
        'pinned runtime and frozen decision. No accent/source/error label inferred. No raw audio '
        'or model copy, NOT on HF; inherited source rights. Current consumer input validity; '
        'review2026-11-10, compact receipt retained with inference; logs expire24h after consumer closes.',
        ['.venv-asr-gate/bin/python -m scripts.check_query_content --query WAV --text TEXT --out NEW'])
    print(json.dumps({'decision': result['decision'], 'sha256': digest}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Query content check failed explicitly')
        raise
