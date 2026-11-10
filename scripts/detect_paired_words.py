"""Run the frozen paired candidate on actual same-text query/reference WAVs."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.paired_detector import BRANCHES, load_model, paired_features, score_arrays, sha
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    network, state = load_model(args.model)
    priors = json.loads((ROOT / 'data/pocket_bulk_plan_v11/g2p_priors.json').read_text())
    words = {r['word'] for r in priors}
    engine = WordEvidenceEngine(ROOT, device=args.device, pitch_method='dio', hidden_layers=(12, 24))
    features = []; identities = []
    for path in (args.query, args.reference):
        path = path.resolve(strict=True); digest = sha(path)
        samples = audio(path.parent, {'audio': path.name, 'sha256': digest})
        detail, arrays = engine.analyze(samples, args.text)
        features.append({'detail': detail, **arrays}); identities.append({'path': str(path), 'sha256': digest})
    p, s = paired_features(*features, state['projection'])
    scores = score_arrays(network, state, p, s)
    result = []
    for word, values in zip(features[0]['detail']['words'], scores, strict=True):
        supported = word['word'] in words
        result.append({**word, 'g2p_contrast_family_supported': supported,
            'source_evidence': dict(zip(BRANCHES, map(float, values))),
            'flags': {b: bool(v >= state['thresholds'][b]) if supported else None for b, v in zip(BRANCHES, values)},
            'actual_hindi_error_probability': None})
    write_group(args.out, {'result.json': {'text': args.text, 'model_sha256': sha(args.model), 'inputs': identities,
        'words': result, 'thresholds': state['thresholds'], 'content_quality_checked': False,
        'warnings': ['Reference must be a trusted Nepali rendition of exactly this text. No automatic content gate in this CLI.',
                     'Scores describe learned source contrast, not validated Hindi-error probabilities.',
                     'Unsupported contrast families abstain. The prosody branch is not separately validated against real intonation errors.']},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in (Path(__file__), ROOT / 'hindi_detector/paired_detector.py')]},
        'Runnable same-text paired inference, word times, phonetic/relative-prosody diagnostics and three source scores. No waveform copied; result.json is the native machine-readable deliverable. Current consumer: morning detector review.',
        ['.venv-indicf5/bin/python -m scripts.detect_paired_words --model MODEL --query WAV --reference NATIVE_WAV --text TEXT --out NEW'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Paired detector inference failed explicitly')
        raise
