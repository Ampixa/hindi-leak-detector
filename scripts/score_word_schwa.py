"""Direct isolated-word inference for the auxiliary schwa intervention components."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import predict_linear
from hindi_detector.interventions import LABEL_SOURCE, intervention_features
from hindi_detector.schwa_probe import compare_site, site_paths
from hindi_detector.word_evidence import WordEvidenceEngine

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Score an explicitly isolated word; does not detect word boundaries or classify actual Hindi leakage.')
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--candidate-id', required=True)
    parser.add_argument('--models', type=Path, default=ROOT / 'reports/intervention_heads_v7/selection/models.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh inference output group')
    matches = [c for c in json.loads((ROOT / 'data/schwa_candidates_v1/candidates.json').read_text()) if c['candidate_id'] == args.candidate_id]
    if len(matches) != 1:
        raise ValueError('Need one exact existing lexical candidate; spelling/meaning/target are not guessed')
    candidate = matches[0]
    models = json.loads(args.models.read_text())
    if set(models) != {'ctc_site', 'pooled_phone'} or any(m['label_source'] != LABEL_SOURCE for m in models.values()):
        raise ValueError('Expected explicit intervention components, not source/human leakage heads')
    wave, rate = sf.read(args.audio, dtype='float32', always_2d=True)
    if wave.shape[1] != 1 or not np.isfinite(wave).all() or not .2 <= len(wave) / rate <= 15 or np.max(np.abs(wave)) < 1e-5:
        raise ValueError('Need finite non-silent explicitly isolated mono word audio between .2 and15 seconds')
    divisor = int(np.gcd(rate, 16000))
    samples = resample_poly(wave[:, 0], 16000 // divisor, rate // divisor).astype('float32') if rate != 16000 else wave[:, 0]
    engine = WordEvidenceEngine(ROOT)
    if any(m['phone_encoder_revision'] != engine.phone_inventory['revision'] for m in models.values()):
        raise ValueError('Intervention component and probe revisions differ')
    with (ROOT / '.sources/frontend/data/frontend/ipa_map.tsv').open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != ['phone', 'ipa_candidate', 'status', 'notes']:
            raise ValueError('IPA candidate map schema changed')
        ipa = {r['phone']: r['ipa_candidate'] for r in reader}
    with engine.torch.inference_mode():
        logits = engine.phone(**engine.extractor(samples, sampling_rate=16000, return_tensors='pt')).logits[0]
        log_probs = logits.log_softmax(-1).cpu().numpy()
    try:
        paths = site_paths(candidate, ipa, engine.vocab)
        if len(paths) != 1:
            raise ValueError('Direct component supports exactly one explicit site')
        site = compare_site(log_probs, paths[0])
    except ValueError as error:
        LOG.warning('Word path unavailable: %s', error)
        result = {'status': 'unavailable_no_replacement_score', 'site_evidence': None,
                  'reduced_vowel_request_scores': None, 'unavailable_reason': str(error)}
    else:
        # Feature/model corruption is a program failure, never a phonetic abstention.
        vectors = intervention_features(log_probs, site, candidate, engine.vocab, engine.special)
        scores = {branch: float(predict_linear(model, vectors[branch][None])[0]) for branch, model in models.items()}
        result = {'status': 'isolated_word_intervention_evidence_not_actual_hindi_classification',
                  'site_evidence': site, 'reduced_vowel_request_scores': scores, 'unavailable_reason': None}
    result.update(word=candidate['word'], candidate_id=args.candidate_id, audio_sha256=sha(args.audio),
                  models_sha256=sha(args.models), seconds=len(samples) / 16000,
                  actual_hindi_error_probability=None, native_variant_support=None, preferred_style_distance=None,
                  interpretation='The trained scores describe synthetic reduced-vowel requests. A reduced vowel can be valid Nepali variation. Native acceptability, Hindi influence and prosody are not inferred.',
                  input_contract='Caller supplies an isolated word. This command does not establish boundary correctness, pronunciation target validity, or intended word identity.')
    files = [Path(__file__), ROOT / 'hindi_detector/interventions.py', ROOT / 'hindi_detector/schwa_probe.py', ROOT / 'hindi_detector/ctc.py', ROOT / 'hindi_detector/word_evidence.py']
    write_group(args.out, {'analysis.json': result, 'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in files]},
                'Direct isolated-word inference for the experimentally trained schwa intervention components; preserves unavailable paths and unknown Hindi/native/style judgments.',
                [str(args.audio.resolve()), str(args.models.resolve()),
                 '.venv-indicf5/bin/python -m scripts.score_word_schwa --audio WORD.wav --candidate-id schwa-ID --models MODELS --out NEW'])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Direct word schwa inference failed; no guessed identity, phone target or error label')
        raise
