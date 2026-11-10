"""Run the actual vowel-change model on new same-text query/reference audio."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_g2p import DualG2P
from hindi_detector.pronunciation_sites import (compile_word, evidence_for_site, paired_site_features,
    select_reference_structure, windows)
from hindi_detector.vowel_detector import decisions, load, score
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=ROOT/'reports/vowel_detector_v19/model.pt')
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['auto','mps','cpu'], default='auto')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    network, state = load(args.model)
    for path, key in [(ROOT/'data/swar_phone_plan_v18/words.json', 'phone_plan_sha256'),
                      (ROOT/'hindi_detector/phone_lattice.py', 'projection_code_sha256'),
                      (ROOT/'hindi_detector/pronunciation_sites.py', 'site_code_sha256')]:
        if sha(path) != state[key]:
            raise ValueError('Model dependency changed: ' + key)
    phonemizer = DualG2P(ROOT)
    identities, waves = [], []
    for path in (args.query, args.reference):
        path = path.resolve(strict=True)
        digest = sha(path)
        waves.append(audio(path.parent, {'audio': path.name, 'sha256': digest}))
        identities.append({'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size})
    long_pair = max(map(len, waves)) > 20*16000
    device = ('cpu' if long_pair else 'mps') if args.device=='auto' else args.device
    if device=='mps' and long_pair:
        raise ValueError('Long inputs require CPU for both sides of the same-text comparison')
    engine = WordEvidenceEngine(ROOT, device=device, pitch_method='dio', temporal_bins=20)
    expected = json.loads((ROOT/'reports/vowel_controls_v19/runtime.json').read_text())
    if engine.phone_inventory != expected['phone']:
        raise ValueError('Selected acoustic encoder differs from model training')
    items = []
    for samples in waves:
        detail, _ = engine.analyze(samples, args.text)
        with engine.torch.inference_mode():
            encoded = engine.phone(**{k: v.to(device) for k,v in
                engine.extractor(samples, sampling_rate=16000, return_tensors='pt').items()})
            lp = encoded.logits[0].log_softmax(-1).cpu().numpy()
        if not np.isfinite(lp).all():
            raise ValueError('Nonfinite actual query/reference phone frames')
        detail['phone_frame_seconds'] = len(samples)/16000/len(lp)
        items.append({'detail': detail, 'phone_log_probs': lp})
    query, reference = items
    if [w['word'] for w in query['detail']['words']] != [w['word'] for w in reference['detail']['words']]:
        raise ValueError('Same-text word identity changed during alignment')
    words, site_records, vectors = [], [], []
    for ordinal, word in enumerate(query['detail']['words']):
        record = {**word, 'sites': [], 'unavailable_reasons': [], 'actual_hindi_error_probability': None}
        try:
            hypothesis = phonemizer.word(word['word'])
            compiled = compile_word(hypothesis, phonemizer.ipa, engine.vocab)
            rcrops = windows(reference, ordinal)
            variant, reduced = select_reference_structure(compiled, rcrops)
            record['g2p_source'] = hypothesis['nepali_default']['source']
            for contrast in variant['contrasts']:
                try:
                    revidence = evidence_for_site(compiled, variant, reduced, contrast, rcrops)
                    qevidence = evidence_for_site(compiled, variant, reduced, contrast, windows(query, ordinal))
                    vector = paired_site_features(qevidence, revidence, contrast)
                    site = {'word_ordinal': ordinal, 'site': {k:v for k,v in contrast.items() if k!='alternative_unit'},
                        'native_phones': variant['phones'], 'query': qevidence, 'reference': revidence,
                        'features': vector.tolist()}
                    site_records.append(site); vectors.append(vector)
                    record['sites'].append(site)
                except ValueError as error:
                    logging.warning('Explicit unavailable pronunciation site %s: %s', word['word'], error)
                    record['unavailable_reasons'].append(str(error))
        except ValueError as error:
            logging.warning('Explicit unsupported word %s: %s', word['word'], error)
            record['unavailable_reasons'].append(str(error))
        words.append(record)
    if vectors:
        values = np.stack(vectors)
        probabilities = score(network, state, values)
        result_decisions = decisions(state, values, probabilities)
        for site, probability, decision in zip(site_records, probabilities, result_decisions, strict=True):
            site.update(controlled_vowel_change_score=float(probability), decision=decision,
                        actual_hindi_error_probability=None)
    for word in words:
        word['decision'] = ('strong_vowel_change' if any(s['decision']=='strong_vowel_change' for s in word['sites'])
            else 'uncertain' if any(s['decision']=='uncertain' for s in word['sites'])
            else 'no_strong_vowel_change' if word['sites'] else 'untested_site')
    result = {'text': args.text, 'model_sha256': sha(args.model), 'encoder_device': device,
        'inputs': identities, 'words': words, 'word_flags': [w['word'] for w in words if w['decision']=='strong_vowel_change'],
        'content_checked_with_asr': False, 'actual_hindi_error_accuracy': None,
        'scope': 'Hindi-compatible vowel deletion/quality contrasts against the trusted same-text reference. Controlled-change scores are not accent probabilities.',
        'limits': 'No flag is not a correctness guarantee; unsupported sounds abstain. Numeric text needs verified spoken expansion. Hindi-specific intonation correctness is not established.'}
    write_group(args.out, {'result.json': result, 'source_snapshot.json': [
        {'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
        for p in [Path(__file__), ROOT/'hindi_detector/dual_g2p.py']]},
        'Direct generic inference on actual new audio/text. result.json preserves exact query/reference hashes, '
        'word intervals, phonetic sequences, dual-G2P site evidence and frozen-model decisions. No fixed-case '
        'lookup or hardcoded word prediction. No audio copy or HF upload. Current consumer detector verification; '
        'review2026-11-09, keep compact result/model pins through final review; source/MMS research licences apply.',
        ['.venv-indicf5/bin/python -m scripts.detect_vowel_words --query WAV --reference TRUSTED_SAME_TEXT_WAV --text NEPALI --out NEW'])
    print(json.dumps({'word_flags': result['word_flags'], 'tested_sites': len(site_records), 'model_sha256': result['model_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Direct vowel-word inference failed explicitly')
        raise
