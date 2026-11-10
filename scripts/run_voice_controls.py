"""Common-decoder identity resynthesis and voice swaps, with preservation gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from hindi_detector.evaluation import predict_linear
from hindi_detector.review_server import read_items
from hindi_detector.voice_control import VoiceControlConverter
from hindi_detector.word_evidence import WordEvidenceEngine
from hindi_detector.word_model import load_bags, predict_bag

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def read16(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    if x.shape[1] != 1 or not np.isfinite(x).all():
        raise ValueError('Control inputs must be finite mono WAVs')
    divisor = int(np.gcd(sr, 16000))
    return resample_poly(x[:, 0], 16000 // divisor, sr // divisor).astype('float32') if sr != 16000 else x[:, 0]


def bag_for(record, arrays):
    return {'words': record['words'], 'phonetics': arrays['phonetics'], 'prosody': arrays['prosody'],
            'phone_mask': np.array([w['phone_evidence_available'] for w in record['words']], dtype=bool)}


def preservation(original, changed, old_bag, new_bag, gate):
    if [w['word'] for w in original['words']] != [w['word'] for w in changed['words']]:
        raise ValueError('Control word identities changed')
    timing = float(np.mean([abs(a[key] - b[key]) for a, b in zip(original['words'], changed['words'], strict=True) for key in ['start', 'end']]))
    dimension = old_bag['phonetics'].shape[1] // 2
    a, b = old_bag['phonetics'][:, :dimension], new_bag['phonetics'][:, :dimension]
    cosine = np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    grid = np.linspace(0, 1, 200)
    pitch = []
    for item in [original, changed]:
        times = np.asarray(item['tracks']['times']) / item['seconds']
        values = np.array([np.nan if v is None else v for v in item['tracks']['pitch_relative_st']])
        pitch.append(np.interp(grid, times, values))
    common = np.isfinite(pitch[0]) & np.isfinite(pitch[1])
    correlation = float(np.corrcoef(pitch[0][common], pitch[1][common])[0, 1]) if common.sum() >= 20 and min(np.std(p[common]) for p in pitch) > .1 else None
    ratio = changed['seconds'] / original['seconds']
    passed = gate['duration_ratio'][0] <= ratio <= gate['duration_ratio'][1] and timing <= gate['word_timing_mae_seconds'] and float(cosine.mean()) >= gate['mean_phone_posterior_cosine'] and correlation is not None and correlation >= gate['relative_pitch_correlation']
    return {'duration_ratio': ratio, 'word_timing_mae_seconds': timing, 'mean_phone_posterior_cosine': float(cosine.mean()),
            'relative_pitch_correlation': correlation, 'approximate_preservation_gate_passed': passed,
            'gate_meaning': 'Automatic diagnostic qualification only; not proof that every Nepali phone/accent is preserved'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not (args.out / 'README.md').is_file() or (args.out / 'report.json').exists() or list(args.out.glob('*.wav')):
        raise ValueError('Need a documented new voice-control output group')
    plan_path = ROOT / 'reports/voice_controls_v3_plan.json'
    plan = json.loads(plan_path.read_text())
    sources = []
    for p in [Path(__file__), ROOT / 'hindi_detector/voice_control.py', ROOT / 'hindi_detector/word_evidence.py', ROOT / 'hindi_detector/local_features.py', ROOT / 'hindi_detector/word_model.py']:
        content = p.read_bytes()
        sources.append({'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(content).hexdigest(), 'source_utf8': content.decode()})
    snapshot = args.out / 'source_snapshot.json'
    if snapshot.exists():
        raise FileExistsError('Inspect prior control source snapshot before retry')
    snapshot.write_text(json.dumps(sources, indent=2) + '\n')
    pack = ROOT / 'data/indicf5_audio8_v1'; registry, _ = read_items(pack)
    selection = json.loads((pack / 'selection.json').read_text())
    questions = [j['id'] for j in selection['jobs'] if '?' in j['text']]
    if not questions:
        raise ValueError('No actual question-text control available')
    chosen_ids = ['sent_085', questions[0]]
    selected = [r for r in registry.values() if r['sentence_id'] in chosen_ids and r['system'] in {'native_recording', 'audio8_dpo_R3', 'indicf5_stock'}]
    if len(selected) != 6:
        raise ValueError('Expected two matched three-source sentences')
    probe = json.loads((ROOT / 'data/native_probe_v2/selection.json').read_text())
    targets = probe['clips'][4:6]
    if len(targets) != 2:
        raise ValueError('Need two independently grouped target references')
    converter = VoiceControlConverter(ROOT)
    styles = {}
    for label, target in zip(['voice_a', 'voice_b'], targets, strict=True):
        p = Path(target['audio'])
        if hashlib.sha256(p.read_bytes()).hexdigest() != target['sha256']:
            raise ValueError('Target reference identity mismatch')
        styles[label] = converter.style(p)
    converted = []
    for row in selected:
        source = pack / row['audio']
        if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Source identity changed')
        original_style = converter.style(source)
        for control in ['identity', 'voice_a', 'voice_b']:
            target_style = original_style if control == 'identity' else styles[control]
            wave, sr = converter.convert(source, original_style, target_style)
            path = args.out / (row['clip_id'] + '_' + control + '.wav')
            clipping = float(np.mean(np.abs(wave) >= 1))
            sf.write(path, wave, sr, subtype='PCM_16')
            converted.append({'original': row, 'control': control, 'path': path.name,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size,
                              'clipping_fraction_before_pcm': clipping})
            with (args.out / 'generated.jsonl').open('a') as handle:
                handle.write(json.dumps(converted[-1], ensure_ascii=False) + '\n'); handle.flush(); os.fsync(handle.fileno())
            LOG.info('Converted %s %s', row['system'], control)
    del converter
    models = json.loads((ROOT / 'reports/wordlab_models_v3/models.json').read_text())
    baseline = json.loads((ROOT / 'reports/source_proxy_v1/model.json').read_text())
    original_bags, _ = load_bags(ROOT / 'reports/wordlab_test_v3')
    lookup = {b['clip_id']: b for b in original_bags}
    engine = WordEvidenceEngine(ROOT)

    def global_score(samples):
        with engine.torch.inference_mode():
            out = engine.phone(**engine.extractor(samples, sampling_rate=16000, return_tensors='pt'), output_hidden_states=True)
        vectors = []
        for layer in [12, 24]:
            values = out.hidden_states[layer][0].cpu().numpy(); vectors.extend([values.mean(0), values.std(0)])
        return float(predict_linear(baseline, np.concatenate(vectors)[None])[0])

    results = []
    originals = {}
    for row in selected:
        original = json.loads((ROOT / 'reports/wordlab_test_v3' / (row['clip_id'] + '.json')).read_text())
        bag = lookup[row['clip_id']]
        originals[row['clip_id']] = (original, bag, global_score(read16(pack / row['audio'])))
    for item in converted:
        row = item['original']; path = args.out / item['path']; original, old_bag, old_global = originals[row['clip_id']]
        changed, arrays = engine.analyze(read16(path), row['text'])
        bag = bag_for(changed, arrays)
        quality = preservation(original, changed, old_bag, bag, plan['approximate_preservation_gate'])
        if item['clipping_fraction_before_pcm'] > .005:
            quality['approximate_preservation_gate_passed'] = False
        scores = {}
        for name, model in models.items():
            a, old_words = predict_bag(model, old_bag); b, new_words = predict_bag(model, bag)
            deltas = [abs(v - u) for u, v in zip(old_words, new_words, strict=True) if u is not None and v is not None]
            scores[name] = {'original_bag': a, 'converted_bag': b, 'bag_score_delta': b - a,
                            'max_absolute_word_score_delta': max(deltas) if deltas else None}
        new_global = global_score(read16(path))
        result = {**item, 'quality': quality, 'word_branches': scores,
                  'old_global_model': {'original_score': old_global, 'converted_score': new_global, 'delta': new_global - old_global}}
        detail = args.out / (path.stem + '.json')
        detail.write_text(json.dumps(changed, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        result.update(detail=detail.name, detail_sha256=hashlib.sha256(detail.read_bytes()).hexdigest(), detail_bytes=detail.stat().st_size)
        results.append(result); LOG.info('Control analyzed %s %s gate=%s', row['system'], item['control'], quality['approximate_preservation_gate_passed'])
    report = {'creator': 'Codex GPT-6 /root', 'status': 'voice_confounds_measured_not_proven_absent',
              'source_hashes': {s['path']: s['sha256'] for s in sources},
              'word_models_sha256': hashlib.sha256((ROOT / 'reports/wordlab_models_v3/models.json').read_bytes()).hexdigest(),
              'voice_converter_model_revision': 'f36e7edfe1684461a8343844af60babc2efbb727',
              'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(), 'controls': results,
              'qualified_controls': sum(r['quality']['approximate_preservation_gate_passed'] for r in results),
              'target_references': [{k: t[k] for k in ['speaker_id', 'audio', 'sha256']} for t in targets],
              'limits': ['Voice conversion itself can change phones/prosody; failed-preservation cases cannot establish voice sensitivity.',
                         'Identity resynthesis separates common-decoder artifacts from additional voice conversion effects.',
                         'Automatic phone/pitch preservation checks are imperfect and do not establish human phonetic identity.',
                         'Source/word models remain weakly supervised; no word-level Hindi accuracy follows from this test.']}
    (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'controls': len(results), 'qualified': report['qualified_controls']}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Voice-confound experiment failed; inspect partial artifacts before retry')
        raise
