"""Frozen-model sensitivity to gain and global F0, with fixed spectral envelopes."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import traceback

import numpy as np
import pyworld
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.pronunciation_sites import compile_word, evidence_for_site, paired_site_features, windows
from hindi_detector.vowel_detector import decisions, load, score
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/'reports/vowel_voice_null_v20'
    base = ROOT/'reports/swar_corpus_features_v18'
    model = ROOT/'reports/vowel_detector_v19'
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((model/'report.json').read_text())
    if sha(model/'model.pt') != report['model_sha256']:
        raise ValueError('Frozen model identity differs')
    network, state = load(model/'model.pt')
    write_group(out, {'protocol.json': {'model_sha256': report['model_sha256'],
        'conditions': ['gain_half', 'world_original_f0', 'world_plus3_semitones', 'world_minus3_semitones'],
        'speakers': ['yt01', 'yt10', 'yt12'], 'selected_before_output_scores': True,
        'world': {'f0_method': 'DIO/stonemask', 'frame_period_ms': 5, 'f0_floor': 50, 'f0_ceiling': 650,
                  'cheaptrick_f0_floor': 50, 'd4c_threshold': 0, 'spectral_envelope_and_aperiodicity_fixed_across_F0_conditions': True},
        'policy': 'No refitting or threshold changes. Newly introduced flags measure sensitivity to gain/vocoder/F0, not actual Hindi-error false positives.'},
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Frozen-model nuisance audit on held-out recorded speakers. controls/*.wav store exact gain/WORLD '
        'waveforms; items.jsonl pins each source/control SHA and measured word-site predictions. report.json '
        'separates reconstruction effects from added global-F0 flags, with per-case counts. World F0 shifts '
        'preserve the estimated spectral envelope and relative F0 shape, but vocoder reconstruction can '
        'change phonetic detail; this is a sensitivity test, not human accent gold. Current consumer morning '
        'detector validation; review2026-11-09. No HF upload; source/MMS research licences. Remove redundant '
        'waveforms24h after final consumer closes, retain compact metrics/pins/config.',
        ['.venv-indicf5/bin/python -m scripts.audit_vowel_voice_null'])
    (out/'controls').mkdir()

    def status(phase, **extra):
        p = out/'status.json'; temp = p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(),
            'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2)+'\n'); temp.replace(p)

    try:
        status('loading_frozen_phone_encoder')
        engine = WordEvidenceEngine(ROOT, device='mps', pitch_method='dio', temporal_bins=20)
        runtime = json.loads((base/'runtime.json').read_text())
        if engine.phone_inventory != runtime['phone']:
            raise ValueError('Actual and null-test phone model differ')
        plan = ROOT/'data/swar_phone_plan_v18'
        ipa = json.loads((plan/'ipa_map.json').read_text())
        rows = {r['word']: r for r in json.loads((plan/'words.json').read_text())}
        source = ROOT/'data/swar_speech_v18'
        selection = {r['id']: r for r in json.loads((source/'selection.json').read_text())}
        real_receipts = {r['id']: r for r in read_rows(base/'items.jsonl') if r['kind']=='real'}
        baseline = defaultdict(list)
        for row in read_rows(ROOT/'reports/swar_sites_v18/items.jsonl'):
            if row['source']=='real' and selection[row['id']]['voice'] in {'yt01','yt10','yt12'}:
                baseline[row['id']].append(row)
        compiled = {w: compile_word(rows[w], ipa, runtime['vocabulary']) for w in {r['word'] for v in baseline.values() for r in v}}
        observations = []
        for number, key in enumerate(sorted(baseline)):
            receipt = real_receipts[key]
            original = load_item(base, receipt)
            samples = audio(source, {'audio': receipt['path'], 'sha256': receipt['sha256']})
            x = samples.astype(np.float64)
            f0, times = pyworld.dio(x, 16000, f0_floor=50, f0_ceil=650, frame_period=5)
            f0 = pyworld.stonemask(x, f0, times, 16000)
            spectral = pyworld.cheaptrick(x, f0, times, 16000, f0_floor=50)
            ap = pyworld.d4c(x, f0, times, 16000, threshold=0, fft_size=(spectral.shape[1]-1)*2)
            conditions = {'gain_half': samples*.5}
            for name, shift in [('world_original_f0', 0), ('world_plus3_semitones', 3), ('world_minus3_semitones', -3)]:
                waveform = pyworld.synthesize(np.ascontiguousarray(f0*(2**(shift/12))), spectral, ap, 16000, frame_period=5)
                waveform = np.pad(waveform, (0, max(0,len(samples)-len(waveform))))[:len(samples)]
                peak = np.max(np.abs(waveform))
                if peak > .98:
                    waveform *= .98/peak
                conditions[name] = waveform.astype(np.float32)
            for condition, waveform in conditions.items():
                path = out/'controls'/(key+'_'+condition+'.wav')
                sf.write(path, waveform, 16000, subtype='PCM_16')
                actual = sf.read(path, dtype='float32')[0]
                with engine.torch.inference_mode():
                    encoded = engine.phone(**{k: v.to(engine.device) for k, v in
                        engine.extractor(actual, sampling_rate=16000, return_tensors='pt').items()})
                    lp = encoded.logits[0].log_softmax(-1).cpu().numpy()
                item = {'detail': {**original['detail'], 'phone_frame_seconds': len(actual)/16000/len(lp)},
                        'phone_log_probs': lp}
                for row in baseline[key]:
                    word = compiled[row['word']]
                    variant = next(v for v in word['variants'] if v['variant_id']==row['variant_id'])
                    contrast = next(c for c in variant['contrasts'] if c['kind']==row['site']['kind'] and c['index']==row['site']['index'])
                    evidence = evidence_for_site(word, variant, row['reference']['reference_supported_context_reductions'],
                                                 contrast, windows(item, row['word_ordinal']))
                    features = paired_site_features(evidence, row['reference'], contrast)[None]
                    probability = score(network, state, features)
                    decision = decisions(state, features, probability)[0]
                    prediction = {'id': key, 'speaker': selection[key]['voice'], 'condition': condition,
                        'word': row['word'], 'ordinal': row['word_ordinal'], 'site': row['site'],
                        'score': float(probability[0]), 'decision': decision, 'features': features[0].tolist(),
                        'source_sha256': receipt['sha256'], 'audio': str(path.relative_to(out)),
                        'audio_sha256': sha(path), 'audio_bytes': path.stat().st_size}
                    observations.append(prediction)
                    with (out/'items.jsonl').open('a') as f:
                        f.write(json.dumps(prediction, ensure_ascii=False)+'\n')
            status('auditing_gain_and_F0_sensitivity', completed_cases=number+1, total_cases=len(baseline))
        summary = {}
        for condition in conditions:
            subset = [r for r in observations if r['condition']==condition]
            flags = [r for r in subset if r['decision']=='strong_vowel_change']
            summary[condition] = {'site_observations': len(subset), 'new_site_flags': len(flags),
                'site_flag_fraction': len(flags)/len(subset), 'cases': len(baseline),
                'cases_with_any_new_flag': len({r['id'] for r in flags})}
        matched = defaultdict(dict)
        for row in observations:
            matched[row['id'], row['ordinal'], row['site']['kind'], row['site']['index']][row['condition']] = row
        added = {}
        for condition in ('world_plus3_semitones','world_minus3_semitones'):
            pairs = [r for r in matched.values() if condition in r and 'world_original_f0' in r]
            added[condition] = {'sites': len(pairs), 'added_flags_over_unchanged_vocoder': sum(
                r[condition]['decision']=='strong_vowel_change' and r['world_original_f0']['decision']!='strong_vowel_change' for r in pairs),
                'median_score_shift': float(np.median([r[condition]['score']-r['world_original_f0']['score'] for r in pairs]))}
        result = {'status': 'complete_frozen_voice_nuisance_audit', 'model_sha256': report['model_sha256'],
            'summary': summary, 'F0_added_effect': added, 'actual_hindi_error_accuracy': None,
            'source_cases': len(baseline), 'speakers': sorted({r['speaker'] for r in observations}),
            'limits': 'New comparison flags under controlled gain/WORLD/F0 changes; not a human pronunciation-error false-positive benchmark.'}
        (out/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        if sha(model/'model.pt') != report['model_sha256']:
            raise ValueError('Frozen model changed during null audit')
        status('complete_voice_nuisance_audit', cases=len(baseline))
    except Exception:
        status('failed_explicitly', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Frozen vowel voice-nuisance audit failed explicitly')
        raise
