"""Independent native-control continuation for the crossed Pocket candidate."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import soundfile as sf
import torch
import yaml
from pocket_tts import TTSModel

from hindi_detector.reference_words import pair_words
from hindi_detector.word_model import load_bags, predict_bag
from scripts.run_pocket_matrix import ROOT, SOURCE, SOURCE_REVISION, save, sha

PACK = ROOT / 'data/pocket_native_challenge_v4'
STATE = ROOT / 'reports/pocket_native_validation_v4'
LOG = logging.getLogger(__name__)


def state(phase, **values):
    save(STATE / 'status.json', {'phase': phase, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(), **values})


def main():
    os.nice(10)
    if not (PACK / 'README.md').is_file() or not (STATE / 'README.md').is_file() or (PACK / 'run.json').exists():
        raise ValueError('Need fresh documented native challenge/state groups')
    torch.set_num_threads(2)
    if subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE_REVISION:
        raise ValueError('Pocket source revision changed')
    subprocess.run(['git', '-C', str(SOURCE), 'diff', '--quiet', 'HEAD'], check=True)
    probe = json.loads((ROOT / 'data/native_probe_v2/selection.json').read_text())
    matrix = json.loads((ROOT / 'data/pocket_wordmatrix_v4/plan.json').read_text())
    references = {}
    for job in matrix['jobs']:
        if job['split'] == 'development':
            references.setdefault(job['speaker_id'], job)
    refs = sorted(references.values(), key=lambda j: j['speaker_id'])
    if len(refs) != 4:
        raise ValueError('Need four actual development reference groups outside model fitting')
    source_jobs = {j['id']: j for j in probe['jobs']}
    if len(source_jobs) != 32 or len(probe['clips']) != 32:
        raise ValueError('Unexpected original native-control shape')
    weights = Path(json.loads((ROOT / 'reports/pocket_pilot_v1/manifest.json').read_text())['finetune']['path'])
    if sha(weights) != matrix['weight_sha256']['pocket_nepali_v1c']:
        raise ValueError('Native challenge teacher checkpoint changed')
    config = yaml.safe_load((ROOT / 'reports/pocket_pilot_v1/nepali_v1c.yaml').read_text())
    config['weights_path'] = str(weights); config['weights_path_without_voice_cloning'] = str(weights)
    cfg = PACK / 'inference.yaml'; cfg.write_text(yaml.safe_dump(config))
    save(PACK / 'run.json', {'creator': 'Codex GPT-6 /root', 'created_utc': datetime.now(timezone.utc).isoformat(),
                            'status': 'running', 'checkpoint_sha256': sha(weights), 'driver_sha256': sha(Path(__file__)),
                            'probe_selection_sha256': sha(ROOT / 'data/native_probe_v2/selection.json'),
                            'reference_policy': 'Different original source group, four held-out development reference voices; never condition on the query recording.',
                            'settings': matrix['settings']})
    save(STATE / 'source_snapshot.json', [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
                                        for p in [Path(__file__), ROOT / 'hindi_detector/word_model.py', ROOT / 'hindi_detector/reference_words.py']])
    state('generating_native_reference_readings', completed=0, planned=32)
    model = TTSModel.load_model(config=cfg, temp=.3, sampler_decode_steps=1, eos_threshold=-4.).eval()
    model.requires_grad_(False)
    if model.device.type != 'cpu' or model.sample_rate != 24000:
        raise ValueError('Unexpected native challenge runtime placement')
    jobs, clips, states = [], [], {}
    waves = PACK / 'pocket'; waves.mkdir(exist_ok=True)
    for i, native in enumerate(probe['clips']):
        original = source_jobs[native['sentence_id']]; ref = refs[i % len(refs)]
        if native['speaker_id'] == ref['speaker_id'] or native['text'] != original['text']:
            raise ValueError('Native challenge reference/text mismatch')
        if sha(Path(native['audio'])) != native['sha256'] or sha(Path(ref['reference_audio'])) != ref['reference_sha256']:
            raise ValueError('Native challenge audio identity changed')
        if native['text'] in {j['text'] for j in matrix['jobs']}:
            raise ValueError('Native challenge duplicates a fitted matrix prompt')
        identifier = f'pocket_native_{i:02d}'
        jobs.append({'id': identifier, 'speaker_id': native['speaker_id'], 'text': native['text'], 'split': 'reserved_evaluation',
                     'reference_source_group': ref['speaker_id'], 'reference_sha256': ref['reference_sha256']})
        clips.append({k: native[k] for k in ['audio', 'sha256', 'text', 'speaker_id']} |
                     {'clip_id': hashlib.sha256(('native-challenge-v4:' + native['clip_id']).encode()).hexdigest()[:24],
                      'sentence_id': identifier, 'system': 'native_recording', 'split': 'reserved_evaluation', 'automatic_leakage_label': None})
        ref_key = ref['reference_sha256']
        with torch.no_grad():
            if ref_key not in states:
                states[ref_key] = model.get_state_for_audio_prompt(Path(ref['reference_audio']))
            torch.manual_seed(0)
            wave = model.generate_audio(states[ref_key], native['text'], copy_state=True).detach().cpu().numpy()
        if wave.ndim == 2 and wave.shape[0] == 1:
            wave = wave[0]
        if wave.ndim != 1 or not np.isfinite(wave).all() or not .2 <= len(wave) / 24000 <= 30:
            raise ValueError('Invalid native-reference synthesis; inspect rather than substitute')
        path = waves / (identifier + '.wav'); sf.write(path, wave, 24000, subtype='PCM_16')
        clips.append({'clip_id': hashlib.sha256(('pocket-challenge-v4:' + identifier).encode()).hexdigest()[:24],
                      'sentence_id': identifier, 'system': 'pocket_nepali_v1c', 'split': 'reserved_evaluation',
                      'speaker_id': ref['speaker_id'], 'text': native['text'], 'audio': str(path.relative_to(PACK)),
                      'sha256': sha(path), 'bytes': path.stat().st_size, 'automatic_leakage_label': None})
        state('generating_native_reference_readings', completed=i + 1, planned=32)
    save(PACK / 'selection.json', {'schema_version': 1, 'jobs': jobs, 'clips': clips, 'external_audio_roots': probe['external_audio_roots']})
    run = json.loads((PACK / 'run.json').read_text()); run.update(status='complete_unreviewed_audio', generated=32); save(PACK / 'run.json', run)
    del model, states
    state('extracting_native_word_evidence')
    with (STATE / 'extract.log').open('w') as log:
        subprocess.run([str(ROOT / '.venv-indicf5/bin/python'), '-m', 'scripts.extract_word_evidence', '--pack', str(PACK),
                        '--out', 'reports/wordlab_pocket_native_v4'], cwd=ROOT,
                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2'},
                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    state('waiting_for_crossed_candidate')
    while True:
        pipeline = json.loads((ROOT / 'reports/pocket_pipeline_v4/status.json').read_text())
        if pipeline['status'] == 'candidate_trained_reliability_unvalidated':
            break
        if pipeline['status'].startswith('failed'):
            raise RuntimeError('Primary candidate pipeline failed; native evaluation cannot proceed')
        time.sleep(15)
    directory = ROOT / 'reports/wordlab_pocket_native_v4'
    bags, manifest = load_bags(directory)
    if manifest['failures'] or len(bags) != 64:
        raise ValueError('Native challenge word coverage incomplete')
    models_path = ROOT / 'reports/pocket_crossed_models_v4/models.json'; models = json.loads(models_path.read_text())
    groups = {}
    for bag in bags:
        group = groups.setdefault(bag['sentence_id'], {})
        if bag['system'] in group:
            raise ValueError('Duplicate native challenge source')
        group[bag['system']] = bag
    predictions = []
    for identifier, group in groups.items():
        if set(group) != {'native_recording', 'pocket_nepali_v1c'}:
            raise ValueError('Native challenge pair incomplete')
        a, b = group['native_recording'], group['pocket_nepali_v1c']
        if a['speaker_id'] == b['speaker_id']:
            raise ValueError('Native control accidentally uses same reference source group')
        pair = pair_words(a, b)
        values = {}
        for branch, model in models.items():
            bag_score, words = predict_bag(model, pair)
            values[branch] = {'bag_source_evidence': bag_score, 'word_source_evidence': words,
                              'high_word_count': sum(w is not None and w >= .5 for w in words),
                              'word_coverage': sum(w is not None for w in words) / len(words)}
        predictions.append({'sentence_id': identifier, 'native_source_group': a['speaker_id'], 'reference_source_group': b['speaker_id'],
                            'native_audio_sha256': a['audio_sha256'], 'reference_audio_sha256': b['audio_sha256'],
                            'words': [{k: w[k] for k in ['word', 'start', 'end']} for w in a['words']], 'branches': values})
    summary = {'status': 'native_control_reference_comparison_complete_reliability_unvalidated', 'native_control_clips': len(predictions),
               'models_sha256': sha(models_path), 'reliable_automatic_detector': False,
               'branches': {name: {'high_bag_scores': sum(r['branches'][name]['bag_source_evidence'] >= .5 for r in predictions),
                                   'clips_with_any_high_word': sum(r['branches'][name]['high_word_count'] > 0 for r in predictions),
                                   'word_count': sum(len(r['words']) for r in predictions),
                                   'high_word_scores': sum(r['branches'][name]['high_word_count'] for r in predictions)} for name in models},
               'limits': ['Native corpus identity supplies weak negative controls, not independently judged accent gold.',
                          '32 source groups are insufficient to establish a low error rate across dialects/contexts.',
                          'Native references and F5 probes have previously been examined; this is exploratory diagnostic validation.',
                          'Different valid intonation can disagree with a synthetic preferred reading without being Hindi leakage.']}
    save(STATE / 'predictions.json', predictions); save(STATE / 'report.json', summary)
    state('native_controls_measured_reliability_unvalidated', report='reports/pocket_native_validation_v4/report.json')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Native Pocket validation failed; inspect exact source and partial outputs before retry')
        state('failed_inspect_logs_before_retry')
        raise
