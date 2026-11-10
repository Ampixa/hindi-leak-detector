"""Pinned CPU Pocket Nepali/Hindi readings of the existing crossed-word matrix."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf
import torch
import yaml
from pocket_tts import TTSModel

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/Users/k2/madhyam/pocket-tts')
SOURCE_REVISION = 'b8a06e1d7ba379944ab0188342cc6c7f78def088'
LOG = logging.getLogger(__name__)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--source-pack', type=Path, default=ROOT / 'data/wordmatrix_v3')
    parser.add_argument('--systems', nargs='+', choices=['pocket_nepali_v1c', 'pocket_hindi_parent'],
                        default=['pocket_nepali_v1c', 'pocket_hindi_parent'])
    args = parser.parse_args()
    pack = args.pack.resolve(strict=True)
    if not (pack / 'README.md').is_file():
        raise ValueError('Document the new Pocket experiment before generation')
    if subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE_REVISION:
        raise ValueError('Pocket source revision changed')
    subprocess.run(['git', '-C', str(SOURCE), 'diff', '--quiet', 'HEAD'], check=True)
    pilot = json.loads((ROOT / 'reports/pocket_pilot_v1/manifest.json').read_text())
    weights = {'pocket_nepali_v1c': Path(pilot['finetune']['path']),
               'pocket_hindi_parent': ROOT / 'models/pocket_hindi_parent/model.safetensors'}
    expected = {'pocket_nepali_v1c': pilot['finetune']['sha256'], 'pocket_hindi_parent': pilot['parent']['sha256']}
    if len(set(args.systems)) != len(args.systems):
        raise ValueError('Duplicate requested Pocket system')
    weights = {name: weights[name] for name in args.systems}
    expected = {name: expected[name] for name in args.systems}
    for key, path in weights.items():
        if sha(path) != expected[key]:
            raise ValueError('Pocket checkpoint identity changed: ' + key)
    tokenizer = ROOT / 'models/pocket_hindi_parent/tokenizer.model'
    inventory = json.loads((tokenizer.parent / 'inventory.json').read_text())
    entry = next(x for x in inventory if x['path'] == 'tokenizer.model')
    if sha(tokenizer) != entry['sha256']:
        raise ValueError('Pocket tokenizer identity changed')
    base = yaml.safe_load((ROOT / 'reports/pocket_pilot_v1/nepali_v1c.yaml').read_text())
    if base['flow_lm']['transformer']['num_layers'] != 6 or base['mimi']['sample_rate'] != 24000:
        raise ValueError('Pocket configuration differs from verified pilot')
    base['flow_lm']['lookup_table']['tokenizer_path'] = str(tokenizer)
    source_selection = args.source_pack.resolve(strict=True) / 'selection.json'
    selected = json.loads(source_selection.read_text())
    if selected['experiment_id'] == 'wordmatrix_v3':
        if len(selected['jobs']) != 512:
            raise ValueError('Locked original crossed experiment needs512 jobs')
        word_groups = selected['word_groups']; experiment_id = 'pocket_wordmatrix_v4'
    elif selected['experiment_id'] in {'human_matched_v9', 'native_reference_pocket_v10'}:
        source_report = json.loads((source_selection.parent / 'report.json').read_text())
        if len(selected['jobs']) != source_report['jobs_per_tts'] or len(selected['contexts']) != source_report['contexts']:
            raise ValueError('Frozen human-context job counts changed')
        if any(job['split'] != 'development' or not job['target_words'] for job in selected['jobs']):
            raise ValueError('Human-context readings are exploratory development, with explicit target words')
        word_groups = {'development': list(selected['words'])}
        experiment_id = 'pocket_human_matched_v9' if selected['experiment_id'] == 'human_matched_v9' else 'native_reference_pocket_v10'
    else:
        raise ValueError('Only inspected original matrix or verified human-context plan is supported')
    jobs = selected['jobs']
    for job in jobs:
        if Path(job['id']).name != job['id'] or job['seed'] != 0:
            raise ValueError('Unexpected crossed job identity/seed')
        if sha(Path(job['reference_audio'])) != job['reference_sha256']:
            raise ValueError('Crossed voice reference bytes changed')
    output = pack / 'pocket'; output.mkdir(exist_ok=True)
    plan = {'experiment_id': experiment_id, 'creator': 'Codex GPT-6 /root',
            'source_selection_sha256': sha(source_selection), 'pocket_source_revision': SOURCE_REVISION,
            'weight_sha256': expected, 'systems': list(weights), 'jobs': jobs,
            'word_groups': word_groups,
            'settings': {'device': 'cpu', 'threads': 2, 'temperature': .3, 'sampler_decode_steps': 1, 'eos_threshold': -4., 'seed': 0},
            'reference_policy': 'Same verified reference WAV as IndicF5; native Pocket conditions on audio only, not reference transcript.',
            'limits': ['Preferred Nepali checkpoint is user-authorized weak supervision, not per-word correctness gold.',
                       'Hindi parent is an additional same-architecture diagnostic source; historical exact fine-tune initialization hash remains unverified.',
                       'Generated voice identity and intended textual emphasis are unverified.']}
    plan_path = pack / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Resume plan/checkpoint/settings changed')
    else:
        save(plan_path, plan)
    run_path = output / 'run.json'
    run = {'creator': 'Codex GPT-6 /root', 'created_utc': datetime.now(timezone.utc).isoformat(),
           'status': 'running', 'plan_sha256': sha(plan_path), 'driver_sha256': sha(Path(__file__)),
           'torch_version': torch.__version__, 'numpy_version': np.__version__,
           'weight_sha256': expected, 'codec_state_sha256': {}, 'completed': 0, 'systems_completed': []}
    if run_path.exists():
        old = json.loads(run_path.read_text())
        if any(old[k] != run[k] for k in ['plan_sha256', 'driver_sha256', 'weight_sha256', 'torch_version', 'numpy_version']):
            raise ValueError('Resume driver/runtime identity changed')
        run = old; run['status'] = 'running'
    else:
        sources = []
        for p in [Path(__file__), SOURCE / 'pocket_tts/models/tts_model.py']:
            sources.append({'path': str(p), 'sha256': sha(p), 'source_utf8': p.read_text()})
        save(output / 'source_snapshot.json', sources)
    receipt = output / 'items.jsonl'
    rows = [json.loads(x) for x in receipt.read_text().splitlines()] if receipt.exists() else []
    existing = {(r['system'], r['id']): r for r in rows}
    if len(existing) != len(rows):
        raise ValueError('Duplicate Pocket receipts')
    lookup = {j['id']: j for j in jobs}
    for (system, identifier), row in existing.items():
        job = lookup[identifier]
        if system not in weights or row['status'] != 'ok_unreviewed' or sha(pack / row['audio']) != row['sha256']:
            raise ValueError('Retained Pocket output changed')
        if any(row[k] != job[k] for k in ['text', 'speaker_id', 'split', 'reference_sha256']):
            raise ValueError('Retained Pocket output/target mismatch')
    torch.set_num_threads(2)
    save(run_path, run)

    def publish():
        clips = [{k: row[k] for k in ['clip_id', 'system', 'speaker_id', 'text', 'split', 'audio', 'sha256']} |
                 {'sentence_id': row['id'], 'automatic_leakage_label': None} for row in existing.values()]
        save(pack / 'selection.json', {'schema_version': 1, 'experiment_id': plan['experiment_id'], 'jobs': jobs,
                                      'clips': clips, 'plan_sha256': sha(plan_path), 'word_groups': word_groups})

    try:
        for system, checkpoint in weights.items():
            if all((system, j['id']) in existing for j in jobs):
                continue
            local = copy.deepcopy(base)
            local['weights_path'] = str(checkpoint); local['weights_path_without_voice_cloning'] = str(checkpoint)
            cfg = output / (system + '.yaml')
            cfg.write_text(yaml.safe_dump(local))
            model = TTSModel.load_model(config=cfg, temp=.3, sampler_decode_steps=1, eos_threshold=-4.).eval()
            model.requires_grad_(False)
            if model.device.type != 'cpu' or model.sample_rate != 24000:
                raise ValueError('Unexpected Pocket inference placement/rate')
            codec_digest = hashlib.sha256()
            for name, tensor in sorted(model.mimi.state_dict().items()):
                codec_digest.update(name.encode()); codec_digest.update(str(tuple(tensor.shape)).encode())
                codec_digest.update(tensor.detach().cpu().float().contiguous().numpy().tobytes())
            run['codec_state_sha256'][system] = codec_digest.hexdigest()
            states = {}
            for job in jobs:
                key = (system, job['id'])
                if key in existing:
                    continue
                if shutil.disk_usage(pack).free < 4 * 1024**3:
                    raise RuntimeError('Pocket experiment disk floor reached')
                path = output / (system + '_' + job['id'] + '.wav')
                if path.exists():
                    raise FileExistsError('Unreceipted audio requires inspection: ' + str(path))
                ref_key = job['reference_sha256']
                if ref_key not in states:
                    with torch.no_grad():
                        states[ref_key] = model.get_state_for_audio_prompt(Path(job['reference_audio']))
                torch.manual_seed(job['seed']); start = time.monotonic()
                with torch.no_grad():
                    wave = model.generate_audio(states[ref_key], job['text'], copy_state=True).detach().cpu().numpy()
                if wave.ndim == 2 and wave.shape[0] == 1:
                    wave = wave[0]
                if wave.ndim != 1 or not np.isfinite(wave).all() or not .2 <= len(wave) / 24000 <= 30:
                    raise ValueError('Invalid Pocket waveform; no silence trimming or substitute output')
                sf.write(path, wave, 24000, subtype='PCM_16')
                clip_id = hashlib.sha256((system + ':' + job['id']).encode()).hexdigest()[:24]
                row = {k: job[k] for k in ['id', 'text', 'speaker_id', 'split', 'reference_sha256']}
                row.update(system=system, clip_id=clip_id, audio=str(path.relative_to(pack)), status='ok_unreviewed',
                           sha256=sha(path), size_bytes=path.stat().st_size, seconds=len(wave) / 24000,
                           generation_wall_seconds=time.monotonic() - start,
                           clipping_fraction_before_pcm=float(np.mean(np.abs(wave) >= 1)))
                with receipt.open('a') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n'); handle.flush(); os.fsync(handle.fileno())
                existing[key] = row; run['completed'] = len(existing); save(run_path, run)
                LOG.info('Generated %s %s: %s/%s %.2fs wall %.2fs audio', system, job['id'], len(existing), len(jobs) * len(weights), row['generation_wall_seconds'], row['seconds'])
            run['systems_completed'].append(system); save(run_path, run); publish()
            del states, model
        run['status'] = 'complete_unreviewed_audio'; publish()
        run['same_codec_parameters_verified'] = len(set(run['codec_state_sha256'].values())) == 1 if len(weights) > 1 else None
    except Exception:
        run['status'] = 'failed_inspect_partial_outputs'
        raise
    finally:
        save(run_path, run)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Pocket crossed-word synthesis failed; inspect partial receipts before resuming')
        raise
