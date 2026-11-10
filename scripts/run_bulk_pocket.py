"""Interleave paired Pocket checkpoints with persistent models and sharded texts."""
from __future__ import annotations

import argparse
from collections import Counter
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
import traceback

import numpy as np
from pocket_tts import TTSModel
import soundfile as sf
import torch
import yaml

from scripts.run_pocket_matrix import ROOT, SOURCE, SOURCE_REVISION, save, sha

LOG = logging.getLogger(__name__)


class WaveformQualityError(ValueError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-pack', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--limit-jobs', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--generation-cutoff-utc', default='2026-11-08T00:30:00+00:00')
    args = parser.parse_args()
    cutoff = datetime.fromisoformat(args.generation_cutoff_utc)
    if cutoff.tzinfo is None:
        raise ValueError('Generation cutoff needs an explicit timezone')
    if not 0 <= args.shard < args.shards <= 8 or args.limit_jobs is not None and args.limit_jobs < 1:
        raise ValueError('Need explicit valid shard/positive optional smoke limit')
    if not (args.out / 'README.md').is_file() or not args.resume and (args.out / 'run.json').exists():
        raise ValueError('Need documented fresh worker output; resume must be explicit')
    selection_path = args.source_pack / 'selection.json'; selected = json.loads(selection_path.read_text())
    if selected['experiment_id'] != 'pocket_bulk_overnight_v11':
        raise ValueError('Only the actual frozen overnight source plan is accepted')
    jobs = [j for j in selected['jobs'] if int(hashlib.sha256(j['text_id'].encode()).hexdigest()[:8], 16) % args.shards == args.shard]
    if args.limit_jobs is not None:
        jobs = [j for j in jobs if j['split'] == 'train'][:args.limit_jobs]
    if not jobs or len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Need unique nonempty actual jobs')
    if subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE_REVISION:
        raise ValueError('Pocket source revision changed')
    subprocess.run(['git', '-C', str(SOURCE), 'diff', '--quiet', 'HEAD'], check=True)
    pilot = json.loads((ROOT / 'reports/pocket_pilot_v1/manifest.json').read_text())
    checkpoints = {'pocket_nepali_v1c': Path(pilot['finetune']['path']), 'pocket_hindi_parent': ROOT / 'models/pocket_hindi_parent/model.safetensors'}
    weights = {'pocket_nepali_v1c': pilot['finetune']['sha256'], 'pocket_hindi_parent': pilot['parent']['sha256']}
    for system, path in checkpoints.items():
        if sha(path) != weights[system]:
            raise ValueError('Actual Pocket weight identity changed')
    tokenizer = ROOT / 'models/pocket_hindi_parent/tokenizer.model'
    token_inventory = json.loads((tokenizer.parent / 'inventory.json').read_text())
    expected = next(r['sha256'] for r in token_inventory if r['path'] == 'tokenizer.model')
    if sha(tokenizer) != expected:
        raise ValueError('Actual Pocket tokenizer changed')
    references = {j['reference_audio']: j['reference_sha256'] for j in jobs}
    for path, digest in references.items():
        if sha(Path(path)) != digest:
            raise ValueError('Actual voice reference changed')
    base = yaml.safe_load((ROOT / 'reports/pocket_pilot_v1/nepali_v1c.yaml').read_text())
    if base['flow_lm']['transformer']['num_layers'] != 6 or base['mimi']['sample_rate'] != 24000:
        raise ValueError('Pocket inference architecture changed')
    base['flow_lm']['lookup_table']['tokenizer_path'] = str(tokenizer)
    plan = {'experiment_id': 'pocket_bulk_overnight_v11', 'selection_sha256': sha(selection_path),
            'source_revision': SOURCE_REVISION, 'weights_sha256': weights, 'shard': args.shard, 'shards': args.shards,
            'limit_jobs': args.limit_jobs, 'job_ids': [j['id'] for j in jobs], 'driver_sha256': sha(Path(__file__)),
            'generation_cutoff_utc': cutoff.isoformat(),
            'settings': {'device': 'cpu', 'threads': 2, 'seed': 0, 'temperature': .3, 'sampler_decode_steps': 1, 'eos_threshold': -4.}}
    plan_path = args.out / 'plan.json'
    if args.resume:
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Resume source/code/settings/job identity changed')
    else:
        save(plan_path, plan)
        files = [Path(__file__), ROOT / 'scripts/run_pocket_matrix.py', SOURCE / 'pocket_tts/models/tts_model.py']
        save(args.out / 'source_snapshot.json', [{'path': str(p), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in files])
    receipt = args.out / 'items.jsonl'; records = [json.loads(s) for s in receipt.read_text().splitlines()] if args.resume else []
    existing = {(r['system'], r['id']): r for r in records}; job_lookup = {j['id']: j for j in jobs}
    if len(existing) != len(records):
        raise ValueError('Duplicate bulk synthesis receipts')
    for (system, identifier), row in existing.items():
        job = job_lookup[identifier]
        if system not in checkpoints or any(row[k] != job[k] for k in ('text', 'speaker_id', 'split', 'reference_sha256')):
            raise ValueError('Resume receipt differs from frozen source job')
        if row['status'] == 'ok_unreviewed' and sha(args.out / row['audio']) != row['sha256']:
            raise ValueError('Retained synthesized waveform changed')
        if row['status'] not in {'ok_unreviewed', 'failed_waveform_quality'}:
            raise ValueError('Unexpected incomplete/error receipt requires inspection')
    output = args.out / 'pocket'; output.mkdir(exist_ok=args.resume)
    torch.set_num_threads(2); models = {}; states = {}; codec = {}
    start_run = time.monotonic()

    def status(phase: str, **fields) -> None:
        save(args.out / 'run.json', {'status': phase, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(),
             'planned_clips': len(jobs) * 2, 'processed_clips': len(existing), 'counts': dict(Counter(r['status'] for r in existing.values())),
             'wall_seconds_this_process': time.monotonic() - start_run, 'weights_sha256': weights, 'codec_sha256': codec,
             'plan_sha256': sha(plan_path), **fields})

    def publish() -> None:
        clips = [{k: r[k] for k in ('clip_id', 'system', 'speaker_id', 'text', 'split', 'audio', 'sha256')} |
                 {'sentence_id': r['id'], 'automatic_leakage_label': None} for r in existing.values() if r['status'] == 'ok_unreviewed']
        save(args.out / 'selection.json', {'schema_version': 1, 'experiment_id': 'pocket_bulk_overnight_v11', 'jobs': jobs,
                                         'clips': clips, 'plan_sha256': sha(plan_path)})

    status('loading_verified_checkpoint_pair')
    try:
        for system, path in checkpoints.items():
            local = copy.deepcopy(base); local['weights_path'] = str(path); local['weights_path_without_voice_cloning'] = str(path)
            cfg = output / (system + '.yaml'); cfg.write_text(yaml.safe_dump(local))
            model = TTSModel.load_model(config=cfg, temp=.3, sampler_decode_steps=1, eos_threshold=-4.).eval()
            model.requires_grad_(False)
            if model.device.type != 'cpu' or model.sample_rate != 24000:
                raise ValueError('Unexpected Pocket inference placement/rate')
            digest = hashlib.sha256()
            for name, tensor in sorted(model.mimi.state_dict().items()):
                digest.update(name.encode()); digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().cpu().float().contiguous().numpy().tobytes())
            codec[system] = digest.hexdigest(); models[system] = model; states[system] = {}
        if len(set(codec.values())) != 1:
            raise ValueError('Pocket decoder parameters differ; same-architecture control invalid')
        for job in jobs:
            if datetime.now(timezone.utc) >= cutoff:
                publish(); status('deadline_cutoff_partial_generation')
                return
            if Path(job['id']).name != job['id'] or job['seed'] != 0:
                raise ValueError('Unexpected bulk job identity/seed')
            for system, model in models.items():
                key = (system, job['id'])
                if key in existing:
                    continue
                if shutil.disk_usage(args.out).free < 20 * 1024**3:
                    raise RuntimeError('Keep20GiB free on the data volume')
                status('generating_paired_pocket_audio', current_job=job['id'], current_system=system)
                path = output / (system + '_' + job['id'] + '.wav')
                if path.exists():
                    raise FileExistsError('Unreceipted waveform requires inspection')
                ref = job['reference_sha256']
                if ref not in states[system]:
                    with torch.no_grad():
                        states[system][ref] = model.get_state_for_audio_prompt(Path(job['reference_audio']))
                row = {k: job[k] for k in ('id', 'text', 'speaker_id', 'split', 'reference_sha256', 'text_id', 'text_kind', 'target_words')}
                row.update(system=system, clip_id=hashlib.sha256((system + ':' + job['id']).encode()).hexdigest()[:24], automatic_leakage_label=None)
                torch.manual_seed(0); started = time.monotonic()
                try:
                    with torch.no_grad():
                        wave = model.generate_audio(states[system][ref], job['text'], copy_state=True).detach().cpu().numpy()
                    if wave.ndim == 2 and wave.shape[0] == 1:
                        wave = wave[0]
                    if wave.ndim != 1 or not np.isfinite(wave).all() or not .2 <= len(wave) / 24000 <= 30 or np.max(np.abs(wave)) < 1e-5:
                        raise WaveformQualityError('Invalid/nonfinite/silent/too-short/over30s output; no substitute or trimming')
                    sf.write(path, wave, 24000, subtype='PCM_16')
                    row.update(status='ok_unreviewed', audio=str(path.relative_to(args.out)), sha256=sha(path),
                               bytes=path.stat().st_size, seconds=len(wave) / 24000,
                               clipping_fraction_before_pcm=float(np.mean(np.abs(wave) >= 1)))
                except WaveformQualityError:
                    row.update(status='failed_waveform_quality', error=traceback.format_exc())
                    LOG.exception('Explicit waveform-quality rejection for %s/%s', system, job['id'])
                row['generation_wall_seconds'] = time.monotonic() - started
                with receipt.open('a') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); handle.flush(); os.fsync(handle.fileno())
                existing[key] = row
                if len(existing) % 32 == 0:
                    publish(); LOG.info('Shard%s completed%s/%s clips', args.shard, len(existing), len(jobs) * 2)
        publish(); status('complete_paired_audio_with_explicit_quality_rejections')
    except Exception:
        status('failed_inspect_logs_before_resume'); LOG.exception('Bulk paired generation stopped; no silent retries')
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('pocket_tts').setLevel(logging.WARNING)
    main()
