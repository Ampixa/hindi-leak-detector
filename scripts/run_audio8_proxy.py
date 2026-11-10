"""Matched Audio8 R3 development synthesis using the inspected native harness."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import time

LOG = logging.getLogger(__name__)
BENCH = Path('/home/ampixa/tts_bench_v0')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not (args.out / 'README.md').is_file():
        raise ValueError('Output retention README required')
    manifest = json.loads(args.manifest.read_text())
    jobs = manifest['jobs']
    if len({j['id'] for j in jobs}) != len(jobs) or not jobs:
        raise ValueError('Nonempty unique jobs required')
    for job in jobs:
        if job['split'] not in {'train', 'development'} or job['seed'] != 0:
            raise ValueError('Reserved benchmark input or unexpected seed')
        if digest(Path(job['reference']['audio'])) != job['reference']['sha256']:
            raise ValueError('Reference checksum mismatch')
    sys.path.insert(0, str(BENCH / 'code'))
    from common import configure_gpu, resources
    os.environ['HF_HOME'] = str(BENCH / 'upstream/cache/hf')
    torch = configure_gpu()
    import numpy as np
    import soundfile as sf
    from transformers import AutoModel, AutoProcessor
    from safetensors.torch import load_file
    sys.path.insert(0, str(BENCH / 'upstream/repos/audio8-nepali'))
    from audio8_tts_data import clean_text
    # Execute the unchanged native synthesis method with its own common module.
    original = BENCH / 'upstream/code'
    common_spec = importlib.util.spec_from_file_location('native_common', original / 'common.py')
    native_common = importlib.util.module_from_spec(common_spec)
    common_spec.loader.exec_module(native_common)
    spec = importlib.util.spec_from_file_location('native_audio8', original / 'synth_audio8.py')
    module = importlib.util.module_from_spec(spec)
    saved_common = sys.modules['common']
    sys.modules['common'] = native_common
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules['common'] = saved_common
    engine = object.__new__(module.Synthesizer)
    engine.torch, engine.device, engine.clean_text = torch, 'cuda', clean_text
    model_dir = BENCH / 'models/audio8'
    weights = BENCH / 'models/audio8_r3/model.safetensors'
    if digest(weights) != '0ab717efa1c7359f0006ad59d50855c0560ead2de78c361f82929c8dbd1adfef':
        raise ValueError('Audio8 R3 checkpoint differs from verified benchmark receipt')
    engine.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    engine.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,
        dtype=torch.float32, low_cpu_mem_usage=True, device_map={'': 'cuda'}, local_files_only=True).eval()
    state = load_file(str(weights), device='cuda')
    missing, unexpected = engine.model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    if any('codec' not in k for k in missing) or unexpected:
        raise ValueError(f'Checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    del state
    torch.cuda.empty_cache()
    codec_path = (model_dir / 'codec.pth').resolve()
    real_load = torch.load

    def codec_load(path, *positional, **keywords):
        if Path(path).resolve() == codec_path:
            keywords['map_location'] = 'cuda'
            with Path(path).open('rb') as source:
                if source.read(4) == b'PK\x03\x04':
                    keywords['mmap'] = True
        return real_load(path, *positional, **keywords)

    torch.load = codec_load
    try:
        with torch.device('cuda'):
            engine.model.load_codec(device='cuda')
    finally:
        torch.load = real_load
    run = {'creator': 'Codex GPT-6 /root', 'status': 'running', 'pid': os.getpid(),
           'checkpoint': str(weights), 'checkpoint_sha256': digest(weights),
           'codec_sha256': digest(codec_path), 'driver_sha256': digest(Path(__file__)),
           'manifest_sha256': digest(args.manifest),
           'native_method_sha256': digest(original / 'synth_audio8.py'),
           'model_code_sha256': digest(model_dir / 'modeling_arktts.py'),
           'codec_code_sha256': digest(model_dir / 'modeling_arktts_codec.py'),
           'settings': {'seed': 0, 'max_new_tokens': 1024, 'temperature': .8,
                        'top_p': .95, 'top_k': 50, 'do_sample': True, 'dtype': 'float32'},
           'waveform_policy': 'Native method; no reranking, silence trimming, or loudness normalization',
           'label_source': 'user source-level proxy assumption, not human pronunciation annotation'}
    run_path = args.out / 'run.json'
    receipt = args.out / 'items.jsonl'
    if receipt.exists() or run_path.exists():
        raise FileExistsError('Retained run exists; inspect before attempting a new run')
    run_path.write_text(json.dumps(run, indent=2) + '\n')
    original_write = sf.write

    def checked_write(path, values, *positional, **keywords):
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite generated waveform')
        return original_write(path, values, *positional, **keywords)

    sf.write = checked_write
    try:
        for index, job in enumerate(jobs):
            resources(True)
            if Path(job['id']).name != job['id']:
                raise ValueError('Unsafe job ID')
            output = args.out / (job['id'] + '.wav')
            if output.exists():
                raise FileExistsError(output)
            start = time.monotonic()
            details = engine.synthesize(job, output)
            info = sf.info(output)
            if info.frames == 0 or info.channels != 1:
                raise ValueError('Empty/non-mono synthesis')
            row = {'id': job['id'], 'speaker_id': job['speaker_id'], 'text': job['text'],
                   'split': job['split'], 'reference_sha256': job['reference']['sha256'],
                   'audio': output.name, 'sha256': digest(output), 'size_bytes': output.stat().st_size,
                   'seconds': info.duration, 'sample_rate': info.samplerate,
                   'status': 'ok_unreviewed', 'method': details, 'wall_seconds': time.monotonic() - start}
            with receipt.open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            LOG.info('Audio8 R3 %s/%s %s %.2fs', index + 1, len(jobs), job['id'], info.duration)
        run.update(status='complete_unreviewed_audio', completed=len(jobs))
    except Exception:
        run['status'] = 'failed_inspect_partial_outputs'
        raise
    finally:
        sf.write = original_write
        run_path.write_text(json.dumps(run, indent=2) + '\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Audio8 proxy synthesis failed; no fallback labels or substitute waveforms')
        raise
