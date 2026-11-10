"""Cache verified actual Swar examples and frame-level phonetic evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import traceback
import unicodedata

import numpy as np
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=ROOT/'data/swar_picks_v17')
    parser.add_argument('--out', type=Path, default=ROOT/'reports/swar_picks_features_v17')
    parser.add_argument('--resume-failed', action='store_true')
    args = parser.parse_args()
    plan = args.plan.resolve(strict=True)
    out = args.out
    if not plan.is_relative_to(ROOT/'data') or not out.resolve().is_relative_to(ROOT/'reports'):
        raise ValueError('Only verified task data/report groups are accepted')
    if out.exists() and not args.resume_failed:
        raise FileExistsError(out)
    if args.resume_failed:
        previous = json.loads((out/'status.json').read_text())
        if previous['phase'] != 'failed_explicitly' or 'Numeric token needs an explicit spoken expansion' not in previous.get('error', ''):
            raise ValueError('Only the inspected numeric-text failure may resume')
        write_group(out/'resume_numeric_v18', {'previous_failure.json': previous, 'source_snapshot.json': {
            'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
            'Resume inspected numeric-transcript failure. Ambiguous numeric-containing cases are explicitly excluded, never silently expanded or dropped. Existing complete feature receipts retained. Review2026-11-09, consumer actual-word corpus.',
            ['--resume-failed on the same frozen corpus; no model/encoder change'])
    inventory = json.loads((plan / 'inventory.json').read_text())
    selection = {r['id']: r for r in json.loads((plan / 'selection.json').read_text())}
    inputs = [r for r in inventory['assets'] if r['kind'] != 'prompt']
    if not inputs or any(r['id'] not in selection or r['kind'] not in {'real','swar','a450'} for r in inputs):
        raise ValueError('Input source inventory differs')
    durations = {}
    for row in inputs:
        info = sf.info(plan/row['path'])
        durations[row['id']] = max(durations.get(row['id'], 0), info.duration)
    if max(durations.values()) > 30:
        raise ValueError('Input exceeds verified full-utterance encoder length contract')
    if not args.resume_failed:
        write_group(out, {'inputs.json': inventory, 'source_snapshot.json': {
            'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha(Path(__file__)),
            'source_utf8': Path(__file__).read_text()}},
            'Actual user Swar picks. details/*.json contain measured word boundaries, phone sequences and relative prosody; '
            'features/*.npz contain full phone log-probability frames and structured word arrays. items.jsonl pins each source '
            'audio and feature SHA; runtime.json pins the acoustic encoders. These are measured observations, not automatic '
            'Hindi labels. User-identified haina/kinki examples are development regressions. No waveform copy or HF upload. '
            'Current consumer word-level leakage diagnostics; review2026-11-09 and remove redundant features after consumer closes.',
            ['.venv-indicf5/bin/python -m scripts.extract_swar_picks',
             'Verified source page and exact audio URLs in data/swar_picks_v17'])
        for name in ('details', 'features'):
            (out / name).mkdir()

    def status(phase, **extra):
        p = out / 'status.json'
        temp = p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(),
            'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2) + '\n')
        temp.replace(p)

    try:
        status('loading_verified_acoustic_models')
        engine = WordEvidenceEngine(ROOT, device='mps', pitch_method='dio', temporal_bins=20)
        (out / 'runtime.json').write_text(json.dumps({'alignment': engine.aligner_inventory,
            'phone': engine.phone_inventory, 'vocabulary': engine.vocab,
            'device_policy': 'Whole same-text real/Swar/Audio8 group CPU if any member exceeds20seconds, otherwise MPS',
            'group_durations_seconds': durations,
            'phone_frame_recipe': 'Original full utterance, no per-word re-encoding; raw normalized log posterior frames',
            'feature_source_sha256': sha(ROOT / 'hindi_detector/word_evidence.py')}, indent=2) + '\n')
        existing = [json.loads(line) for line in (out/'items.jsonl').read_text().splitlines()] if (out/'items.jsonl').exists() else []
        done = {(r['id'], r['kind']) for r in existing}
        excluded = 0
        for ordinal, row in enumerate(inputs):
            if (row['id'], row['kind']) in done:
                continue
            if any(unicodedata.category(c)[0] == 'N' for c in selection[row['id']]['text']):
                exclusion = {**row, 'reason': 'Numeric source text needs independently verified spoken expansion', 'status': 'excluded_input_not_pronunciation_label'}
                with (out/'input_exclusions.jsonl').open('a') as f: f.write(json.dumps(exclusion, ensure_ascii=False)+'\n')
                excluded += 1
                logging.warning('Explicit numeric-source exclusion %s/%s', row['id'], row['kind'])
                continue
            samples = audio(plan, {'audio': row['path'], 'sha256': row['sha256']})
            device = 'cpu' if durations[row['id']] > 20 else 'mps'
            if engine.device != device:
                engine.alignment.to(device); engine.phone.to(device); engine.device = device
                engine.torch.mps.empty_cache()
            text = selection[row['id']]['text']
            detail, arrays = engine.analyze(samples, text)
            with engine.torch.inference_mode():
                encoded = engine.phone(**{k: v.to(engine.device) for k, v in
                    engine.extractor(samples, sampling_rate=16000, return_tensors='pt').items()})
                values = encoded.logits[0].log_softmax(-1).cpu().numpy()
            if values.ndim != 2 or values.shape[1] != len(engine.vocab) or not np.isfinite(values).all():
                raise ValueError('Invalid full phone posterior frames')
            arrays['phone_log_probs'] = values.astype(np.float32)
            identity = row['id'] + '_' + row['kind']
            dp = out / 'details' / (identity + '.json')
            fp = out / 'features' / (identity + '.npz')
            detail.update(source_audio_sha256=row['sha256'], original_pick_id=row['id'], source_kind=row['kind'],
                          phone_frame_seconds=len(samples) / 16000 / len(values))
            dp.write_text(json.dumps(detail, ensure_ascii=False, allow_nan=False) + '\n')
            np.savez_compressed(fp, **arrays)
            receipt = {**row, 'text': text, 'status': 'ok', 'encoder_device': device, 'speaker_id': selection[row['id']]['voice'], 'detail': str(dp.relative_to(out)),
                       'detail_sha256': sha(dp), 'features': str(fp.relative_to(out)), 'features_sha256': sha(fp)}
            with (out / 'items.jsonl').open('a') as f:
                f.write(json.dumps(receipt, ensure_ascii=False) + '\n')
            status('extracting_actual_picks', completed=ordinal + 1, total=len(inputs))
            print(json.dumps({'id': row['id'], 'source': row['kind'], 'words': [
                {k: w[k] for k in ('word', 'start', 'end', 'phone_probe_sequence')}
                for w in detail['words'] if w['word'] in {'हैन', 'किनकि', 'किनभने', 'कस्तो', 'अहिले', 'व्यवस्था', 'कमसेकम'}]},
                ensure_ascii=False), flush=True)
        status('complete_actual_picks_features', completed=len(inputs)-excluded, excluded_inputs=excluded, planned=len(inputs))
    except Exception:
        status('failed_explicitly', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Actual Swar pick extraction failed explicitly')
        raise
