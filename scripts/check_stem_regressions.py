"""Development-only regression receipts for the new acoustic head."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from hindi_detector.acoustic_deletion import WordAcoustics
from hindi_detector.anchored_inference import AnchoredInference
from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.stem_deletion import StemVowelNetwork, SCHEMA, interpolate_stem
from scripts.calibrate_asr_gate import audio

ROOT = Path(__file__).resolve().parents[1]
CASES = [('real_yt_yt01_008', 'किनकि', None), ('real_yt_yt03_019', 'नेकपा', 4),
         ('real_yt_yt03_018', 'तलको', 0), ('real_yt_yt04_032', 'हुनलाई', 14),
         ('real_yt_yt07_053', 'चलाउनको', 14), ('real_yt_yt12_089', 'नगरौ', 7),
         ('real_yt_yt03_019', 'पिढको', 8)]


def main():
    out = ROOT / 'reports/stem_regressions_v83'
    model = ROOT / 'reports/stem_training_v83/model.pt'
    if out.exists():
        raise FileExistsError(out)
    state = torch.load(model, map_location='cpu', weights_only=False)
    if state['schema'] != SCHEMA:
        raise ValueError('Wrong candidate architecture')
    for name, digest in {**state['code_sha256'], **state['resource_files']}.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen inference input differs: ' + name)
    network = StemVowelNetwork(len(state['phone_vocabulary']))
    network.load_state_dict(state['weights'], strict=True)
    network.eval()
    frontend = WordAcoustics(state['phone_vocabulary'], state['phone_configurations'])
    source = ROOT / 'data/swar_speech_v18'
    selection = {r['id']: r for r in json.loads((source / 'selection.json').read_text())}
    assets = {(r['id'], r['kind']): r for r in json.loads((source / 'inventory.json').read_text())['assets']}
    write_group(out, {'inputs.json': {'candidate_sha256': sha(model), 'script_sha256': sha(Path(__file__)),
        'source_selection_sha256': sha(source / 'selection.json'), 'inventory_sha256': sha(source / 'inventory.json'),
        'feature_extraction_adapter': 'Existing AnchoredInference extracts production word bounds and24 query-only phone features/evidence. Its v71 head scores are discarded; only the v83 candidate is reported.',
        'adapter_code_sha256': sha(ROOT / 'hindi_detector/anchored_inference.py'),
        'adapter_model_sha256': sha(ROOT / 'reports/anchored_training_v71/model.pt'), 'cases': CASES,
        'limits': 'Development regressions, not independent test. Real counterpart speech is unlabeled. Pidh listener-similar, not a definitive negative. Haina quality/prosody outside deletion head.'}},
        'New-head known-occurrence regression check. inputs.json pinned candidate/source/adapter; items.jsonl source hash, observed bounds and per-site new logits/flags/abstentions; report.json known-confirmed retention counts. '
        'Codex GPT-6 /root, owner Ampixa/user, active consumer v83 development regression. Review2026-11-10; compact results retained, raw diagnostics disposable24h after closure. No new audio/encoders, NOT on HF; inherited source/MMS rights.',
        ['.venv-indicf5/bin/python -m scripts.check_stem_regressions'])
    runtime = AnchoredInference(ROOT, ROOT / 'reports/anchored_training_v71/model.pt')
    if any(runtime.state[k] != state[k] for k in ('phone_inventory', 'secondary_phone_inventory', 'aligner_inventory')):
        raise ValueError('Feature adapter and candidate encoder/aligner differ')
    if runtime.engine.phone_inventory != state['stem_recipe']['checkpoint_inventory']:
        raise ValueError('Early acoustic encoder differs from frozen candidate')
    stem_layers = torch.nn.Sequential(*list(runtime.engine.phone.wav2vec2.feature_extractor.conv_layers)[:5]).eval()
    records = []
    for identifier in dict.fromkeys(c[0] for c in CASES):
        for kind in ('swar', 'real'):
            asset = assets[identifier, kind]
            samples = audio(source, {'audio': asset['path'], 'sha256': asset['sha256']})
            evidence = runtime.analyze(samples, selection[identifier]['text'])
            with torch.inference_mode():
                x = runtime.engine.extractor(samples, sampling_rate=16000, return_tensors='pt').input_values.to(runtime.engine.device)
                raw = stem_layers(x[:, None])[0].cpu().numpy()
            for _, target, ordinal in (c for c in CASES if c[0] == identifier):
                matches = [(i, w) for i, w in enumerate(evidence['words']) if w['word'] == target and (ordinal is None or i == ordinal)]
                if len(matches) != 1:
                    raise ValueError(f'Confirmed word identity differs: {identifier}/{target}/{ordinal}')
                index, word = matches[0]
                record = {'id': identifier, 'kind': kind, 'word': target, 'ordinal': index,
                          'audio_sha256': asset['sha256'], 'sites': [], 'abstentions': list(word['unavailable_reasons'])}
                for site in word['sites']:
                    try:
                        mel, context, phone, diagnostic = frontend.extract(samples, (word['start'], word['end']), site['evidence'], site['native_phones'], site['native_phone_index'], site['features'])
                        stem = interpolate_stem(raw, (word['start'], word['end']), len(samples)).astype(np.float16).astype(np.float32)
                        mel = np.concatenate([stem, mel[64:66]], axis=0)
                        phone = np.clip((phone - state['scaler']['mean']) / state['scaler']['scale'], -8, 8).astype(np.float32)
                        with torch.inference_mode():
                            z = float(network(torch.from_numpy(mel[None]), torch.from_numpy(context[None]), torch.from_numpy(phone[None])).item())
                        record['sites'].append({'phone_index': site['native_phone_index'], 'logit': z,
                            'candidate': z >= state['logit_threshold'], 'query_geometry': diagnostic})
                    except ValueError as error:
                        record['abstentions'].append(str(error))
                record['flagged'] = any(s['candidate'] for s in record['sites'])
                record['tested'] = bool(record['sites'])
                records.append(record)
                with (out / 'items.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                print(json.dumps(record, ensure_ascii=False), flush=True)
    confirmed = [r for r in records if r['kind'] == 'swar' and r['word'] != 'पिढको']
    result = {'status': 'complete_development_regressions', 'known_confirmed_deletion_occurrences': len(confirmed),
              'flagged': sum(r['flagged'] for r in confirmed), 'tested': sum(r['tested'] for r in confirmed),
              'actual_hindi_error_accuracy': None, 'production_changed': False, 'model_sha256': sha(model), 'items_sha256': sha(out / 'items.jsonl')}
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Development regression failed explicitly; no live model change')
        raise
