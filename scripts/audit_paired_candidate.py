"""Development-only operating thresholds and fixed-order physical word controls."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import binary_metrics
from hindi_detector.paired_detector import BRANCHES, load_feature, load_model, paired_features, score_arrays, sha
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio
from scripts.train_bulk_detector import dataset, flatten

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-group', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--kind', choices=['operating', 'physical'], required=True)
    parser.add_argument('--texts', type=int, default=20)
    args = parser.parse_args()
    if args.out.exists() or args.texts < 1:
        raise ValueError('Fresh output and positive text limit required')
    inputs = json.loads((args.model_group / 'inputs.json').read_text())
    directory = Path(inputs['features']); network, state = load_model(args.model_group / 'model.pt')
    frozen = inputs['frozen_receipts']; priors = json.loads((ROOT / 'data/pocket_bulk_plan_v11/g2p_priors.json').read_text())
    def groups(part):
        result = defaultdict(list)
        for r in frozen[part]:
            result[r['text_id']].append(r)
        return result
    write_group(args.out, {'inputs.json': {'model_group': str(args.model_group.resolve()), 'model_sha256': sha(args.model_group / 'model.pt'),
        'kind': args.kind, 'fixed_order_text_limit': args.texts},
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Frozen candidate diagnostic: report.json compact outcomes, predictions.json actual word outputs. Physical mode retains generated *.wav and per-wave SHA/bytes in predictions; these are known inserted-source controls, not natural Hindi-error gold. Operating mode selects conservative thresholds on development native focal controls only, then applies once to test.',
        ['.venv-indicf5/bin/python -m scripts.audit_paired_candidate --model-group GROUP --kind operating|physical --out NEW', 'Physical WAVs active for review; delete within24h after consumer finishes, retaining hashes/compact metrics. Original audio and model remain unchanged.'])
    output = []
    if args.kind == 'operating':
        dev = dataset(groups('development'), directory, state['projection'], {r['word'] for r in priors})
        dp, ds, labels = flatten(dev, focal_only=True)
        scores = score_arrays(network, state, dp, ds)
        thresholds = {}
        for i, branch in enumerate(BRANCHES):
            # Finite-sample empirical rate, not a population false-positive guarantee.
            native = np.sort(scores[labels == 0, i]); allowed = int(.1 * len(native))
            index = max(0, len(native) - allowed - 1)
            thresholds[branch] = float(np.nextafter(native[index], np.float32(np.inf)))
        test = dataset(groups('reserved_evaluation'), directory, state['projection'], {r['word'] for r in priors})
        tp, ts, labels_test = flatten(test, focal_only=True); test_scores = score_arrays(network, state, tp, ts)
        summary = {b: {'threshold': thresholds[b], 'development': binary_metrics(labels, scores[:, i], thresholds[b]),
                       'test': binary_metrics(labels_test, test_scores[:, i], thresholds[b])} for i, b in enumerate(BRANCHES)}
        report = {'summary': summary, 'selection': 'maximum approximately10% development-native focal flag rate; threshold independent of test outcomes',
                  'actual_hindi_accuracy': None, 'warning': 'Empirical source controls only; no real Hindi-error or population false-positive guarantee.'}
    else:
        engine = WordEvidenceEngine(ROOT, device='mps', pitch_method='dio', hidden_layers=(12, 24))
        chosen = list(groups('reserved_evaluation').items())[:args.texts]
        for tid, receipts in chosen:
            loaded = {(r['row']['system'], r['row']['speaker_id']): load_feature(directory, r) for r in receipts}
            voices = sorted({r['row']['speaker_id'] for r in receipts}); voice = voices[0]
            native = loaded['pocket_nepali_v1c', voice]; hindi = loaded['pocket_hindi_parent', voice]
            reference = loaded['pocket_nepali_v1c', voices[1]]
            focal = native['receipt']['focal_word_indices']; text = native['receipt']['row']['text']
            waves = {}
            for name, item in [('native', native), ('hindi', hindi)]:
                row = item['receipt']['row']; waves[name] = audio(Path(row['source_pack']), row)
            for condition, donor in [('sham_native_word_rejoin', native), ('physical_hindi_word_insert', hindi)]:
                samples = waves['native'].copy()
                for i in sorted(focal, reverse=True):
                    n, h = native['detail']['words'][i], donor['detail']['words'][i]
                    start, end = int(n['start'] * 16000), int(n['end'] * 16000)
                    part = waves['native' if condition.startswith('sham') else 'hindi'][int(h['start'] * 16000):int(h['end'] * 16000)].copy()
                    width = min(80, len(part) // 4, start, len(samples) - end)
                    if width < 2:
                        raise ValueError('Insufficient physical boundary support')
                    # Same5ms fade-to-zero seam treatment in sham and inserted controls.
                    samples[start-width:start] *= np.linspace(1, 0, width)
                    samples[end:end+width] *= np.linspace(0, 1, width)
                    part[:width] *= np.linspace(0, 1, width); part[-width:] *= np.linspace(1, 0, width)
                    samples = np.concatenate([samples[:start], part, samples[end:]])
                wav = args.out / (tid + '_' + condition + '.wav'); sf.write(wav, samples, 16000, subtype='PCM_16')
                # Decode actual retained PCM, not an unquantized substitute.
                retained, rate = sf.read(wav, dtype='float32')
                if rate != 16000:
                    raise ValueError('Retained physical control rate changed')
                detail, arrays = engine.analyze(retained, text)
                p, s = paired_features({'detail': detail, **arrays}, reference, state['projection'])
                scores = score_arrays(network, state, p, s)
                output.append({'text_id': tid, 'condition': condition, 'text': text, 'audio': wav.name,
                    'sha256': sha(wav), 'bytes': wav.stat().st_size, 'native_clip_id': native['receipt']['clip_id'],
                    'hindi_clip_id': hindi['receipt']['clip_id'], 'reference_clip_id': reference['receipt']['clip_id'],
                    'words': [{**{k: w[k] for k in ('word', 'start', 'end')}, 'focal': i in focal,
                        'label': int(i in focal and condition == 'physical_hindi_word_insert'),
                        'scores': dict(zip(BRANCHES, map(float, scores[i])))} for i, w in enumerate(detail['words'])]})
                LOG.info('Physical controls %s/%s', len(output), 2 * len(chosen))
        summary = {}
        for branch in BRANCHES:
            focal = [(w['label'], w['scores'][branch]) for r in output for w in r['words'] if w['focal']]
            unchanged = [w['scores'][branch] for r in output if r['condition'] == 'physical_hindi_word_insert' for w in r['words'] if not w['focal']]
            summary[branch] = {'known_inserted_source_focal': binary_metrics(np.array([v[0] for v in focal]), np.array([v[1] for v in focal]), state['thresholds'][branch]),
                'unchanged_neighbor_flag_rate': float(np.mean(np.array(unchanged) >= state['thresholds'][branch])) if unchanged else None}
        report = {'summary': summary, 'texts': len(chosen), 'waveforms': len(output), 'actual_hindi_accuracy': None,
            'limits': ['Fixed first test texts, one conditioning query per text; dependent words, no population interval.',
                       'Known Hindi source word inserted with original donor duration; source word is not necessarily a pronunciation error.',
                       'Identical fade seam algorithm on sham/native and Hindi inserts; voice mismatch and seams remain possible cues.',
                       'Relative whole-utterance prosody is recomputed, so untouched words can move in feature space.']}
    (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (args.out / 'predictions.json').write_text(json.dumps(output, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Candidate diagnostic failed explicitly')
        raise
