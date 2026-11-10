"""Test a same-word reference-conditioned direction on train/dev only."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.local_features import PROSODY_NAMES
from hindi_detector.prosody_direction import direction_features, PITCH_COLUMNS, FEATURE_NAMES
from scripts.build_prosody_interventions import detail
from scripts.extract_corrected_training import read_rows
from scripts.train_prosody_transfer import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / 'reports/prosody_interventions_v49'
    original = ROOT / 'reports/structured_evidence_v15'
    out = ROOT / 'reports/prosody_direction_development_v50'
    if out.exists():
        raise FileExistsError(out)
    plan = {row['id']: row for row in json.loads((ROOT / 'data/prosody_transfer_v49/transfer_plan.json').read_text())}
    # v49 reserved scores were already opened. They cannot be a fresh holdout
    # for this new formulation, so this fitting script never loads their arrays.
    rows = [row for row in read_rows(directory / 'rows.jsonl') if row['status'] == 'ok_pitch_interventions'
            and row['split'] in {'train', 'development'}]
    hindi = {(row['text_id'], row['row']['speaker_id']): row for row in read_rows(original / 'items.jsonl')
             if row['kind'] == 'original' and row['status'] == 'ok_structured_features'
             and row['row']['system'] == 'indicf5_stock' and row['row']['content_quality']['decision'] == 'pass'}
    raw = []
    for row in rows:
        selected = plan[row['id']]
        hb = hindi[selected['text_id'], selected['conditioning_groups']['reference']]
        if hb['row']['speaker_id'] == selected['conditioning_groups']['carrier']:
            raise ValueError('Hindi template must be an independently conditioned render, not the physical target donor')
        hd = detail(original, hb)
        ordinal = row['word_ordinal']
        if hd['words'][ordinal]['word'] != row['word']:
            raise ValueError('Hindi same-word template identity differs')
        hv = np.array([hd['words'][ordinal]['prosody'][name] for name in PROSODY_NAMES])[PITCH_COLUMNS]
        for output in row['outputs']:
            if output['condition'] not in {'sham', 'native_transfer', 'hindi_transfer'}:
                continue
            path = directory / output['features']
            if sha(path) != output['features_sha256']:
                raise ValueError('Physical measured input changed')
            with np.load(path, allow_pickle=False) as arrays:
                q = arrays['query_prosody'][ordinal, PITCH_COLUMNS]
                n = arrays['reference_prosody'][ordinal, PITCH_COLUMNS]
            raw.append({'id': row['id'], 'word': row['word'], 'split': row['split'], 'condition': output['condition'],
                        'label': output['source_style_label'], 'q': q, 'n': n, 'h': hv, 'independent_hindi_template_sha256': hb['row']['sha256']})
    train_raw = [row for row in raw if row['split'] == 'train']
    scale = np.maximum(np.std(np.stack([row['q'] - row['n'] for row in train_raw]), axis=0), 1.)
    values = np.stack([direction_features(row['q'], row['n'], row['h'], scale) for row in raw])
    labels = np.array([row['label'] for row in raw], np.float32)
    train = np.array([row['split'] == 'train' for row in raw])
    mean, standard = values[train].mean(0), np.maximum(values[train].std(0), .2)
    x = torch.from_numpy(np.clip((values - mean) / standard, -8, 8).astype(np.float32))
    y = torch.from_numpy(labels)
    torch.manual_seed(20261110)
    torch.set_num_threads(2)
    network = nn.Linear(len(FEATURE_NAMES), 1)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.01, weight_decay=.2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.))
    best, history = None, []
    for epoch in range(1, 501):
        optimizer.zero_grad()
        loss = criterion(network(x[train]).squeeze(-1), y[train])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite reference-conditioned fitting loss')
        loss.backward()
        optimizer.step()
        with torch.inference_mode():
            dev_loss = float(criterion(network(x[~train]).squeeze(-1), y[~train]))
        history.append({'epoch': epoch, 'train_loss': float(loss.detach()), 'dev_loss': dev_loss})
        if best is None or dev_loss < best['dev_loss']:
            best = {'epoch': epoch, 'dev_loss': dev_loss, 'state_dict': {k: v.detach().clone() for k, v in network.state_dict().items()}}
    network.load_state_dict(best['state_dict'], strict=True)
    with torch.inference_mode():
        scores = network(x).squeeze(-1).sigmoid().numpy()
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite conditioned pitch scores')
    high = max(.5, float(scores[(~train) & (labels == 0)].max()) + .00001)
    report = {'status': 'complete_train_dev_only_direction_baseline', 'best_epoch': best['epoch'], 'high': high,
        'train': evaluate(scores[train], labels[train], high), 'development': evaluate(scores[~train], labels[~train], high),
        'development_at_point5': evaluate(scores[~train], labels[~train], .5),
        'test_policy': 'No v49 reserved arrays loaded; new formulation requires newly generated untouched word families before reporting a held-out result.',
        'actual_hindi_error_accuracy': None, 'promotion': 'not_promoted_reference_conditioning_is_not_error_gold'}
    state = {'schema': 'same_word_pitch_direction_v50', 'state_dict': best['state_dict'], 'pitch_scale': scale,
             'mean': mean, 'scale': standard, 'high': high, 'feature_names': FEATURE_NAMES, 'pitch_columns': PITCH_COLUMNS,
             'epoch': best['epoch'], 'actual_hindi_error_accuracy': None}
    write_group(out, {'report.json': report, 'history.json': history,
        'scores.json': [{**{k: v for k, v in row.items() if k not in {'q', 'n', 'h'}},
                         'features': features.tolist(), 'source_style_score': float(score)} for row, features, score in zip(raw, values, scores, strict=True)],
        'inputs.json': {'physical_rows_sha256': sha(directory / 'rows.jsonl'), 'original_items_sha256': sha(original / 'items.jsonl')},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
                               for p in (Path(__file__), ROOT / 'hindi_detector/prosody_direction.py')]},
        'Development-only seven-input reference-conditioned pitch-style baseline. Contrast '
        'query to same-word NativeB and independently rendered HindiB, whereas physical '
        'positive used HindiA donor on NativeA carrier. This tests whether word-conditioned '
        'direction helps the failed global style head. model.pt selected state/training-only '
        'normalization/high threshold; model_inventory.json SHA/bytes. scores.json train/dev '
        'only, no v49 reserved arrays loaded because that experiment was already inspected. '
        'Not a real Hindi-error model and not live. Current consumer decide whether fresh '
        'independent-word synthesis is justified; review2026-11-10. Keep compact results and '
        'selected checkpoint only while active; no rawaudio copies/HF upload; inherited rights.',
        ['.venv-indicf5/bin/python -m scripts.fit_prosody_direction'])
    torch.save(state, out / 'model.pt')
    (out / 'model_inventory.json').write_text(json.dumps({'sha256': sha(out / 'model.pt'), 'bytes': (out / 'model.pt').stat().st_size}, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Reference-conditioned development experiment failed explicitly')
        raise
