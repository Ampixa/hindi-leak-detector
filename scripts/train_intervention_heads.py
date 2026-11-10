"""Fit request-class components; freeze choices before reading held-out features."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import binary_metrics, predict_linear
from hindi_detector.interventions import LABEL_SOURCE, require_disjoint, require_partition

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/intervention_heads_v7'
LOG = logging.getLogger(__name__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_features(directory: Path, rows: list[dict], branch: str, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    vectors, raw = [], []
    for row in rows:
        path = (directory / row['features']).resolve(strict=True)
        if not path.is_relative_to(directory.resolve()) or sha(path) != row['feature_sha256']:
            raise ValueError('Intervention feature checksum/path mismatch')
        with np.load(path, allow_pickle=False) as cache:
            if cache['parent_audio_sha256'].item() != row['sha256']:
                raise ValueError('Intervention feature/audio parent mismatch')
            vector = cache[branch].copy(); baseline = float(cache['raw_ctc_alternative_minus_lexicon'])
        if vector.shape != (dimension,) or not np.isfinite(vector).all() or not np.isfinite(baseline):
            raise ValueError('Invalid intervention feature vector')
        vectors.append(vector); raw.append(baseline)
    return np.stack(vectors).astype(float), np.array(raw)


def fit(x: np.ndarray, y: np.ndarray, l2: float) -> dict:
    if x.ndim != 2 or y.shape != (len(x),) or set(y) != {0, 1} or not np.isfinite(x).all():
        raise ValueError('Need finite paired binary intervention training data')
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-4)
    normalized = (x - mean) / scale

    def loss_and_gradient(parameters):
        weight, bias = parameters[:-1], parameters[-1]
        logits = normalized @ weight + bias
        probabilities = np.exp(-np.logaddexp(0., -logits))
        residual = probabilities - y
        loss = np.mean(np.logaddexp(0., logits) - y * logits) + l2 * np.dot(weight, weight)
        gradient = np.r_[normalized.T @ residual / len(y) + 2 * l2 * weight, residual.mean()]
        return float(loss), gradient

    result = minimize(loss_and_gradient, np.zeros(x.shape[1] + 1), jac=True, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-7})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Intervention optimizer did not converge: ' + str(result.message))
    return {'weight': result.x[:-1].tolist(), 'bias': float(result.x[-1]), 'feature_mean': mean.tolist(),
            'feature_scale': scale.tolist(), 'l2': l2, 'optimizer': 'scipy L-BFGS-B',
            'training_loss': float(result.fun), 'optimizer_iterations': int(result.nit)}


def log_loss(labels, scores):
    clipped = np.clip(scores, 1e-12, 1 - 1e-12)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)))


def pair_ordering(rows, scores):
    pairs = defaultdict(dict)
    for row, score in zip(rows, scores, strict=True):
        pairs[(row['word'], row['speaker_id'], row['length_scale'])][row['condition']] = float(score)
    if any(set(p) != {'lexicon_path', 'site_count_alternative'} for p in pairs.values()):
        raise ValueError('Test lost a matched intervention condition')
    differences = [p['site_count_alternative'] - p['lexicon_path'] for p in pairs.values()]
    return {'pairs': len(differences), 'ordered_as_requested': sum(v > 0 for v in differences),
            'ties': sum(v == 0 for v in differences), 'median_difference': float(np.median(differences))}


def main() -> None:
    if not (OUT / 'README.md').is_file() or (OUT / 'selection').exists():
        raise ValueError('Need a documented fresh head experiment; never refit after seeing held-out scores')
    directory = ROOT / 'reports/intervention_features_v7'
    manifest_path = directory / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
    if manifest['failures'] or manifest['label_source'] != LABEL_SOURCE:
        raise ValueError('Need complete explicit intervention evidence, never human labels inferred from a renderer')
    records = manifest['items']
    parts = {name: [r for r in records if r['partition'] == name] for name in ('train', 'development', 'reserved_evaluation')}
    for name, rows in parts.items():
        require_partition(rows, name)
    require_disjoint(parts['train'], parts['development'])
    require_disjoint(parts['train'] + parts['development'], parts['reserved_evaluation'])
    train_y = np.array([r['intervention_class_id'] for r in parts['train']])
    dev_y = np.array([r['intervention_class_id'] for r in parts['development']])
    models, trials, choice = {}, {}, {}
    for branch in ('ctc_site', 'pooled_phone'):
        x, _ = load_features(directory, parts['train'], branch, manifest['feature_dimensions'][branch])
        dev, _ = load_features(directory, parts['development'], branch, manifest['feature_dimensions'][branch])
        fitted = []
        for l2 in (.001, .01, .1, 1.):
            model = fit(x, train_y, l2)
            loss = log_loss(dev_y, predict_linear(model, dev))
            fitted.append((loss, l2, model))
        loss, l2, model = min(fitted, key=lambda item: (item[0], item[1]))
        model.update(branch=branch, phone_encoder_revision=manifest['phone_encoder_revision'], label_source=LABEL_SOURCE,
                     score_meaning='Probability of the requested reduced-vowel input class under this renderer distribution; not actual Hindi-error probability.',
                     actual_hindi_accuracy=None,
                     training_words=sorted({r['word'] for r in parts['train']}), development_words=sorted({r['word'] for r in parts['development']}),
                     training_speaker_ids=sorted({r['speaker_id'] for r in parts['train']}), development_speaker_ids=sorted({r['speaker_id'] for r in parts['development']}))
        models[branch] = model
        trials[branch] = [{'l2': value, 'development_log_loss': score} for score, value, _ in fitted]
        choice[branch] = {'l2': l2, 'development_log_loss': loss}
    source_files = [Path(__file__), ROOT / 'hindi_detector/interventions.py', ROOT / 'hindi_detector/evaluation.py']
    snapshots = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in source_files]
    # Immutable choice/weight artifact exists before any reserved NPZ is opened.
    write_group(OUT / 'selection', {'models.json': models, 'selection.json': {'choices': choice, 'trials': trials,
                'feature_manifest_sha256': sha(manifest_path), 'test_features_loaded': False,
                'selection_rule': 'Lowest development log loss per prespecified feature branch; no test threshold tuning.'},
                'source_snapshot.json': snapshots},
                'Freeze intervention component weights/settings using only training and development features.',
                [str(manifest_path), '.venv-indicf5/bin/python -m scripts.train_intervention_heads'])
    test_rows = parts['reserved_evaluation']; test_y = np.array([r['intervention_class_id'] for r in test_rows])
    branches, predictions = {}, []
    raw_baseline = None
    for branch, model in models.items():
        x, raw = load_features(directory, test_rows, branch, manifest['feature_dimensions'][branch])
        scores = predict_linear(model, x)
        branches[branch] = {'request_class_metrics': binary_metrics(test_y, scores), 'paired_ordering': pair_ordering(test_rows, scores),
                            'site_kind_metrics': {kind: binary_metrics(test_y[np.array([r['site_kind'] == kind for r in test_rows])], scores[np.array([r['site_kind'] == kind for r in test_rows])]) for kind in ('internal', 'final')}}
        for row, score in zip(test_rows, scores, strict=True):
            predictions.append({'id': row['id'], 'word': row['word'], 'speaker_id': row['speaker_id'], 'length_scale': row['length_scale'],
                                'condition': row['condition'], 'intervention_class_id': row['intervention_class_id'],
                                'branch': branch, 'reduced_vowel_request_score': float(score), 'actual_hindi_error_label': None})
        raw_baseline = raw
    baseline = binary_metrics(test_y, (raw_baseline >= 0).astype(float))
    report = {'status': 'intervention_components_trained_evaluated_not_validated_hindi_detector',
              'training_clips': len(parts['train']), 'development_clips': len(parts['development']), 'test_clips': len(test_rows),
              'training_words': len({r['word'] for r in parts['train']}), 'development_words': len({r['word'] for r in parts['development']}),
              'test_words': len({r['word'] for r in test_rows}), 'test_voice_ids': sorted({r['speaker_id'] for r in test_rows}),
              'raw_ctc_zero_threshold_request_metrics': baseline, 'branches': branches,
              'test_feature_reads_after_selection_artifact': True, 'feature_manifest_sha256': sha(manifest_path),
              'actual_hindi_leakage_accuracy': None, 'actual_native_acceptability_accuracy': None,
              'overall_goal_achieved': False,
              'limits': ['All target classes describe known renderer phone requests; acoustic adherence and human acceptability are not gold.',
                         'Test jointly holds out12 lexemes and one voice ID; it cannot establish generalization across real people, vocoders or natural speech.',
                         'Two rates and two conditions per word are dependent observations;48 test clips are not48 independent native speakers.',
                         'Current human-bank words and original IndicF5/Pocket focus lexemes were excluded from the entire intervention corpus.',
                         'Source voice, condition, decoder timing and realized duration are not model inputs. No prosody error classifier is established.',
                         'These are auxiliary schwa-realization components; Pocket remains the preferred TTS reference and actual Hindi leakage remains unvalidated.']}
    write_group(OUT / 'evaluation', {'report.json': report, 'test_predictions.json': predictions},
                'Evaluate frozen schwa request-class components on jointly unseen lexical and renderer-voice partitions, retaining the exact limits.',
                ['../selection/models.json SHA ' + sha(OUT / 'selection/models.json'), 'Feature manifest SHA ' + sha(manifest_path)])
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Intervention component fitting/evaluation stopped; inspect frozen choice before retry')
        raise
