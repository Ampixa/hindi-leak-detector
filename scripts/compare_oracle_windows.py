"""Claude-requested train/dev-only localization versus resolution experiment."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.anchored_detector import load
from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_runtime import PhonePair
from hindi_detector.exact_phone_ctc import path_score
from hindi_detector.phone_lattice import VOWELS
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def metrics(records, mode, quantile=.99):
    if not records:
        return None
    short = np.array([r['S2'][mode] for r in records])
    removed = np.array([r['P1w'][mode] for r in records])
    original = np.array([r['S_world'][mode] for r in records])
    threshold = float(np.quantile(short, quantile, method='higher')) + .00001
    paired = lambda a, b: float(np.mean((a > b).astype(float) + .5 * (a == b)))
    return {'sources': len(records), 'threshold': threshold, 'P1w_recall': float(np.mean(removed >= threshold)),
            'S2_flag_rate': float(np.mean(short >= threshold)), 'paired_S2_P1w_concordance': paired(removed, short),
            'paired_original_P1w_concordance': paired(removed, original)}


def summarize(records):
    result = {m: {str(q): metrics(records, m, q) for q in (.95, .99)} for m in ('anchor', 'oracle')}
    strata = {}
    for lo, hi, name in [(40, 60, '40_to_60ms'), (60, 90, '60_to_90ms'), (90, 151, '90ms_and_up')]:
        selected = [r for r in records if lo <= r['retained_ms'] < hi]
        strata[name] = {'sources': len(selected), 'groups': len({r['phone_collision_group'] for r in selected}),
                        **{m: metrics(selected, m) for m in ('anchor', 'oracle')}}
    groups = sorted({r['phone_collision_group'] for r in records})
    rng = np.random.default_rng(760)
    estimates = []
    for _ in range(500):
        selected = [r for g in rng.choice(groups, len(groups), replace=True) for r in records if r['phone_collision_group'] == g]
        estimates.append(metrics(selected, 'oracle')['P1w_recall'])
    q90 = np.quantile([r['S2']['anchor'] for r in records], .9, method='higher')
    hard = [r for r in records if r['S2']['anchor'] >= q90]
    misses = float(np.mean([r['S2']['retained_interval_coverage'] < .5 for r in hard]))
    return {**result, 'duration_strata': strata,
            'oracle_recall_95_cluster_percentile_interval': np.quantile(estimates, [.025, .975]).tolist(),
            'bootstrap': {'unit': 'phonetic-collision group, voices kept together', 'seed': 760, 'replicates': 500},
            'top_decile_anchor_S2_sources': len(hard), 'top_decile_under_half_retained_coverage': misses}


def main():
    out = ROOT / 'reports/oracle_window_experiment_v76'
    if out.exists():
        raise FileExistsError(out)
    _, state = load(ROOT / 'reports/anchored_training_v71/model.pt')
    write_group(out, {'protocol.json': {
        'source_partitions': ['train', 'development'], 'families': ['S2', 'P1w', 'S_world'],
        'script_sha256': sha(Path(__file__)),
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for part in ('train', 'development') for p in
            (ROOT / ('reports/native_main_controls_' + part + '_v71/items.jsonl'),
             ROOT / ('reports/native_main_content_' + part + '_v71/result.json'))},
        'statistic': 'Mean over two probes of exact blank-minus-any-vowel CTC per frame, clipped[-10,10] per probe.',
        'anchor_window': 'Existing production-realigned deletion-anchor gap, frozen24feature columns0/12.',
        'oracle_window': 'Same S2 retained duration for all matched families. S2 centered on known retained region; P1w on actual splice point; S_world on original focal region center. Diagnostic only; never inference input.',
        'threshold': 'Within each diagnostic partition, S2 quantile95/99 with higher order statistic plus1e-5; not a deployment threshold or new classifier.',
        'strata_ms': [[40, 60], [60, 90], [90, 151]], 'bootstrap': '500 draws of phonetic-collision groups, recompute99th-percentile diagnostic threshold each draw; seed760.',
        'go_on_anchor_redesign': 'Both train and dev:oracle recall>=.80, cluster lower bound>=.65, oracle-minus-anchor recall>=.25, >=.60 of top-decile anchor S2 examples cover<half the retained interval.',
        'resolution_no_go': 'Oracle recall<.60 in either partition, or shortest-bin paired concordance<.80 while longest>=.95 (interpret bins only with>=20sources,>=10groups).',
        'positive_control_warning': 'Matched P1w versus original oracle concordance<.95 requires checking positives and representation; it does not prove phonetic labels or a generator defect.',
        'scope': 'No held-out test reads, fitting, threshold retuning or live changes; this experiment chooses measurement work only.'}},
        'Train/dev-only oracle versus inferred gap experiment requested by actual Claude post-training review. protocol.json predeclares diagnostic windows/statistic/gates; items.jsonl matched source geometry and per-family/probe evidence; '
        'PART.report.json each diagnostic partition; report.json combined decision; progress.json atomic progress. Owner Ampixa/user; review2026-11-10. Active consumer choose anchor redesign or encoder-resolution work before more teacher generation. '
        'No WAV/encoder/checkpoint copies or HF upload, inherited research rights. Physical labels are interventions, not human phonetic truth; known edit geometry is unusable during real inference. Retain compact result while research consumes it; diagnostics expire24h after consumer closes.',
        ['.venv-indicf5/bin/python -m scripts.compare_oracle_windows'])
    pair = PhonePair(ROOT, state)
    summaries = {}
    for partition in ('train', 'development'):
        directory = ROOT / ('reports/native_main_controls_' + partition + '_v71')
        accepted = set(json.loads((ROOT / ('reports/native_main_content_' + partition + '_v71/result.json')).read_text())['accepted_ids'])
        rows = read_rows(directory / 'items.jsonl')
        if any(r['split'] != partition for r in rows):
            raise ValueError('Diagnostic source partition differs')
        lookup = {(r['pair_id'], r['family']): r for r in rows if r['status'] == 'measured' and r['job_id'] in accepted}
        shorts = sorted([r for r in lookup.values() if r['family'] == 'S2'], key=lambda r: r['pair_id'])
        records = []
        for index, short in enumerate(shorts):
            record = {k: short[k] for k in ('pair_id', 'word', 'voice', 'phone_collision_group')}
            record.update(partition=partition, retained_ms=short['edit']['replacement_samples'] / 16)
            for family in ('S2', 'P1w', 'S_world'):
                r = lookup[short['pair_id'], family]
                wave = audio(directory, {'audio': r['audio'], 'sha256': r['audio_sha256']})
                duration = short['edit']['replacement_samples'] / 16000
                center = (r['edit']['start_sample'] + r['edit']['replacement_samples'] / 2) / 16000
                start, end = center - duration / 2, center + duration / 2
                if not 0 <= start < end <= len(wave) / 16000:
                    raise ValueError('Oracle diagnostic interval is outside waveform')
                scores, coverage, probes = [], [], []
                for probe, lp in enumerate(pair.encode(wave)):
                    dt = len(wave) / 16000 / len(lp)
                    a, b = int(np.floor(start / dt)), int(np.ceil(end / dt))
                    vowels = frozenset(pair.vocabularies[probe][t] for t in VOWELS if t in pair.vocabularies[probe])
                    frames = lp[a:b]
                    native, blank = path_score(frames, [vowels]), path_score(frames, [])
                    margin = float(np.clip((blank - native) / len(frames), -10, 10))
                    scores.append(margin)
                    covers = [max(0., min(end, w['gap_end_frame'] * dt) - max(start, w['gap_start_frame'] * dt)) / duration for w in r['evidence'][probe]['windows']]
                    coverage.append(float(np.mean(covers)))
                    probes.append({'probe': probe, 'oracle_margin': margin, 'oracle_frames': [a, b], 'frame_seconds': dt, 'anchor_coverage': covers})
                record[family] = {'anchor': (r['features'][0] + r['features'][12]) / 2,
                    'oracle': float(np.mean(scores)), 'retained_interval_coverage': float(np.mean(coverage)),
                    'oracle_interval': [start, end], 'audio_sha256': r['audio_sha256'], 'probes': probes}
            records.append(record)
            with (out / 'items.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            progress = out / 'progress.partial'
            progress.write_text(json.dumps({'partition': partition, 'sources_done': index + 1, 'sources_total': len(shorts)}) + '\n')
            progress.replace(out / 'progress.json')
        summaries[partition] = summarize(records)
        (out / (partition + '.report.json')).write_text(json.dumps(summaries[partition], indent=2) + '\n')
        print(partition, json.dumps(summaries[partition]), flush=True)
    anchor_go = all(s['oracle']['0.99']['P1w_recall'] >= .8 and s['oracle_recall_95_cluster_percentile_interval'][0] >= .65 and
                    s['oracle']['0.99']['P1w_recall'] - s['anchor']['0.99']['P1w_recall'] >= .25 and
                    s['top_decile_under_half_retained_coverage'] >= .6 for s in summaries.values())
    no_go = any(s['oracle']['0.99']['P1w_recall'] < .6 for s in summaries.values())
    result = {'status': 'complete_train_development_oracle_experiment', 'summaries': summaries,
              'anchor_redesign_go': anchor_go, 'oracle_overall_recall_below_60_percent': no_go,
              'decision': 'Anchor redesign meets predeclared diagnostic gate' if anchor_go else 'No anchor-only redesign justified; inspect encoder sensitivity, physical intervention validity and duration strata before another fit.',
              'test_read': False, 'model_changed': False, 'actual_hindi_error_accuracy': None,
              'items_sha256': sha(out / 'items.jsonl')}
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'summaries'}), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Train/dev oracle experiment failed explicitly; no model change')
        raise
