"""Readable result page generated only from actual fitted-model receipts."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    predictions = json.loads(args.predictions.read_text())
    jobs = {j['id']: j for j in json.loads(args.selection.read_text())['jobs']}
    metrics = report['held_out_source_proxy_metrics']
    native = report['reserved_controls_score_distributions']['native_recording']
    native_high = round(native['n'] * native['fraction_over_0_5'])
    rows = []
    for system, values in report['reserved_controls_score_distributions'].items():
        rows.append(f'<tr><td>{html.escape(system)}</td><td>{values["n"]}</td><td>{values["median_indicf5_like_score"]:.3f}</td><td>{values["fraction_over_0_5"]:.1%}</td></tr>')
    pairs = {}
    for row in predictions:
        pairs.setdefault(row['sentence_id'], {})[row['system']] = row
    players = []
    for sentence_id in sorted(pairs)[:6]:
        pair = pairs[sentence_id]
        cells = []
        for system in ['indicf5_stock', 'audio8_r3']:
            row = pair[system]
            cells.append(f'<div><strong>{html.escape(system)}</strong><p>IndicF5-like score {row["indicf5_like_score"]:.3f}</p><audio controls preload="none" src="/audio/{html.escape(row["clip_id"], quote=True)}"></audio></div>')
        players.append(f'<article><p class="sentence" lang="ne">{html.escape(jobs[sentence_id]["text"])}</p><div class="pair">{"".join(cells)}</div></article>')
    limits = ''.join(f'<li>{html.escape(value)}</li>' for value in report['limits'])
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IndicF5 / Audio8 proxy — actual model results</title><style>
    :root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#f5f8fc;color:#182f4b;font:17px/1.65 "Avenir Next",Arial,sans-serif}}main{{max-width:1000px;margin:auto;padding:28px 24px 70px}}header{{font-size:14px;color:#4b627a;border-bottom:1px solid #c3d2e4;padding-bottom:12px}}h1{{font-size:40px;line-height:1.15;letter-spacing:-1px;max-width:22ch}}h2{{font-size:25px;margin-top:40px}}p{{max-width:78ch}}.lead{{font-size:21px}}.notice{{border-left:4px solid #1656a4;padding:12px 20px;background:#e4eef9}}.stats{{display:flex;gap:45px;flex-wrap:wrap;margin:30px 0}}.number{{font-size:39px;display:block;font-weight:600}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{text-align:left;padding:12px 8px;border-bottom:1px solid #c3d2e4}}.table{{overflow-x:auto}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:25px}}audio{{width:100%}}article{{padding:15px 0 28px;border-bottom:1px solid #c3d2e4}}.sentence{{font-size:25px}}code{{overflow-wrap:anywhere;font-size:14px}}a{{color:#1656a4}}footer{{font-size:14px;border-top:1px solid #c3d2e4;padding-top:20px;margin-top:40px}}@media(max-width:650px){{main{{padding:22px 16px}}h1{{font-size:32px}}.pair{{grid-template-columns:1fr}}.stats{{gap:20px}}}}
    </style></head><body><main><header>ampixa / Hindi-leakage research · 6 November 2026</header><h1>The first source-proxy model is trained.</h1><p class="lead">Original IndicF5 supplies the Hindi-like proxy; Audio8 R3 supplies the Nepali-like proxy. This follows the user's explicit starting assumption and needs no human labels to fit.</p><div class="stats"><div><span class="number">{report['training_clips']}</span>matched training clips</div><div><span class="number">{report['development_clips']}</span>development clips</div><div><span class="number">{metrics['accuracy']:.1%}</span>held-out source-proxy accuracy</div></div><p class="notice">That accuracy measures agreement with the source labels. The fixed threshold also gives high scores to {native_high}/{native['n']} natural Nepali controls, so it is unreliable as an absolute Hindi-leakage flag. Verified Hindi-leakage accuracy remains unmeasured.</p><h2>What was trained</h2><p>A frozen XLSR speech encoder supplies mean and standard deviation features from layers 12 and 24. A small, regularized logistic head learns from {report['training_speaker_groups']} training speaker groups. Four other development speaker groups select the checkpoint. The reserved test has {report['test_clips']} source-labeled clips across {report['test_speaker_groups']} voice groups. The original benchmark remains excluded from fitting.</p><p>All new Audio8 development clips use exactly the IndicF5 reference WAV bytes, reference transcript and target sentence. Train-only normalization, one random seed, a fixed 0.5 reporting threshold and explicit overlap/hash checks keep the comparison reproducible. Cross-corpus real-person speaker identity is still unverified.</p><h2>Reserved test and diagnostic controls</h2><p>Score 0 means Audio8-like; score 1 means IndicF5-like under the source-label assumption. The score is uncalibrated. Native recordings and Audio8 R2 are controls, with no human leakage labels assigned.</p><div class="table"><table><thead><tr><th>System</th><th>Clips</th><th>Median score</th><th>Scores ≥ 0.5</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p>On matched reserved pairs, IndicF5 scores above Audio8 in {report['paired_ordering']['indicf5_above_audio8_fraction']:.1%} of {report['paired_ordering']['pairs']} pairs. The majority-class accuracy baseline is 50% because the proxy test is balanced.</p><h2>Listen to the actual scored pairs</h2><p>These examples show model identities. Use the separate blind form for any future independent pronunciation judgments.</p>{''.join(players)}<h2>What this establishes</h2><ul>{limits}</ul><p>The next scientific test is whether the score follows pronunciation changes when speaker, recording channel and synthesizer are controlled. Reviewed schwa spans, or controlled same-model pronunciation variants, can establish that relationship. Human review is a validation step; it was not a prerequisite for this first fit.</p><footer>Actual weights: <code>reports/source_proxy_v1/model.json</code>. Counts, metrics and predictions are retained alongside the model with hashes and provenance. Source: <code>scripts/train_source_proxy.py</code>. No human annotation was invented or overwritten. <a href="/">Open the optional blind evaluation form</a>.</footer></main></body></html>'''
    args.out.write_text(document)


if __name__ == '__main__':
    main()
