"""Render actual v1/v2 results, tradeoffs and selective-decision coverage."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=Path, required=True)
    parser.add_argument('--agreement', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.revision / 'report.json').read_text())
    agreement = json.loads((args.agreement / 'report.json').read_text())
    predictions = json.loads((args.agreement / 'predictions.json').read_text())
    jobs = {j['id']: j for j in json.loads(args.selection.read_text())['jobs']}
    old = report['previously_inspected_benchmark']; fresh = report['fresh_native_negative_proxy_probe']
    selective = agreement['source_proxy_metrics']
    table = []
    for name in ['full', 'short_1p5']:
        a, b = fresh['v1'][name], fresh['v2'][name]
        table.append(f'<tr><td>{"Full native clips" if name == "full" else "1.5-second native windows"}</td><td>{a["high_score_count"]}/{a["n"]}</td><td>{b["high_score_count"]}/{b["n"]}</td></tr>')
    control_rows = []
    for system, values in agreement['benchmark_controls'].items():
        control_rows.append(f'<tr><td>{html.escape(system)}</td><td>{values["positive_flags"]}</td><td>{values["negative_decisions"]}</td><td>{values["uncertain"]}</td></tr>')
    pairs = {}
    for row in predictions:
        pairs.setdefault(row['sentence_id'], {})[row['system']] = row
    players = []
    for key in sorted(pairs)[:6]:
        cells = []
        for system in ['indicf5_stock', 'audio8_r3']:
            row = pairs[key][system]
            cells.append(f'<div><strong>{html.escape(system)}</strong><p>V1 {row["v1_score"]:.3f} · V2 {row["v2_score"]:.3f}<br>{html.escape(row["decision"])}</p><audio controls preload="none" src="/audio/{row["clip_id"]}"></audio></div>')
        players.append(f'<article><p lang="ne" class="sentence">{html.escape(jobs[key]["text"])}</p><div class="pair">{"".join(cells)}</div></article>')
    limits = ''.join(f'<li>{html.escape(item)}</li>' for item in report['limitations'])
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Hindi-leakage proxy · V2 comparison</title><style>
    :root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#f5f8fc;color:#182f4b;font:17px/1.65 "Avenir Next",Arial,sans-serif}}main{{max-width:1000px;margin:auto;padding:28px 24px 70px}}header,footer{{font-size:14px;color:#4b627a;border-top:1px solid #c3d2e4;padding-top:14px}}h1{{font-size:40px;line-height:1.15;letter-spacing:-1px;max-width:23ch}}h2{{font-size:25px;margin-top:40px}}p{{max-width:78ch}}.lead{{font-size:21px}}.notice{{border-left:4px solid #1656a4;padding:12px 20px;background:#e4eef9}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{text-align:left;padding:12px 8px;border-bottom:1px solid #c3d2e4}}.table{{overflow-x:auto}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:25px}}audio{{width:100%}}article{{padding:15px 0 28px;border-bottom:1px solid #c3d2e4}}.sentence{{font-size:25px}}code{{overflow-wrap:anywhere;font-size:14px}}a{{color:#1656a4}}footer{{margin-top:40px}}@media(max-width:650px){{main{{padding:22px 16px}}h1{{font-size:32px}}.pair{{grid-template-columns:1fr}}}}
    </style></head><body><main><header>ampixa / Hindi-leakage research · 7 November 2026</header><h1>Fewer flags on natural Nepali. A clear tradeoff on source classification.</h1><p class="lead">V2 adds natural Nepali negative proxies and short-window training. It was fitted on k2’s Apple M2 Ultra CPU; the L40S was not used for this revision.</p><h2>The fresh native-speech check</h2><p>Thirty-two unused source speaker groups were selected and locked before V2 fitting. Each recording has a full view and a 1.5-second view. These are 32 speakers, not 64 independent recordings.</p><div class="table"><table><thead><tr><th>High-score flags at fixed 0.5</th><th>V1</th><th>V2</th></tr></thead><tbody>{''.join(table)}</tbody></table></div><p>The full-clip V2 flag-rate 95% interval is {fresh['v2']['full']['wilson95_for_proxy_flag_rate'][0]:.1%}–{fresh['v2']['full']['wilson95_for_proxy_flag_rate'][1]:.1%}. This small weak-label probe does not establish a production false-positive rate.</p><h2>The old benchmark: the regression is visible</h2><p>On its 48 IndicF5/Audio8 source-labeled clips, V1 accuracy is <strong>{old['v1_source_proxy_metrics']['accuracy']:.1%}</strong>; V2 is <strong>{old['v2_source_proxy_metrics']['accuracy']:.1%}</strong>. V2 gives high scores to {old['v2_controls']['native_recording']['high_score_count']}/24 original native controls, versus {old['v1_controls']['native_recording']['high_score_count']}/24 for V1. Neither model is uniformly better.</p><p class="notice">These are proxy-source results. Both synthesizer families appear in fitting and test, and the old benchmark was already inspected during V1 work. No independently verified Hindi-pronunciation accuracy or schwa-site sensitivity is claimed.</p><h2>Disagreement is now an explicit result</h2><p>The conservative decision rule returns a positive or negative proxy label only when both fixed-0.5 heads agree; otherwise it returns <strong>uncertain</strong>. This rule was added after seeing V2’s tradeoff, so its results on these same probes are exploratory.</p><p>On the 48 source-labeled benchmark items, the rule covers <strong>{selective['covered']}/48 ({selective['coverage']:.1%})</strong> and abstains on {selective['abstained']}. Accuracy among covered items is {selective['covered_accuracy']:.1%}; correct decisions across <em>all</em> items are {selective['correct_fraction_of_all_items']:.1%}. Positive recall counting abstentions as missed detections is {selective['positive_recall_including_abstentions']:.1%}.</p><div class="table"><table><thead><tr><th>System · 24 clips each</th><th>Hindi-like proxy</th><th>Nepali-like proxy</th><th>Uncertain</th></tr></thead><tbody>{''.join(control_rows)}</tbody></table></div><p>Agreement between correlated classifiers is not a calibrated confidence guarantee. Coverage loss is shown rather than hidden in a higher conditional accuracy.</p><h2>What changed in fitting</h2><p>The original source assumptions remain: original IndicF5 is the Hindi-like proxy, Audio8 R3 is the preferred Nepali-like proxy. Natural corpus speech is now a separate weak negative anchor. No human labels were invented or overwritten.</p><p>The frozen encoder and inference recipe stay compatible. Training uses {report['training']['views']} views from {report['training']['parent_clips']} recordings across twelve source speaker groups. Development uses 24 recordings from four other groups. All views of a parent remain together. Twelve representation/regularization candidates were compared using training-speaker cross-validation only; selected {html.escape(report['selected_recipe']['representation'])}, L2 {report['selected_recipe']['l2']}. The convex fit converged and the threshold stayed at 0.5.</p><h2>Listen with both scores</h2>{''.join(players)}<h2>Limits and next validation</h2><ul>{limits}</ul><p>Next: collect complete short utterances across more voices and TTS lineages, then test pronunciation changes with the speaker/channel held fixed. Word-level schwa evidence and a trained prosody head remain separate unfinished components.</p><footer>Artifacts: <code>reports/source_proxy_v1/</code>, <code>reports/source_proxy_v2/</code>, <code>reports/source_proxy_v2_agreement/</code>. Source, model weights, solver checks, candidate scores and feature/selection hashes are retained. <a href="/">Optional blind validation form</a>.</footer></main></body></html>'''
    args.out.write_text(page)


if __name__ == '__main__':
    main()
