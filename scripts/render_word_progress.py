"""Readable experiment status from measured artifacts; native JSON kept alongside."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import binary_metrics

ROOT = Path(__file__).resolve().parents[1]


def query_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['query_clip_id']].append(row)
    output = {}
    for branch in ['phonetics', 'prosody', 'combined']:
        y, p, native = [], [], []
        for group in groups.values():
            if len({r['system'] for r in group}) != 1:
                raise ValueError('Reference query identity disagreement')
            score = float(np.mean([r['bag_scores'][branch] for r in group]))
            system = group[0]['system']
            if system == 'native_recording':
                native.append(score)
            elif system in {'indicf5_stock', 'audio8_r3'}:
                y.append(int(system == 'indicf5_stock')); p.append(score)
        output[branch] = {'source_bag_metrics': binary_metrics(np.array(y), np.array(p)),
                          'native_high': sum(s >= .5 for s in native), 'native_count': len(native)}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    def read(name):
        return json.loads((ROOT / name).read_text())
    local = read('reports/wordlab_models_v3/report.json')
    reference = read('reports/reference_word_v3/report.json')
    query = query_summary(read('reports/reference_word_v3/word_predictions.json'))
    matrix = read('data/wordmatrix_v3/selection.json')
    f5 = read('data/wordmatrix_v3/indicf5/run.json')
    receipts = [json.loads(line) for line in (ROOT / 'data/wordmatrix_v3/indicf5/items.jsonl').read_text().splitlines()]
    if len({r['id'] for r in receipts}) != len(receipts) or any(r['status'] != 'ok_unreviewed' for r in receipts):
        raise ValueError('Matrix generation receipt integrity failure')
    voice = read('reports/voice_controls_v3/report.json')
    sites = read('reports/schwa_matrix_f5_v3/report.json')
    ranks = read('reports/schwa_matrix_reference_scores_v3/report.json')
    bank = read('reports/native_schwa_reference_v3/reference.json')
    checkpoint_probe = read('reports/wordlab_checkpoint_probe_v3/report.json')
    swaps = [c for c in voice['controls'] if c['control'] != 'identity' and c['quality']['approximate_preservation_gate_passed']]
    groups = {}
    for name in ['wordlab_training_v3', 'wordlab_test_v3', 'wordlab_pocket_probe_v3', 'wordlab_native_wordbank_v3', 'wordlab_matrix_f5_v3']:
        path = ROOT / 'reports' / name
        if (path / 'manifest.json').exists():
            m = json.loads((path / 'manifest.json').read_text())
            groups[name] = {k: m[k] for k in ['successes', 'failures']}
            groups[name]['word_rows'] = sum(row.get('words', 0) for row in m['items'] if row['status'] == 'ok_unreviewed_word_evidence')
        else:
            events = [json.loads(line) for line in (path / 'items.jsonl').read_text().splitlines()] if (path / 'items.jsonl').exists() else []
            groups[name] = {'status': 'running_partial', 'processed': len(events), 'successes': sum(r['status'] == 'ok_unreviewed_word_evidence' for r in events), 'failures': sum(r['status'] != 'ok_unreviewed_word_evidence' for r in events)}
    gpu_path = ROOT / 'reports/audio8_matrix_gpu_status.json'
    gpu = json.loads(gpu_path.read_text()) if gpu_path.exists() else {'status': 'not_observed_in_this_snapshot'}
    steps = [
        {'step': 1, 'task': 'Localize words and separate pronunciation from prosody', 'status': 'implemented',
         'detail': 'Pinned forced alignment, conditional phonetic posteriors, 29 relative pitch/energy/duration features, real timed playback.'},
        {'step': 2, 'task': 'Train independent and preferred-reading comparison branches', 'status': 'trained; accuracy inadequate',
         'detail': 'Six branches trained with speaker-group model selection. Positive utterances do not turn every word into a Hindi gold label.'},
        {'step': 3, 'task': 'Cross words, sentence context and voices', 'status': 'generation and extraction in progress',
         'detail': '1,024 planned clips: 512 per system, 24 held-out-by-split focus words, four authored contexts, 16 reference groups. Different-voice Audio8 references will supply non-self negatives.'},
        {'step': 4, 'task': 'Test actual schwa-site alternatives against native variation', 'status': 'native bank and IndicF5 temporal probe complete; coverage limited',
         'detail': '120 original native recordings, 20 candidate lexemes, six source speaker groups per lexeme. Thirteen words support native site references. Exact CTC site-count alternatives marginalize over three vowel qualities; unsupported phones are explicit exclusions.'},
        {'step': 5, 'task': 'Audit voice, decoder and checkpoint shortcuts', 'status': 'initial controls measured; unresolved',
         'detail': 'Identity resynthesis, two common-voice conversions, Pocket parent/fine-tune probes. Failed preservation controls cannot establish voice bias. Old whole-clip scores failed runtime compatibility and are excluded.'},
        {'step': 6, 'task': 'Fit the crossed-reference model and validate localized errors', 'status': 'next; requires completed crossed data',
         'detail': 'Use same-word preferred readings across different voices, keep text/voice splits fixed, report per-word and per-phenomenon performance when independent evidence exists. Human annotation is optional validation, never a prerequisite for source-proxy experiments.'}]
    result = {'creator': 'Codex GPT-6 /root', 'updated_utc': datetime.now(timezone.utc).isoformat(),
              'machine': 'k2: Apple M2 Ultra Mac Studio, 64 GB; large artifacts on T9. Audio8 generation: shared L40S with resource-aware queue.',
              'status': 'substantial_word_and_prosody_work_active_not_validated_detector', 'steps': steps,
              'word_evidence': groups, 'independent_branches': local['branches'],
              'reference_query_level_results': query, 'reference_pair_count': reference['test_pair_bags'],
              'reference_aggregation': 'Mean bag probability across preferred references for each unique query. Fixed postprocessing, no fitting or threshold tuning. Queries remain statistically grouped by text/speaker.',
              'matrix': {'planned': 2 * len(matrix['jobs']), 'indicf5_complete': len(receipts), 'indicf5_status': f5['status'], 'audio8_last_observation': gpu},
              'temporal_site_probe': {'indicf5': sites, 'native_supported_anchor_words': bank['supported_anchor_words'], 'reference_relative_coverage': ranks},
              'external_checkpoint_probe': checkpoint_probe,
              'voice_controls': {'total': len(voice['controls']), 'preservation_passed': voice['qualified_controls'],
                                 'qualified_voice_swaps': len(swaps), 'independent_originals_for_qualified_swaps': len({c['original']['clip_id'] for c in swaps}),
                                 'old_global_model_results': 'excluded: runtime compatibility failed'},
              'human_word_gold_used': 0, 'validated_word_hindi_accuracy': None,
              'scope': 'Hindi leakage into Nepali TTS. Nepal Bhasa/Newa remains a separate future task; the inspected Ampixa frontend is Nepali.',
              'limits': ['All reported classifier accuracy is weak source-bag separation on previously inspected exploratory material, not word-level Hindi accuracy.',
                         'Normalized pitch/loudness/global tempo tests do not establish invariance to a complete voice change.',
                         'Phone probe and lexicon candidates are not validated Nepali schwa gold.',
                         'MMS alignment is a CC BY-NC research dependency; a suitable replacement is needed for commercial deployment.']}
    write_group(args.out, {'progress.json': result}, 'Readable status and executable work plan for the active word/prosody detector.',
                ['Actual experiment manifests and source/word models under reports/',
                 'python -m scripts.render_word_progress --out NEW',
                 'Driver SHA ' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()])
    rows = []
    for branch in ['phonetics', 'prosody', 'combined']:
        old = local['branches'][branch]['previously_inspected_benchmark']['source_bag_metrics']
        new = query[branch]
        rows.append(f'<tr><th>{escape(branch)}</th><td>{old["accuracy"]:.1%}</td><td>{new["source_bag_metrics"]["accuracy"]:.1%}</td><td>{new["native_high"]}/{new["native_count"]}</td></tr>')
    cards = ''.join(f'<li><strong>{escape(s["task"])}</strong><span>{escape(s["status"])}</span><p>{escape(s["detail"])}</p></li>' for s in steps)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ampixa · word detector work</title>
<style>body{{margin:0;background:#f7f5ef;color:#202b31;font:17px/1.65 system-ui,sans-serif}}main{{max-width:1000px;margin:auto;padding:30px 24px 70px}}header,footer{{font-size:14px;border-bottom:1px solid #c9cbc3;padding-bottom:15px}}h1{{font-size:36px;line-height:1.2}}h2{{font-size:24px;margin-top:36px}}a{{color:#315eae}}.facts,.flow{{display:flex;gap:20px;flex-wrap:wrap}}.facts p{{flex:1;min-width:180px;border-top:2px solid #007d78;padding-top:10px}}.facts b{{display:block;font-size:29px}}.flow span{{border:1px solid #b1b9b1;padding:10px}}ol{{padding-left:23px}}li{{margin:24px 0;padding-left:10px}}li span{{display:block;color:#007d78;font-size:14px}}li p{{margin:5px 0}}table{{border-collapse:collapse;width:100%;font-size:16px}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #c9cbc3}}.table{{overflow:auto}}.note{{font-size:14px;color:#52615d}}footer{{border-top:1px solid #c9cbc3;border-bottom:0;margin-top:40px;padding-top:15px}}</style>
<main><header>ampixa / speech research · <a href="/">Open the listening lab</a></header><h1>Catch pronunciation and prosody inside the word.</h1><p>The system now aligns actual speech, exposes separate phonetic and prosodic evidence, and compares a word against preferred readings. The current classifiers are too weak to claim accurate Hindi-leakage detection.</p><p class="note">Measured snapshot: {escape(result['updated_utc'])}. Running-job counts can change. {escape(result['machine'])}</p>
<div class="facts"><p><b>{len(receipts)} / 512</b>IndicF5 matrix clips complete</p><p><b>120</b>native word-bank recordings extracted</p><p><b>30</b>listening cases with timed word playback</p></div>
<h2>Architecture</h2><div class="flow"><span>Text + WAV</span><span>Word alignment</span><span>Phone paths / relative prosody</span><span>Preferred-word differences</span><span>Weak bag supervision</span><span>Timed evidence, separate scores</span></div><p>Absolute pitch and loudness are excluded from the local feature heads. A positive IndicF5 utterance supplies a weak bag label; it does not label every word as a Hindi error. The reference branch compares matching word sequences and rejects self-reference negatives.</p>
<h2>Results we can defend</h2><div class="table"><table><tr><th>Branch</th><th>Independent word head: source-bag accuracy</th><th>Preferred-reference head: unique-query accuracy</th><th>Reference head: native high scores</th></tr>{''.join(rows)}</table></div><p class="note">Both accuracy columns use 24 IndicF5 and 24 Audio8 source queries from the already inspected benchmark. Reference results average preferred readings per unique query; duplicated pair comparisons are not counted as independent examples. None of these numbers measures word-level Hindi accuracy.</p>
<h2>Work sequence</h2><ol>{cards}</ol><h2>Checkpoint transfer</h2><p>On six Pocket parent/fine-tune seed pairs, the independent phonetic and combined heads each score the Hindi parent higher in {checkpoint_probe['branches']['phonetics']['parent_above_fine_tune']}/6 and {checkpoint_probe['branches']['combined']['parent_above_fine_tune']}/6 cases. The independent prosody head does so in {checkpoint_probe['branches']['prosody']['parent_above_fine_tune']}/6 cases. These cover only three texts; checkpoint role is a diagnostic direction, not accent gold. Prosody therefore needs stronger text/context-conditioned preferred readings before reliance.</p><h2>Voice and decoder controls</h2><p>Only {voice['qualified_controls']} of {len(voice['controls'])} conversions passed the predeclared automatic preservation gates. The two qualified voice swaps come from one native original. This cannot establish speaker invariance. Historical whole-clip head scores in the new runtime failed compatibility and are excluded from this conclusion.</p><p>Audio8 queue at its last observation: <strong>{escape(gpu['status'])}</strong>. It waits for spare compute and retains the native memory/disk guards.</p>
<h2>Schwa evidence</h2><p>The temporal probe changes one candidate vowel-count site at a time and sums exact CTC paths. It marginalizes over ʌ, ə and ɐ rather than treating vowel quality as a fixed accent label. On the 512 IndicF5 matrix clips, {sites['supported']} of {sites['candidate_word_occurrences']} candidate word hypotheses were representable; {sites['excluded']} were explicitly excluded. Some counts include carrier words and multiple candidate hypotheses, so these are not independent target-word examples.</p><p>The native bank supports {bank['supported_anchor_words']} lexemes. Only {ranks['supported']} of {ranks['site_comparisons']} supported matrix site comparisons currently have enough same-word native references; {ranks['unsupported']} remain unsupported. The query speaker and exact audio are excluded from the native reference calculation. Missing vocabulary/reference coverage cannot become a low-risk prediction.</p><p>Raw deletion alternatives also win frequently in native speech. They can reflect native variation, lexicon mistakes or probe bias; they are not automatically Hindi errors.</p><p class="note">The cross-lingual phone recognizer and lexicon paths remain unvalidated for Nepali. MMS forced alignment has a noncommercial research licence. Nepal Bhasa/Newa requires its own verified frontend and data.</p><footer>Private research. Native result: {escape(str(args.out / 'progress.json'))}. No human judgments were fabricated; no public Ampixa site was changed.</footer></main></html>'''
    (ROOT / 'review/wordlab-progress.html').write_text(page)
    print(json.dumps({'progress': str(args.out / 'progress.json'), 'page': 'review/wordlab-progress.html', 'matrix_f5': len(receipts)}))


if __name__ == '__main__':
    main()
