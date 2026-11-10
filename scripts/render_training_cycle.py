"""Readable internal review and native announcement draft for the actual run."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

from hindi_detector.artifacts import write_group

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'reports/training_cycle_20261109_v74'
    names = {'training': 'reports/anchored_training_v71/report.json', 'test': 'reports/anchored_test_v74/report.json',
        'runtime': 'reports/anchored_runtime_v74/report.json', 'short_audit': 'reports/short_vowel_localization_v75/report.json',
        'oracle': 'reports/oracle_window_experiment_v76/report.json', 'examples': 'reports/more_listening_v77/examples.json',
        'claude': 'reports/claude_adversarial_v68/post_training/triage.json', 'feedback': 'reports/listener_feedback_v73/applied.json',
        'confirmations': 'reports/listener_confirmations_v79/feedback.json'}
    evidence = {k: json.loads((ROOT / p).read_text()) for k, p in names.items()}
    if evidence['test']['physical_control_admission_passed'] or evidence['oracle']['anchor_redesign_go']:
        raise ValueError('This rejected-run report must not describe a passing experiment')
    now = datetime.now(timezone.utc).isoformat()
    evidence['source_files'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in names.values()}
    write_group(out, {'evidence.json': evidence, 'plan.json': {'updated_utc': now,
        'steps': [{'step': 'Actual Claude adversarial review and verified fixes', 'status': 'complete'},
                  {'step': 'Generate2404 approved24-layer teacher readings across 512 new lexemes', 'status': 'complete'},
                  {'step': 'Fit grouped-CV head, freeze externally, then open held-out physical test', 'status': 'complete; candidate rejected'},
                  {'step': 'Benchmark persistent post-utterance inference', 'status': 'complete; not streaming'},
                  {'step': 'Claude post-training review and train/dev oracle experiment', 'status': 'complete; anchor-only redesign no-go'},
                  {'step': 'Offer more playable existing candidates, record listener feedback', 'status': 'complete; three further occurrences confirmed by user'}],
        'next_decision': 'Check short-vowel representation and physical intervention validity with independently marked real-word intervals before another model fit. Do not recycle the spent test or lower its threshold.',
        'publication': 'No public release or message sent; plain-text draft only.'}},
        'Readable review of completed9November training cycle and direct private word comparisons. index.html is the primary review interface; evidence.json compact verified source snapshots/hashes, plan.json completed actions and next decision, '
        'announcement.txt native paste-ready draft for user only, inventory.json sizes/SHA, preview.log/preview.launch.json owned loopback review server logs/receipt once launched. No audio/checkpoint/encoder copies. '
        'Owner Ampixa/user; review2026-11-10. Active consumer tomorrow research-preview decision. NOT on HF; source usage limits inherited; no public publication. Keep compact evidence while benchmark consumes it; preview.log disposable24h after preview closes.',
        ['python3 -m scripts.render_training_cycle', 'Sources listed and hashed in evidence.json; working-tree renderer source, no Git repository available.'])
    design = '''
Design plan: white#fbfcff, ink#132a52, blue#214eb6, mist#e8efff,
plum#7d2851 for rejected admission, secondary#536276. Avenir Next body and
headings; Kohinoor Devanagari for Nepal text. Left-aligned reading layout:
[तलको confirmed | पिढको similar] -> [decision + physical-test table]
-> [three paired word players] -> [measured diagnostic + timing + draft].
Review: removed generic metric-card hero; the listener's two words introduce
the actual phonetic distinction. No motion, decorative badges or all-caps labels.
index.html and announcement.txt are newly authored summaries of evidence.json.
No user audio was listened to by Codex; playback offered for human inspection.
'''
    with (out / 'README.md').open('a') as stream:
        stream.write(design)
    test, timing = evidence['test'], evidence['runtime']
    table = []
    labels = {'P1': 'Focal interval removed', 'P1w': 'Removed after matched vocoding', 'P1u': 'Wider removal boundary',
        'S0': 'Reinserted unchanged vowel', 'S1': 'Same-duration cut elsewhere', 'S2': 'Shortened vowel retained',
        'S_world': 'Unchanged vocoder reconstruction', 'S_original': 'Untouched source'}
    for family in ('P1', 'P1w', 'S0', 'S1', 'S2', 'S_world', 'S_original', 'P1u'):
        row = test['families'][family]
        table.append(f'<tr><th>{labels[family]}</th><td>{row["flagged"]} / {row["attempted"]}</td><td>{"Detection" if family.startswith("P") else "False flag"}</td></tr>')
    players = []
    for n, example in enumerate(evidence['examples']):
        confirmation = ('Confirmed by you after listening. This is a detector-selected discovery, not an independent accuracy test.'
                        if example['status'] == 'user_confirmed' else 'Unconfirmed candidate. Pronunciation acceptability remains unresolved.')
        buttons = []
        for kind, label in [('swar', 'Swar TTS'), ('real', 'Original recording')]:
            interval = example['readings'][kind]
            url = 'http://100.80.123.8:8770/swar-audio/' + example['id'] + '/' + kind
            buttons.append(f'<div><button type="button" data-player="p{n}{kind}" data-start="{interval["start"]}" data-end="{interval["end"]}">Play {label} word</button><audio id="p{n}{kind}" controls preload="none" src="{url}" aria-label="{html.escape(example["word"])} {label}"></audio></div>')
        players.append(f'<article><h3 lang="ne">{html.escape(example["word"])}</h3><p>{confirmation} The flag persists against both qualified Native teacher references.</p><div class="pair">{"".join(buttons)}</div><details><summary>Sentence context</summary><p lang="ne">{html.escape(example["text"])}</p></details><a href="{example["link"]}" target="_blank" rel="noopener">Open full word evidence</a></article>')
    oracle = evidence['oracle']['summaries']
    diagnostics = ''.join(f'<tr><th>{part.capitalize()}</th><td>{r["oracle"]["0.99"]["sources"]}</td><td>{r["anchor"]["0.99"]["P1w_recall"]:.1%}</td><td>{r["oracle"]["0.99"]["P1w_recall"]:.1%}</td></tr>' for part, r in oracle.items())
    draft = 'We are previewing a word-level pronunciation checker for Nepali TTS, with playable evidence for Hindi-compatible vowel changes. Seven occurrences, including नेकपा, तलको, हुनलाई, चलाउनको and नगरौ, have listener confirmation; पिढको remains a false-flag check. General accuracy and prosody detection are unvalidated. Our larger physical-edit test rejected the newest head, so the existing baseline stays live. Results, limitations and listening examples: [insert reviewed release link].\n'
    (out / 'announcement.txt').write_text(draft)
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nepali pronunciation research —9 November results</title><style>
*{box-sizing:border-box}body{margin:0;background:#fbfcff;color:#132a52;font:17px/1.65 "Avenir Next","Kohinoor Devanagari",sans-serif}main{max-width:1040px;margin:auto;padding:36px 28px 70px}h1{font-size:clamp(30px,5vw,49px);line-height:1.15;max-width:24ch;margin:22px 0}h2{font-size:27px;line-height:1.3;margin-top:45px}h3{font-size:32px;line-height:1.25;margin:0}p{max-width:74ch}a{color:#214eb6;text-underline-offset:4px}header,small,footer{color:#536276;font-size:14px}.listener{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:28px 0;background:#e8efff;padding:24px;border-radius:14px}.listener strong{font-size:37px;display:block}.listener p{margin:0}.verdict{padding:18px 22px;background:#f7eaf2;border-left:5px solid #7d2851}table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:left;padding:12px 9px;border-bottom:1px solid #d4dff0}th{font-weight:600}thead{background:#e8efff}.scroll{overflow-x:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{margin:26px 0 38px;padding:22px 0;border-top:2px solid #cddaf0}audio{width:100%;display:block;margin:12px 0}button{font:inherit;background:#214eb6;color:white;border:0;border-radius:6px;padding:10px 14px;cursor:pointer}button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #7d2851;outline-offset:4px}summary{cursor:pointer;color:#214eb6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#e8efff;padding:20px;font:14px/1.7 "Avenir Next",sans-serif}footer{margin-top:40px;padding-top:20px;border-top:1px solid #d4dff0}.status{min-height:1.5em;color:#7d2851}nav{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:650px){main{padding:24px 16px}.pair,.listener{grid-template-columns:1fr}th,td{padding:10px 6px}.listener strong{font-size:31px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body><main><header>Ampixa pronunciation research / 9 November 2026 / private review</header><h1>The short vowel is the hard part.</h1><p>Your listening distinguishes a useful find from a doubtful flag. That distinction matters more than a high model score.</p><div class="listener"><div><strong lang="ne">तलको</strong><p>Confirmed useful detection by you.</p></div><div><strong lang="ne">पिढको</strong><p>You heard similar readings. Keep as a false-flag check, with no definitive phonetic label.</p></div></div><div class="verdict"><strong>The new candidate failed admission.</strong><p>Its conservative threshold caught only 1 of 93 focal removals and 6 of 93 vocoded removals. It has not replaced the working reference-conditioned detector.</p></div><h2>What the completed run contains</h2><p>2,404 new readings from the approved 24-layer step 250000 Nepali Pocket teacher, across 512 new lexemes. Four conditioning groups belong to each text partition. After source-only eligibility and unedited lexical-content screening, the fit used 385 sources across 184 lexemes; development used 111 sources.</p><p>Train-only cross-validation separated phonetic-collision groups and conditioning voices. A four-feature linear head was selected, calibrated on development, and copied with verified hashes to this Mac before the single held-out test. The test contained 93 eligible sources across 50 new lexemes and 4 conditioning groups.</p><p>Original AI4Bharat IndicF5 remains the pinned Hindi-source comparator. TTS source identity is never a pronunciation label. All new controlled-edit speech here comes from the approved Nepali teacher.</p><h2>The frozen physical-control test</h2><p>Threshold 98.6957 on this head’s uncalibrated 0–100 evidence scale. No post-edit abstentions occurred. Counts below describe physical interventions, not real Hindi-error accuracy.</p><div class="scroll"><table><thead><tr><th>Control</th><th>Flagged / attempted</th><th>Meaning</th></tr></thead><tbody>''' + ''.join(table) + '''</tbody></table></div><p>Shortened vowels forced the threshold upward. Three previously confirmed deletion examples fell below it in that runtime check. All 15 train-CV candidates also fell short at the demanding operating point; switching to another already-fitted head would not meet the target.</p><h2>Three more confirmed finds</h2><p>These are existing-detector discoveries from the previously inspected Swar corpus. You confirmed all three after listening. These are retrospective discovery annotations, not a held-out accuracy sample. Each word button plays the estimated word interval; the audio controls allow full-sentence context. Tailscale access is required.</p><p id="playStatus" class="status" role="status" aria-live="polite"></p>''' + ''.join(players) + '''<h2>Where the measurements fail</h2><p>A 24-source training diagnostic found both misplaced gaps and weak short-vowel emissions. Claude then proposed a matched experiment using known edit geometry. We ran it on 379 training and 109 development source pairs, comparing retained short vowels, removals and original vocoded speech.</p><p>In the table, each partition’s diagnostic threshold permits about 1% flags on its retained-vowel controls. Known intervals did not rescue the blank-versus-vowel statistic.</p><div class="scroll"><table><thead><tr><th>Diagnostic partition</th><th>Sources</th><th>Removal detection, inferred gap</th><th>Removal detection, known interval</th></tr></thead><tbody>''' + diagnostics + '''</tbody></table></div><p><strong>Anchor-only redesign failed its declared gate.</strong> The known-interval statistic also has short-vowel and intervention-validity limits. This does not prove every source phoneme was correct or that every removal was phonetic deletion. Before another model fit, the next decision needs a better short-vowel representation and independently marked real-word intervals. The held-out test is spent.</p><h2>HindiBench timing is feasible; its score needs more work</h2><p>A persistent runtime processed 8 development utterances in a median ''' + f'{timing["processing_seconds_median"]:.2f}' + ''' seconds, p95 ''' + f'{timing["processing_seconds_p95"]:.2f}' + ''' seconds on k2. Median processing/audio duration was ''' + f'{timing["real_time_factor_median"]:.3f}' + '''; cold initialization took ''' + f'{timing["cold_initialization_seconds"]:.2f}' + ''' seconds.</p><p>This analyzes completed utterances with bidirectional encoders. It is not streaming. It tests supported internal-vowel deletion sites, reports word coverage, and provides no calibrated Hindi-influence probability or validated prosody score. The timed candidate failed admission.</p><h2>Tomorrow’s announcement draft</h2><p>This is a research-preview draft for you to send after choosing a release link. No message or public release has been sent.</p><pre>''' + html.escape(draft) + '''</pre><nav><a href="announcement.txt">Plain-text announcement</a><a href="evidence.json">Verified evidence snapshot</a><a href="plan.json">Execution and next decision</a><a href="http://100.80.123.8:8770/swar">Current private detector</a></nav><footer>Created ''' + html.escape(now) + ''' UTC by Codex GPT-6 /root. Actual Claude Opus5.5 reviews preceded training and reviewed its failure. The 24-source diagnostic completed concurrently with the last review and was not included in its prompt. No general accuracy, Hindi-specific prosody, Nepal Bhasa/Newa validation, or human-speaker generalization is established. See README.md for provenance, source hashes, usage and retention.</footer></main><script>
const status=document.getElementById('playStatus');let stopTimer=null;
for(const button of document.querySelectorAll('button[data-player]'))button.addEventListener('click',async()=>{
 clearInterval(stopTimer);for(const audio of document.querySelectorAll('audio'))audio.pause();
 const audio=document.getElementById(button.dataset.player),start=Number(button.dataset.start),end=Number(button.dataset.end);
 try{if(audio.readyState<1){await new Promise((resolve,reject)=>{audio.addEventListener('loadedmetadata',resolve,{once:true});audio.addEventListener('error',()=>reject(new Error('Audio could not load; check Tailscale.')),{once:true});audio.load();});}
 audio.currentTime=start;await audio.play();status.textContent='Playing selected word.';
 stopTimer=setInterval(()=>{if(audio.currentTime>=end||audio.ended){audio.pause();clearInterval(stopTimer);status.textContent='Selected word finished. Use the controls for sentence context.';}},25);
 }catch(error){status.textContent=error.message;}
});
</script></body></html>'''
    (out / 'index.html').write_text(page)
    inventory = [{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size,
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.iterdir()) if p.is_file() and p.name != 'inventory.json']
    (out / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(out / 'index.html')


if __name__ == '__main__':
    main()
