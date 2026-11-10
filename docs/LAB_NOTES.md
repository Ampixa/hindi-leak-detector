# Hindi checkpoint leakage lab

Measure Hindi pronunciation patterns that persist when a Hindi TTS checkpoint is
fine-tuned for Nepali. Nepal Bhasa/Newa is a separate future target: the existing
Ampixa `nepa-newa-text-frontend` is a Nepali frontend, not a Nepal Bhasa G2P.

The current actual-word review is **http://100.80.123.8:8770/swar** (Tailscale).
Main acoustic inference/training runs on k2, an M2 Ultra with64GB RAM, under
`/Volumes/T9/hindi-detector`. The verified24-layer Pocket teacher runs on L40;
stock AI4Bharat IndicF5 supplies the Hindi-source comparison. Audio8 appears only
as a comparison already present on the user's Swar page.

## Fresh sentences outside Swar, 2026-11-09

Private listening: **http://100.115.106.117:8782/**, served from k2 through an
owned SSH forward bound only to this Mac’s Tailscale address. It contains 26 new
sentence strings, four matched conditioning voices and two TTS systems: 208
actual readings. The source systems are Original AI4Bharat IndicF5 and the
approved 24-layer step250000 Nepali Pocket teacher. Each query uses two other
Native conditioning voices; no waveform is compared to itself. The live v31
checkpoint and thresholds stayed frozen; the rejected v71 head is not used.

Fourteen contexts revisit the seven listener-confirmed words, two revisit the
listener-similar पिढको case, and ten introduce focal spellings absent from the
live detector’s historical fit/development metadata and the Swar word review.
All sentence strings were checked against previous selection manifests. This
establishes fresh text, not morphological-root or unseen-model independence.
Earlier confirmations do not label these new utterances.

All 208 readings were generated and analyzed. Lexical checks: Pocket 104 pass;
IndicF5 93 pass and 11 quarantine. Focal-word flags among content-pass queries:
IndicF5 60/93, Pocket 4/104. These are candidate counts, not accuracy or a Native
false-positive rate. Additional candidates with text pass and two qualified
Native references include सहमति, नैतिक and बैठक. All new human labels remain
null. Full sentence/word playback and every unflagged/uncertain case remain
available; only 126/724 word positions per system have tested contrasts.

The initial L40 destination reserve guard stopped after 73 outputs; a measured
capacity recovery allowed one more before another stop. The remaining 30 used
the identical approved CUDA teacher and memory/SSH transport to k2’s guarded
storage. A regenerated old sample matched its PCM WAV SHA exactly before any
new streamed output was accepted. No reserve was lowered or unrelated storage
deleted. The L40 partial mirror is incomplete; k2 has all verified outputs.

Artifacts: `data/cross_system_sentences_v80`,
`reports/cross_system_content_v80`, `reports/cross_system_results_v80`,
`reports/cross_system_teacher_stream_v80`,
`reports/cross_system_batch_parity_v80`, and
`reports/cross_system_review_service_v80`. The 186 unit tests pass; initial
persistent-batch word decisions and reference qualifications match the
unchanged production CLI. Source model families appeared in earlier work;
this is an outside-Swar discovery check, not a new held-out accuracy benchmark.
No classifier fitting, prosody claim, public release or external message.

## Current word detector,2026-11-09

Private review: **http://100.80.123.8:8770/swar**. The selected checkpoint is
`reports/task_vowel_release_v31/model.pt`, SHA-256
`58422b5a360149406cdb5a12e63e3360a0f4133224d26b556f9136468c2fce85`.
`active_vowel_model.json` is the word-detector pointer. The separate
`active_training.json` points to an older completed source-classification
experiment; its source accuracy is not Hindi-error accuracy.

The release flags both supplied development regressions: हैन has a vowel-quality
change, and किनकि has a missing internal vowel. A correctly pronounced किनकि in
another sentence remains unflagged. These words and all six picked sentences
were excluded from fitting, but the two known errors were used as development
regressions to select the final release. They are not an independent accuracy
sample. The21 Swar word flags across90 usable recordings are candidates,
not confirmed Hindi errors. Nine have support against both qualified teacher
references. All six picks are available; four numeric cases now use explicitly
ASR-corroborated spoken expansions, while six ambiguous cases remain unscored.
The new तिरेर flag is relative to the real reading; neither generated teacher
reference qualifies to corroborate its expected final vowel.

Architecture: dual Nepali/Hindi G2P builds competing phone hypotheses. Frozen MMS
word alignment and two frozen phone encoders (XLSR53 and LV60) measure local CTC
path margins and vowel-region evidence at three boundary settings. Two MLP16
heads,945 parameters each, score deletion and quality changes separately. Each
uses57 measured query/reference/difference features; both probes must support
the final phonetic validity gates. Deletion uses the secondary probe and238
training pairs. Quality retains the primary-probe head fitted on418 combined-kind
pairs. Development contains147 pairs. The reference must support the expected
Native vowel; conflicting qualified references yield uncertainty. No word lookup
supplies a prediction, and generator/speaker IDs or voice embeddings are not
model inputs.

The quality head was rolled back to an earlier frozen candidate because the
lowest-development-loss combined head missed the known हैन regression. There is
no word-specific inference rule. This release choice and all retrospective
metrics are preserved in its report and source snapshots.

Fresh controlled audits on44 word families and two held-out conditioning
groups: deletion34/39 detected,0/39 sham flags; quality17/25 detected,0/25 sham flags across two cohorts. The enlarged quality
follow-up detected13/18 edits on16 new word families; all those source/donor
readings passed lexical-content checks. The model remained frozen.
Source/donor nuclei were selected for clear acoustic evidence. These are known
waveform edits, not independent real Hindi-error labels. Earlier tiny nucleus
cuts were detected in only3/47 deletion and1/8 quality cases; that failed
sensitivity diagnostic is also retained. Zero observed sham flags does not imply
zero Native-error false positives. General Hindi-error accuracy remains unknown. The reference-mode audit across86
cases flagged14 words in human recordings and21 in Swar when both were compared
with the generated teacher. These are disagreements, potentially including
acceptable Native reductions and reference mistakes. A fixed human-to-human
same-word audit found5 flagged pairs among102 comparisons/18 word families;
context and pronunciation variation prevent treating that as a Native-error FPR.
Only260 of1433 word positions have a supported modeled vowel contrast; the rest
are not assessed for other pronunciation errors.

Eight cold inference checks passed, including both regressions with explicit and
cached teacher references, the correct किनकि control, and two words outside the
799-word G2P cache. On80 human-reference sites, half-volume changes produced no
flags. WORLD reconstruction produced2 flags; raising/lowering pitch three
semitones added none over that reconstruction. Both known errors remained
flagged through all eight transformed checks. This measures nuisance sensitivity,
not validated Hindi-like intonation.

Direct inference **on k2**:

```sh
cd /Volumes/T9/hindi-detector
.venv-indicf5/bin/python -m scripts.detect_nepali_leakage --query QUERY.wav --text 'NEPALI TEXT' --out reports/NEW_RESULT --check-query-content
```

The original86 review texts have two cached verified24-layer teacher renderings.
Content checks leave usable cached references for84 texts; the other two return
`unscored_reference_quality`. Unseen text automatically generates two references on L40 using the exact teacher,
checks lexical content with Nepali Zipformer/IndicConformer, and then qualifies
individual phonetic sites with both probes. Only content-pass new references
are used; cached references follow the same content-pass policy. A content pass
is never a pronunciation label. The command fails
explicitly if generation/content/reference checks fail. `--cached-only` disables
new rendering. Alternatively repeat `--reference TRUSTED_SAME_TEXT.wav` to supply
references. Explicit references are trusted inputs and are not ASR-screened.
`--check-query-content` also checks the query with Nepali Zipformer and
IndicConformer, retaining a separate Hindi-Conformer transcript audit. Gross
wrong-text mismatch returns `unscored_content_mismatch`, with no pronunciation
judgment. Quarantine stays explicit and does not suppress the known किनकि error.
Without this option, the caller supplies trusted matching text. All258 actual
recordings were separately content-audited: Swar74 pass/12 quarantine, real60/26,
Audio8 64/22, no gross rejects; these are lexical diagnostics, never accent labels.

New-reference synthesis needs the existing private k2→L40 SSH/runtime setup;
the22,824-byte head download alone does not include the acoustic model weights.

`result.json` contains a scored/abstained status, word intervals, tested contrasts, reference qualifications,
per-probe evidence, conservative consensus and `word_flags`. A descriptive bank
records17 observed human occurrences across15 contrast entries. Where available,
the CLI/review exposes these examples as variation context; the bank never changes
scores or decisions and does not assert acceptability or Hindi origin. One qualified
reference is reported as weaker support than two; unsupported sounds/contrasts
remain untested. No flag does not guarantee correct speech. Relative pitch and
duration are available as diagnostics; Hindi-specific prosody/intonation accuracy
has not been established. Nepal Bhasa/Newa remains a separate future target.

Prosody admission experiments are complete and failed: the recovered source
head flagged332 real-human versus263 Swar word positions; physical pitch
transfers detected2/37 with1/74 Native-control flags; physical word rhythm
detected2/34 with0/68 Native-control flags at the frozen thresholds. The latter
reuses word families from the pitch audit, held out from rhythm fitting. A
same-word independent-Hindi-reference pitch baseline also failed development.
These source-style results are not Hindi-error accuracy; no prosody head joins
the live vowel model. Exact reports and pre-fit protocol correction are retained
in `reports/prosody_admission_v55`, `reports/prosody_transfer_training_v49`,
`reports/prosody_direction_development_v50` and `reports/rhythm_transfer_training_v52`.

Release evidence: `reports/task_vowel_release_v31/review_report.json`,
`reports/fresh_gap_audit_v32`, `reports/fresh_waveform_audit_v31`,
`reports/release_voice_audit_v34`, `reports/release_cold_verification_v31`, and
`reports/new_text_inference_v35`, `reports/quality_followup_summary_v38`,
`reports/release_cold_verification_v43`, `reports/query_guard_verification_v44`,
and `reports/release_handoff_v39/runtime_manifest.json`. Current browser review preserves exact original
recordings and supports word/sound replay; it does not edit the original Swar
site. Source/model/corpus licences, including MMS research restrictions, apply.

## Historical source experiments,2026-11-07

The completed Pocket candidate is **not reliable**. Across held-out focus words and reference groups, its three heads separate Pocket Hindi parent from Pocket Nepali at only 55.1%, 55.9% and 57.0%. These are weak source-role metrics, not actual word-error accuracy. The combined head gives a high score to at least one word in 30/32 native-control clips. Native preprocessing also limits that check. See reports/pocket_crossed_models_v4 and reports/pocket_native_validation_v4.

The crossed data now contains 512 IndicF5, 512 Pocket Hindi-parent and 512 Pocket Nepali readings. Both Pocket checkpoints have verified identical decoder parameters. Focus lexemes and reference groups are separated across splits; reference voices are also held out. Same-architecture controls reduce one confound but do not establish voice invariance or pronunciation correctness.

The next foundation is real human speech. data/human_regions_v5 recovers 42,276 upstream utterance metadata rows from the pinned IndicVoices-R revision and links all 25,372 staged clips. All verified regions are West Bengal districts: Alipurduar, Darjeeling, Jalpaiguri and Kalimpong. Nepal-region coverage, current residence and Hindi exposure remain unverified. Region never supplies an error label.

data/human_g2p_bank_v5 selects 32 word families, 232 real recordings and 187 original speaker groups. Both the actual Ampixa Nepali frontend and Hindi eSpeak G2P outputs are preserved as alternative hypotheses. The human extraction completed all 232 recordings without processing failures, finding 264 planned-word occurrences. Competing paths are representable for 20/32 families; unsupported units remain explicit. Hindi-path preference is **not** a Hindi-leakage judgment: native speakers can favor either path, and G2P/probe/alignment errors remain possible.

The user wants both broad native Nepali variation and the preferred TTS style. Keep three separate outputs: native-variant support, preferred-style distance and Hindi-phone evidence. No human error/style labels have been fabricated. Current acoustic evidence does not calibrate any of these judgments. The first supported path bank is an observation bank, not a trained human-word classifier.

The listening page now serves 32 human word cases plus the existing 30 TTS comparison cases. Human words occur in different sentence contexts; their contours cannot establish matched intonation errors. The first occurrence is fixed rather than selected by score. Short automatic spans are flagged, original full recordings remain playable, and no cropped WAV copies are saved. Local staging can concatenate utterances: a text/duration match to one upstream utterance is candidacy, not proof of original audio identity. Verified original utterance context and Nepal-region coverage are next data requirements before prosody validation.

A 120-recording native bank covers 20 candidate lexemes across six source groups each. Temporal CTC hypotheses change one candidate schwa-count site at a time, marginalizing over three central-vowel qualities. Thirteen lexemes have sufficient supported native site references. Unsupported phones and insufficient reference coverage stay explicit. Raw deletion evidence frequently occurs in native speech and is never equated with Hindi leakage.

The existing 96-clip benchmark remains outside fitting. Separate development data has 32 native, 32 IndicF5 and 32 Audio8 clips from twelve training and four development source groups. Source identities are user-authorized **weak bag labels**, never fabricated word/human gold. Human review is optional validation, not a prerequisite for these experiments. The optional human-supervised trainer retains its own genuine-judgment requirements. Old blind forms remain at ports 8768/8769; the listening lab is read-only and does not overwrite annotations.

Use module entry points from this checkout, e.g.
`python -m scripts.extract_comparison_features` and
`python -m scripts.train_pronunciation_head --help`.

After two independent native reviewers agree on each usable pronunciation label,
the k2 commands are:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/k2/madhyam/pocket-tts/.venv/bin/python -m scripts.train_pronunciation_head --annotations annotations/development_labels.jsonl --features reports/development_features_v1 --pack data/development_v1 --variety '<same reviewer-specified variety>' --out reports/pronunciation_head_NEW
PYTHONDONTWRITEBYTECODE=1 /Users/k2/madhyam/pocket-tts/.venv/bin/python -m scripts.evaluate_pronunciation_head --model reports/pronunciation_head_NEW/model.json --annotations annotations/labels.jsonl --features reports/comparison_features_v1 --pack data/indicf5_audio8_v1 --out reports/pronunciation_eval_NEW
```

Training requires at least eight agreed clips per class across four training
speaker groups and four development clips covering both classes. These are
plumbing minimums, not enough evidence for deployment. Uncertain/disputed/other
errors stay excluded and counted. The final evaluator holds Audio8 out of model
fitting, reports counts and speaker-group uncertainty, and refuses reused
speaker IDs or normalized texts. Cross-corpus real-person identity is unverified.
No overall MOS rating is converted into a Hindi-leakage label.

## Reproduce

Python 3.11+ with numpy and soundfile:

```sh
python3 scripts/fetch_sources.py
python3 -m hindi_detector.cli mine --frontend-root .sources/frontend --hindi-csv .sources/hindi/hindi/data/extra_large.csv --out data/schwa_candidates_NEW
python3 -m unittest discover -s tests -v
python3 -m hindi_detector.cli prosody /absolute/mono.wav --out reports/prosody_NEW
```

No output group is silently overwritten. First run: 34,952 Hindi dictionary rows,
5,690 overlapping rows, 1,929 overlapping rows excluded for unsupported phone
symbols, 386 candidate pairs/words. These are **unreviewed**: 283 include a
word-final schwa contrast, and entries such as आज show why a lexicon output must
not become a gold target without listening and native review.

Hindi phone aliases are used only to mine schwa presence/absence. They normalize
vowel length and affricates and cannot be used as acoustic IPA labels. Unsupported
symbols exclude a row with an explicit counter. No Hindi rule is applied to Newa.

## Acoustic evidence

Use a 16 kHz mono word crop and a local pinned phoneme model. Optional dependencies:
`torch`, `transformers`, `scipy` (see pyproject). The checked probe model is
`facebook/wav2vec2-xlsr-53-espeak-cv-ft`
@`2c733782da5604684829819a5eb744c193fe9398`, blank ID 0, 392 vocabulary tokens.
It is a cross-lingual research probe, not a validated Nepali/Newa phone recognizer.

```sh
python3 -m hindi_detector.cli score-audio word.wav --model-directory /absolute/model --hypotheses /absolute/reviewed.json --out reports/word_NEW
```

The hypothesis JSON contract is authored by this project, not assumed from an
external dataset: `review_status: "approved"`, `reviewer: "..."`,
`target_tokens: ["..."]`, `hindi_tokens: ["..."]`. Use **exact model vocabulary
tokens**, never raw frontend tokens. Unknown/special tokens fail explicitly.
Positive `hindi_minus_target_log_likelihood` supports the Hindi hypothesis over
the target hypothesis. It is not a calibrated leakage probability. Both paths
must be acoustically possible. Exact CTC sums all valid paths; tests compare
against exhaustive enumeration including repeated phones.

## Storage and provenance

Created by Codex (GPT-6), session `/root`, 2026-11-06 UTC on cdjk, initially at
`/Users/cdjk/github/hindi-detector`; user/Ampixa owns the work. No prior repository
or Git history existed. Original source in `hindi_detector/`, tests in `tests/`,
reproduction helpers in `scripts/`; `.gitignore`, `pyproject.toml`, and
`sources.lock.json` describe the project and pinned inputs. `data/` and `reports/`
groups have separate inventory READMEs. `.sources/README.md` and generated
inventory document every downloaded source. `review/` is the local review UI.
All are active for this task. No model is uploaded to HF. Any model download is
documented separately; external licences continue to apply.

Last review: 2026-11-09; next review: 2026-11-10. Keep source and compact results
until superseded. Delete transient diagnostics within 24 hours of finishing;
retain models only while this experiment consumes them. Do not delete or modify
other projects' checkpoints. Local disk at start: 35 GB available. k2 internal
disk: about 5.2 GiB available; T9: about 274 GiB available. Work is placed on T9.

The review URLs require Tailscale on the viewing device. Both backend servers
bind only to loopback on k2; Tailscale Serve forwards private ports 8768/8769.
See reports/review_connectivity.json for verification and restart instructions.

## User-authorized source-proxy baseline

On 2026-11-06 the user directed us to use original IndicF5 as the Hindi-like
source and Audio8 as the preferred Nepali source. This permits immediate weak
supervision; native review is optional validation, not a prerequisite for fitting
this baseline. scripts/train_source_proxy.py uses matched development examples,
keeps source assumptions explicitly separate from human judgments, and leaves
the original benchmark outside fitting. New Audio8 R3 development synthesis uses
the same 32 texts and exact reference WAVs/transcripts as IndicF5. Actual results
and model weights are in reports/source_proxy_v1. Held-out source-proxy accuracy
is 35/48 (72.9%); IndicF5 ranks above Audio8 in 23/24 matched pairs. The fixed
threshold gives high scores to 11/24 native controls, so it is unreliable as an
absolute leakage flag. Read the result at http://100.80.123.8:8768/results .

Reproduce on k2:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/k2/madhyam/pocket-tts/.venv/bin/python -m scripts.train_source_proxy --development-pack data/development_v1 --development-features reports/development_features_v1 --audio8-pack data/audio8_proxy_v1 --audio8-features reports/audio8_proxy_features_v1 --test-pack data/indicf5_audio8_v1 --test-features reports/comparison_features_v1 --out reports/source_proxy_NEW
PYTHONDONTWRITEBYTECODE=1 /Users/k2/madhyam/pocket-tts/.venv/bin/python -m scripts.score_source_proxy --audio /absolute/mono.wav --head reports/source_proxy_v1/model.json --encoder models/phoneme_probe --out reports/score_NEW
```

A high score means IndicF5-like under the source assumption. It may reflect
pronunciation, vocoder, timing or channel differences; it is not a calibrated
Hindi-leakage probability or a word-level schwa diagnosis. Test voices and texts
are held out, but both TTS architectures appear in fitting and test.

## Direct word/prosody inference

Use the pinned k2 `.venv-indicf5` word runtime, inventoried in `runtime/word_localization.freeze`:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-indicf5/bin/python -m scripts.analyze_word_audio --audio /absolute/mono.wav --text 'खानामा नमक थोरै छ।' --models reports/wordlab_models_v3/models.json --out reports/word_analysis_NEW
```

Returns real word times, separate weak phonetic/prosody scores, central-vowel evidence and normalized contours. Direct/cached word scores matched exactly in the verified smoke case. Numeric tokens require explicit spoken expansion. Six word/reference branches are research models, not calibrated Hindi probabilities. The old global head failed feature-runtime compatibility in the word runtime; its voice-conversion scores are excluded. Only 4/18 converted controls passed automatic preservation gates, including two true voice swaps from one native original; this cannot establish voice invariance.

MMS alignment is a noncommercial research dependency (CC BY-NC4); replace it appropriately before commercial deployment. OpenVoice is used only for research confound controls, not as an accent-model input. Model inventories pin actual bytes, revisions and licences. Newa requires separately verified G2P and audio.

The external Pocket checkpoint diagnostic covers six seed/text pairs from three texts: phonetics and combined score the Hindi parent higher in 5/6 pairs; prosody does so in 0/6. This is checkpoint ordering, not accent accuracy, and the historical exact parent hash remains unverified. These results motivate text/context-conditioned prosody supervision rather than relying on source identity alone. Detailed native status: reports/wordlab_progress_v3_checkpoint_audit/progress.json.

Original-audio verification (v6): all232 pinned original recordings reproduce their staged mono WAVs byte for byte using the inspected mean-channel/sphn0.2.1/peak0.95/PCM16 recipe. The pilot measured a 256-sample shift at 24 kHz relative to a zero-delay polyphase resample; this is a measured two-clip diagnostic, not a universal correction. Verification is complete under data/human_original_audio_v6; original stereo payloads stay unchanged, and low channel agreement is flagged before any prosody-reference acceptance. Exact byte reproduction establishes source correspondence, never a pronunciation label. reports/human_staging_audit_v5 records actual staging sources and unknown historical execution revision.

## Auxiliary schwa request component (v7)

The same-voice control experiment rendered960 clips:48 words, five public renderer voices, two rates, and two requested one-site vowel counts. All32 human-bank and24 original IndicF5/Pocket focus words were excluded. A site-likelihood component scored37/48 (77.1%) on jointly unseen words and one renderer voice, compared with24/48 from the raw CTC zero threshold. A pooled-phone component scored29/48 (60.4%). These are request-class metrics, not actual Hindi or native-pronunciation accuracy. Model choices/weights were frozen before test NPZ reads. Reports and exact limits: reports/intervention_heads_v7. Pocket remains the preferred TTS comparator; Kala is only an auxiliary intervention renderer.

Direct isolated-word inference (research component):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-indicf5/bin/python -m scripts.score_word_schwa --audio /absolute/isolated_word.wav --candidate-id schwa-00029 --models reports/intervention_heads_v7/selection/models.json --out reports/schwa_word_NEW
```

Caller must provide an isolated word and exact lexical candidate; the command does not establish word identity, boundary correctness or the validity of the target pronunciation. Unsupported paths yield no replacement score. Native support, preferred style and actual Hindi probability stay unknown. Direct/cache scores reproduced to numerical precision. No prosody classifier is established by this isolated-word experiment.

The completed human crop audit reproduced161 cached scores exactly.28 of264 planned-word occurrences change G2P preference across seven fixed windows;103 have unsupported phone mappings;132 keep the same sign, and one has incomplete/tied evidence. These warnings are visible in review/human_word_data_v6 and the private listening page. Agreement is not pronunciation correctness.

## Actual-word transfer (v8, completed measurement)

`scripts/apply_schwa_components.py` applies the frozen v7 components to verified existing word alignments across1928 actual speech clips and1844 candidate-word occurrences. Every crop is independently encoded at seven fixed boundary recipes, matching isolated-word training. Short, silent, impossible or unsupported paths remain unavailable. No real Hindi-error or native-acceptability labels are inferred from system identity or speaker geography.

The full run completed1928 clips with zero execution failures and1844 planned word occurrences. The site component has1157 baseline scores,1104 occurrences with all seven windows measured, and293 occurrences with request-class changes across windows. IndicF5 scores above Pocket in25/64 supported reserved matrix pairs; the Pocket Hindi parent does so in43/64. These are descriptive request-score orderings, not Hindi-error accuracy. Scores on human speech are not false-alarm labels because actual realizations/acceptability remain unreviewed. Full evidence: `reports/real_word_transfer_v8`; summary and per-word comparisons: `reports/real_word_transfer_summary_v8`.

The private review at http://100.80.123.8:8770/ now serves110 cases:48 fixed matrix examples spanning all24 focus words,32 human word families, and30 earlier comparisons. Per-word request scores, seven independent crops, speaker-excluding observed human ranks, and same-text Pocket pitch/timing differences remain separate. Sparse voicing yields no pitch-shape comparison (fixed50% coverage contract, not a learned accent threshold). All86 software tests pass; they do not measure accent accuracy. Desktop/mobile layout, all eight human audio streams, word playback and the live metrics were verified. Superseded v5/v6 case/index copies are removed with byte-exact reconstruction proofs and retained manifests in their retirement receipts. To reproduce v8 from the former v6 base, first restore that base using `reports/review_v6_retirement_v8/deletion_plan.json`.

## Human sentences read by both TTS systems (v9, active)

Only five original matrix words overlap the human bank. `data/human_matched_v9` freezes63 verified original human contexts covering all32 bank words, with two different-district examples per family and two separate TTS reference groups. Three stereo-review examples were excluded from this prosody-anchor selection; they were not assigned accent labels. Contexts deduplicate into126 jobs per TTS,252 planned generated readings. All data remain exploratory development evidence.

Pocket's126 readings completed in `data/human_matched_pocket_v9`. Primary IndicF5 generation uses `data/human_matched_f5_v9` on k2 MPS; the first CPU reading is retained separately as a cross-device diagnostic after its103-second runtime. `scripts/continue_human_matched.py` extracts Pocket word evidence, waits only on the verified owned F5 process, then extracts completed F5 readings. No automatic synthesis restart or error-label inference. Exact sources, worker PIDs and phase are in `reports/human_matched_v9`; `/progress` exposes current generation/analysis counts. Model and reference hashes are checked by the existing strict inference adapters; the Pocket adapter now accepts this explicit frozen source plan and selected systems.


## Scaled Pocket candidates, v11/v12

Final review 2026-11-08T04:59:20.983869+00:00. final_summary.json records measured outcomes and inference command.99 software tests passed. Readable results: http://100.80.123.8:8770/overnight. Source candidate512 train texts; physical candidate192 train/32 dev/64 test texts,576 controls. Post-ASR63 test pairs: combined balanced source accuracy64.3%; conservative recall22.2%, native-sham false flags4.8%. Real Hindi-error accuracy unknown; prosody near chance. Training finished01:59UTC before08:00Nepal; final delivery missed cutoff. cache_deletion_plan.json/receipt.json document unused installer cache removal: logical171505791B, measured free delta22437888B. No unique audio/model removed. Review corpus/controls by2026-11-09; retain for evaluation reproduction/expanded quality coverage only while consumed. Logs expire24h after reviewed completion. physical_inference_launch.json/log records final CLI check. All *_launch.json and *.log in this group are owned task handles and explicit traces. No publication or automatic janitor.

Run the exact inference command in reports/overnight_v11/final_summary.json on k2 from /Volumes/T9/hindi-detector. Both query and trusted Nepali reference must read the same text. CLI does not apply ASR and abstains on unsupported contrast families. Original source/conditioning/focal-word partitions remain separate.


## Active requested comparison: teacher24-layer / IndicF5 (v14)

active_comparison.json points to the current run. Preferred Nepali comparator: L40 teacher step250000,24 layers, SHA2b4636e3aed0e4f2d035630e504be95fc58f494b690a851a91bfe0dd8299516f. Comparator: original ai4bharat/IndicF5 revisionfb652a96e508091045890dae5bb99fe511facbcf, SHAba7f3671180fb7784e24bd1dafc96e729a38ce02e7f6d3877cdef32525a1865c. Old k2 v1c and Pocket Hindi roles are superseded, not silently relabeled. Frozen plan:768 texts /1536 conditioning jobs per system /3072 planned recordings. Shared <=5s prompts and matching transcripts are auditable in reference_inventory.json. First8 recordings per model passed finite-waveform and content checks. Controller resumes only exact code/model/job locks and screens actual generated content; it does not turn model identity into Hindi-error gold or retrain old heads automatically. Readable live review: http://100.80.123.8:8770/comparison.102 software tests passed.

New source inventory: scripts/build_teacher_f5_plan.py constructs the corrected text/prompt lock; hindi_detector/corrected_synthesis.py strictly loads the actual teacher training inference path or original IndicF5; scripts/run_corrected_synthesis.py records waveform hashes/quality errors; scripts/continue_corrected_pair.py checks both smoke identities, controls owned bulk workers and screens their outputs. Source snapshots, exact PID/commands and code locks are retained per group. Never restart failed workers blindly. Existing model licences apply; MMS-derived prompt alignment remains a research dependency. No public publishing or messages. active_comparison.json is created by Codex /root, active owner Ampixa/user, source the user-requested v14 model manifest; retain until superseded, review2026-11-09, regenerate from model_pair.json and controller handle, NOT on HF.

## Corrected word detector training (v15 evidence / v16 fitting)

`active_training.json` selects the current checkpoint; the private review at
http://100.80.123.8:8770/comparison shows its actual fitting counts and separately
labels development and reserved results. Original IndicF5 and the L40 24-layer
Pocket teacher are the only TTS sources. Old six-layer Pocket and Pocket Hindi
recordings and weights are excluded from this run.

`scripts/extract_corrected_training.py` accepts complete content-passing matched
quartets, extracts temporal central-vowel/phone evidence and relative prosody,
and constructs matched physical word insertions and teacher sham controls.
`scripts/content_gate_rpc.py` runs the pinned ASR screen in an isolated runtime.
`hindi_detector/word_evidence.py` measures explicit schwa counterfactual sites;
unsupported phone paths remain unavailable. `hindi_detector/structured_detector.py`
forms separate phonetic and contextual prosody inputs without hidden speech
embeddings. `scripts/train_corrected_detector.py` fits fresh small neural heads
with equal matched-focal weights, limited background weight and a within-word
ranking loss. The earlier globally weighted v15 pilot is retained as a diagnostic,
not the selected candidate. Source labels establish provenance, not actual Hindi
pronunciation or intonation correctness.

`scripts/train_corrected_preview.py` refreshes development-only weights from the
currently complete groups; `scripts/audit_structured_pilot.py` exports fixed-order
development examples. `scripts/finish_corrected_training.py` waits for the full
stream and recovers observed long-input MPS kernel failures on CPU for the whole
matched unit before the final fit. Original failure receipts remain intact. The
final model and development thresholds are written before reserved feature arrays
are read. `scripts/detect_structured_words.py` provides direct same-text paired
inference, uses CPU for both inputs when either exceeds20 seconds, and preserves
uncertain/unsupported outcomes. It does not itself apply ASR.

`scripts/build_native_variation_plan.py` and `scripts/audit_native_variation.py`
prepare and score63 verified human recordings from58 source speaker groups with
two new teacher references per text, independently of fitting. Reported regions
are West Bengal; geography does not establish Hindi exposure or error labels.
`scripts/finalize_corrected_training.py` waits for final training and that audit,
verifies the same model SHA, and assembles compact metrics with reference
sensitivity keyed by actual word occurrence. `tests/test_structured_detector.py`
covers structured geometry and the matched objective;108 software tests passed
on k2. Software tests do not measure accent detection accuracy.

Artifact ownership: Codex (GPT-6), session `/root`, for Ampixa/user, created
2026-11-08 on cdjk and k2; exact UTC timestamps, original paths, inputs, code
snapshots, waveform hashes and process commands are recorded in the documented
v14/v15/v16 groups. No Git revision is available in this workspace. No model/data
upload to HF. Existing source licences apply, including the MMS noncommercial
research restriction. Review these active groups2026-11-09. Keep selected models,
configs, source locks and compact reports for current inference/reproduction;
remove redundant features, physical controls and logs within24h after their
consumer closes. Historical wrong-source bulk WAVs remain superseded and need a
dependency-checked cleanup; they are not inputs to current training.


### Video v3 inference demonstration

Added by Codex /root on 2026-11-09 UTC for the user-requested Hindi-scorer film. `scripts/score_stem_audio.py` exposes the existing pinned v83 deletion head on an arbitrary mono WAV plus exact Nepali text file. It verifies the frozen model/code/resource pins, uses the existing feature adapter, and reports tested sites, candidate words, and abstentions. Requires the K2 research checkout and its existing frozen model/encoder resources; it does not download or publish a release. An actual run on real_yt_yt01_008 Swar audio returned किनकि as candidate, one word tested. Receipt: video/hindiness_film_v3/reference/cli-demo-kinaki.json. This is a development smoke check, not independent validation. The resulting film and source analysis are reviewed at video/hindiness_film_v3/review.html; all asset provenance is in that project README and manifests.
