# Experiment reports inventory

Created by Codex (GPT-6), session `/root`, 2026-11-06 UTC. Owner Ampixa/user.
Original run host k2 at `/Volumes/T9/hindi-detector/reports`; compact results sync
to `/Users/cdjk/github/hindi-detector/reports` on cdjk. Status active.
Last review 2026-11-06; next review 2026-11-13. Current consumer: Hindi transfer lab.

Each experiment group has an inventory README with file names, checksums, input
paths and commands: `k2_candidates_v1/` (candidate-mining reproduction),
`k2_phone_probe_v1/` (cross-lingual greedy phone outputs), `k2_prosody_v1/`
(descriptive timing/F0 features), `pocket_pilot_v1/` (matched parent/fine-tune
listening outputs and manifests; details there). These names describe the planned
inventory and a group is complete only if its command succeeds / manifest says so.

`k2_model_fetch.log` is temporary acquisition output; delete within 24h after
inventory verification. `k2_pocket_pilot.log` is temporary run output; preserve
tracebacks until diagnosed, then delete within 24h of the run review.
`session_status.json` records compact final run status and cleanup measurements.
`cleanup.json` records deleted k2 scratch files, their sizes, hashes and reasons.
`cleanup_local.json` records removed browser QA snapshots, screenshots and Python
bytecode from cdjk. `source_inventory.json` pins authored source files by SHA-256.
No training success or leakage accuracy is implied by any smoke-test group.
No models/recordings uploaded to HF. Known duplicates: small reports synced to
cdjk; model weights remain on k2 only. Keep compact results until superseded;
retained audio must have an active listening consumer, with dated review.

## Active detector goal, 2026-11-06

`active_goal.json` records current milestones, the user's original-IndicF5
clarification and recovery from a k2 reboot. `goal_source_inventory.json` records
the current authored source/config hashes. `comparison_qc_v1/` verifies all 96
matched native/audio8/IndicF5 files and records waveform QC; it is not accuracy.
`comparison_features_v1/` has its own compact-feature inventory and retention
policy. `indicf5_acquisition.log` and `indicf5_synthesis.log` are transient setup
and inference output; explicit failures are kept until their incident is
summarized, then logs expire within 24h. `goal_cleanup.json` records removal of
task-only package caches, transfer scratch files and browser QA output with sizes
and available checksums. Source and model inventories support reproduction.
The native review app remains an active consumer of annotations/ and the audio;
do not remove human judgments or required listening clips during this goal.

`development_features_v1/` contains 64 verified compact features on k2, with a
manifest copy on cdjk. `validation_goal.json` records executable checks of both
feature groups and the guarded training/evaluation pipeline, not leakage accuracy.
Its current consumer is implementation review; retain until superseded. Neither
MOS exports nor model provenance is human Hindi-leakage supervision.

`completion_audit.json` records requirement-by-requirement evidence and the missing
human supervision that blocks fitting and independent evaluation. Created by
Codex GPT-6 /root on 2026-11-06 UTC; owner Ampixa/user; active reference for
resuming this goal, review by 2026-11-13. Keep until superseded by actual labels
and a trained/evaluated result. No held-out accuracy is claimed.

`review_connectivity.json` records restored private Tailscale ports, verified HTTP/audio
access, restart/removal commands and the reboot limitation. Created 2026-11-06 UTC
by Codex GPT-6 /root; owner Ampixa/user. Active annotation consumer; review by
2026-11-13, retain compact incident evidence until superseded. No model or data copy.

`audio8_proxy_features_v1/` holds 32 new Audio8 development feature caches on k2,
with manifest-only metadata on cdjk. `source_proxy_v1/` retains the fitted linear
head, source-label metrics, predictions and training inventory, each inventoried
by its generated README. These are weak source-proxy results, not independently
verified Hindi-leakage accuracy. Source and model outputs remain internal.

`source_proxy_smoke_v1/` on k2 documents direct-WAV inference against the actual
fitted head: its score matched the original cached-feature score exactly. Keep
as compact model verification evidence until superseded; normal seven-day review.
`source_proxy_cleanup.json` records every deleted task-owned temporary GPU member,
size and SHA after no-open-file checks and verified k2 replacements. Raw transient
synthesis log bytes are no longer retained; all 32 success/no-EOS outcomes, exact
source, model/code hashes and receipts remain restorable on cdjk/k2. The older
completion audit records the pre-proxy-label state and is historical evidence.

## Word/prosody expansion, 2026-11-07

Current consumer: requested localized pronunciation/prosody detector. Owner Ampixa/user, Codex GPT-6 /root. Original execution on k2 T9, small JSON/models/source snapshots synchronized to cdjk; no new HF uploads. Review by 2026-11-14. Groups wordlab_{pilot,training,test,pocket_probe,native_wordbank,matrix_f5}_v3 inventory source snapshots, hashes, compact features and failures. NPZs and per-clip contours remain on k2 unless needed by the listening app. wordlab_models_v3 and reference_word_v3 retain six weak local/reference heads and source-proxy metrics. wordlab_direct_smoke_v3 verifies direct WAV versus cached inference. Native CTC site/reference/audit groups document hypothesis evidence and explicit unsupported coverage, never human accent gold. wordlab_progress_v3/progress.json is a native snapshot behind the readable /progress website.

voice_controls_v3 retains 18 controlled conversions for the current listening/confound consumer. voice_runtime_compatibility_v3 excludes incompatible historical global-head scores. voice_controls_v3_setup_incident.json records the resolved missing-helper setup failure with no generated audio. audio8_matrix_gpu_status.json is the most recent observed queue state, with its own observation timestamp; it is not a live success claim. audio8_matrix_execution_v3 inventories recovery/restored inputs and current owned worker source/state when present. These runtime logs expire within 24h of their consumer ending; active queues/services keep their logs. Completed experiment/install logs and package cache are removed after results verification and logged in wordlab_cleanup_v3.json; raw deleted diagnostics are not restorable.

Source inventories pin current authored code, not reconstructed historical code. Per-experiment source snapshots preserve exact captured source and timestamps where available. Current model/code limitations remain in each result. Never remove active annotations, source corpus inputs, selected models, running queue references or verified dataset outputs during this goal.

wordlab_checkpoint_probe_v3 inventories actual predictions on the six Pocket parent/fine-tune pairs (three texts). Its result is directional checkpoint ordering, not word-accent accuracy. wordlab_progress_v3_checkpoint_audit is the latest readable-status native snapshot and includes that diagnostic. Earlier progress snapshots are compact historical evidence for this active work; keep the latest as the review consumer, remove earlier redundant snapshots at the next dated review. wordlab_cleanup_local_v3.json logs removed browser QA files; verification_word_v3.json records the final source/runtime/HTTP checks and measured storage when present.

## Pocket replacement and human-data grounding, 2026-11-07

Current consumer: the user's localized Hindi-leakage detector, including broad native variation and preferred TTS style as separate targets. Owner Ampixa/user, creator Codex GPT-6 /root. Review by 2026-11-14; NOT uploaded to HF. Original execution k2 /Volumes/T9/hindi-detector; compact result/source metadata synced to cdjk, large features/models/audio stay on T9.

pocket_pipeline_v4 inventories the completed 1,024-reading Pocket continuation; wordlab_pocket_matrix_v4 inventories verified word/prosody features; pocket_crossed_models_v4 inventories three fitted weak reference heads, selection and held-out source-role results. pocket_native_validation_v4 and wordlab_pocket_native_v4 record the completed native-control failure. These heads remain failed baselines, not deployment candidates. pocket_switch_v4 retains the concrete switch/reliability plan and verified original reference identities. Its cancelled_input_cleanup.json, when present, records retired GPU model/reference inputs; model restore remains pinned, no unique source audio is deleted.

human_g2p_selection_v5 records real-word selection; human_g2p_evidence_v5 inventories 232 real recording extractions and the resolved controller setup incident; human_g2p_summary_v5 retains 264 planned-word observations and within-source-group summaries. Twenty of 32 families have representable competing paths. No native acceptability, preferred-style or Hindi-error accuracy exists. data/human_regions_v5 and data/human_g2p_bank_v5 own the actual metadata and selection manifests. review/human_word_data_v5 owns the compact timed listening cases; original corpus WAVs are referenced in place, not duplicated.

Current raw experiment/controller logs are disposable within 24h of completed-result review; keep compact failures, source snapshots, manifests and conclusions. No automated cleanup job exists. The older wordlab_progress_v3 snapshots are historical; the live /api/current-run and active_goal.json carry current status. Native annotations are independent and never overwritten.

Original-audio grounding, 2026-11-07: human_staging_audit_v5 inventories the two actual preparation sources, preserving unknown historical authorship/execution revision. data/human_original_audio_pilot_v6 retains a resolved strict-mono setup failure, actual stereo format inspection, timing diagnostic and two byte-identical native recipe reconstructions. data/human_original_audio_v6 is the expanded selected-original recovery with explicit resume/source-amendment evidence and channel-review flags. verification_human_v5, when present, records current code/HTTP/browser/storage checks; human_review_cleanup_v5 records removed owned QA output. Owner Ampixa/user; active detector data-quality consumer; review by 2026-11-14. Compact metadata copied to cdjk; raw source payloads remain on k2 and are not new human accent labels.

## Boundary audit and auxiliary intervention components,2026-11-07

human_boundary_audit_v6 retains fixed-window audits for232 human clips/264 planned-word occurrences; its result records132 consistent,28 sign-changing,103 unsupported and one incomplete/tied occurrence.161 baseline scores reproduce exactly. human_boundary_smoke_v6 records the two-clip baseline check that preceded it. g2p_scope_audit_v6 records the frontend README/cited-reference discrepancy and prospective public annotation sources; no inaccessible IPA audio or LDC sample payload was retained or used for training. review/human_word_data_v6 is the current62-case listening data with per-recording sensitivity warnings; v5 is superseded for serving and can be removed after dependency/retention review.

intervention_acoustic_check_v7 inventories the eight-control independent phone sanity check: four paired requests order in the intended direction, though the zero threshold favors deletion for all eight. intervention_features_v7 inventories960 verified feature caches, without class/voice/decoder timing as predictors. intervention_heads_v7 freezes choices/weights before reserved-feature reads and records the two-component synthetic-request evaluation. intervention_direct_smoke_v7 records cache/direct numerical agreement; intervention_direct_latest_v7 checks the subsequent exception-boundary refactor. These compact checks remain current consumers for reproducible direct inference, not additional independent accuracy tests. Actual Hindi/native/prosody accuracy remains unvalidated.

Auxiliary renderer/runtime: models/kala_controls contains only the verified77.7MB ONNX and small pinned config/card files; runtime/kala_controls_README.md/freeze/environment.json identify the isolated runtime and actual native APIs. data/schwa_interventions_v7 owns the frozen48-word/5-voice960-control plan and render receipts. No935MB training checkpoint or new HF upload. Model CC-BY-SA4.0; native helper MIT; encoder/alignment licences unchanged. Owner Ampixa/user, active detector component consumer, next review2026-11-14. Task-owned package cache and completed logs are disposable after import/model verification; required source, receipts, selected model weights and active original-audio work remain intact.

2026-11-07 real-word transfer: real_word_transfer_v8 and real_word_transfer_summary_v8 retain complete1928-clip/1844-word evaluation, crop abstentions, source snapshots and per-word human/style comparisons; not error-label accuracy. ui_transfer_v8 holds compact browser/API outcomes; screenshots and transient snapshots are deleted after QA with hashes. human_matched_v9 holds exact-human-context generation/controller receipts. Data/evidence groups have dedicated provenance. review_cleanup_v8 and review_v6_retirement_v8 retain exact recovery proofs and measured deletion receipts for superseded UI copies.

Current2026-11-09: task_vowel_release_v31 selected head/review; quality_followup_summary_v38 and release_quality_gap_audit_v38 enlarged heldout quality audit; release_cold_verification_v43 final8 CLI checks; query_guard_verification_v44 final5 guard checks; swar_content_audit_v41 all258 lexical checks; coverage_and_prosody_v46 exact coverage/next steps; release_handoff_v39 current runtime, copies, HTTP/browser/storage. New512 quality renders evaluation-only. Descriptive Native observations do not change predictions. Actual Hindi-error/intonation accuracy unvalidated. Obsolete bulk storage review records10.6GB unused audio and blocked cleanup;0 files removed/space reclaimed. Owner Ampixa/user, creator Codex GPT-6 /root,next review2026-11-10. Compact evidence retained while model active, raw diagnostics expire24h after consumer closes; no HF publication.
