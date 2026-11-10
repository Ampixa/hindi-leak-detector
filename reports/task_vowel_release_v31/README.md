# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/task_vowel_release_v31`.
Creation and last review (UTC): 2026-11-08T23:52:12.866578+00:00. Status: active.
Next review: 2026-11-15.

Purpose and current consumer: Morning release candidate: one checkpoint packages deletion and vowel-quality heads over shared dual phone evidence. release_policy.json explicitly records use of the two development regression labels for release rollback, not gradient training. No word-specific inference branch. model.pt selected weights/normalization/dev thresholds/pins; report.json retrospective test evidence. No encoder/audio copies or HF upload. Current consumer actual Swar scoring/cold/causal validation; owner Ampixa/user, review2026-11-09. Keep current selected model and compact evidence; inherited source/MMS research rights. Broad real Hindi-error and intonation accuracy remain unestablished.

## File inventory

- `release_policy.json`: 1035 bytes; SHA-256 `1b7a281b42b84b2ff2772abacf78454efd8727e8970f7e88c6d0a972930fe5ab`.
- `source_snapshot.json`: 6548 bytes; SHA-256 `1b8b659adb9a94453f108fc182bdef59f75185649c8715875a53a498d371e07e`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.build_vowel_release

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

review_report.json / review_additions.json / review_provenance.json: current private review metrics and compact measured IPA/phone-focus additions, exact sizes/input/member hashes in review_provenance.json. No audio duplicate. Review2026-11-09; retain while this model is current.

review_amendment_v36.json documents unchanged model, new teacher-reference audit (14 human/21 Swar flags across86 cases),247/1368 supported word-contrast coverage and amended review_report.json SHA. Original review_report SHA retained, not byte copy. Human flags are variation/reference disagreement, not gold errors. Review2026-11-09,next2026-11-10; current user review consumer.

native_observations.json is descriptive observed real-speech variation, never a prediction/training input or acceptability label. native_observation_provenance.json inventories exact new bank/additions hashes and source. Review2026-11-09,next2026-11-10; retain compact context while current user review/inference consumes it. No waveform copies, NOT on HF, inherited source/MMS rights.

review_finalization_v45.json inventories final pooled-quality review/counts,8/8 cold and5/5 guard checks, independent lexical-content audit, per-site teacher-reference corroboration and priority_flags.json. priority flags remain hypotheses except the two user-reported development regressions. Descriptive observed human context never changes predictions. All file/input/source/page hashes retained. Model bytes unchanged; review2026-11-09,next2026-11-10. Current consumer private morning model review; retain compact evidence while model consumed, transient diagnostics expire24h after consumer closes.

2026-11-09 v56 review amendment:4 explicitly corroborated numeric cases added,90/96 scored,21 Swar candidates/9 two-teacher-supported, failed prosody heads not promoted. Compact rollback, exact source/result receipts and target SHA in reports/review_amendment_v56. No model/threshold/source-WAV changes. Review2026-11-10.

Final delivery search metadata amendment: reports/review_delivery_v57 pins exact updated review_report.json and HTML; original numeric text is searchable and available in an expandable disclosure. No prediction change. Review2026-11-10.
