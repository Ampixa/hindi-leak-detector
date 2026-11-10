# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/task_vowel_release_v31/actual_swar`.
Creation and last review (UTC): 2026-11-08T23:55:09.769169+00:00. Status: active.
Next review: 2026-11-15.

Purpose and current consumer: Frozen stronger paired model applied to actual Swar corpus after held-out evaluation. cases.json has all exact source/word intervals/local phonetic evidence and decisions; report.json two user regressions/correct same-word check/coverage, no invented labels. No new waveform/encoder copy. Current consumer morning release decision, review2026-11-09; keep compact conclusions while selected model is active. NOT on HF; inherited source and MMS research licence.

## File inventory

- `report.json`: 2461 bytes; SHA-256 `16686a316e8d966d9cc5b449761938f078f434436e7f435d5140eaf4b1fafd8a`.
- `cases.json`: 58220087 bytes; SHA-256 `3ed729f2888194c85fdf209cb1e26557fe663713f89cfe336ef5fb9755b51bfa`.
- `source_snapshot.json`: 7915 bytes; SHA-256 `c1600b1fdd1dc2f3f55e2f998db7bc7c2945421c28abfecfd65623cd19cd1c47`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.apply_paired_local

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

2026-11-09 v56 review amendment:4 explicitly corroborated numeric cases added,90/96 scored,21 Swar candidates/9 two-teacher-supported, failed prosody heads not promoted. Compact rollback, exact source/result receipts and target SHA in reports/review_amendment_v56. No model/threshold/source-WAV changes. Review2026-11-10.
