# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/anchored_training_v71`.
Creation and last review (UTC): 2026-11-09T08:16:28.911323+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Reviewed large local-deletion fitting. model.pt one selected linear head, train-only normalization, hardest-family dev threshold, full code/resource/encoder/aligner pins. selection.json all train-only CV candidate losses; cv_predictions.json exact out-of-fold physical scores and membership; fitting_rows.json metadata only; report.json measured/attempted source/family counts; frozen_selection.json model/threshold/protocol identity before any test open. No epoch checkpoints or raw WAV/encoder copy. Owner Ampixa/user, review2026-11-10; retain selected head/compact evidence while release/test consumes it. Not on HF; source/MMS research rights inherited. Physical-edit CV/calibration is not real Hindi-error accuracy. Every missing post-edit observation remains in data reports.

## File inventory

- `protocol.json`: 2393 bytes; SHA-256 `0dfb01c0f29e0ad42d97462f7066ea70fd0177edc77e8b37d596d3137c22eeac`.
- `source_content_exclusions.json`: 4840 bytes; SHA-256 `7c43d29ab84a0d98b4f4a1e7a85b3127cfd968c0967147a56ea9d8dd4c7b7534`.
- `source_snapshot.json`: 21003 bytes; SHA-256 `f2bf41890c5cf96fd0b93f22c89d608d17ecb764e28c739eae95ad03c7ebed23`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.train_anchored_detector

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

External verified copy on cdjks-MacBook-Pro-2.local at /Users/cdjk/github/hindi-detector/reports/anchored_training_v71; external_freeze.json records byte hashes before any held-out score. Source original k2:/Volumes/T9/hindi-detector/reports/anchored_training_v71. Owner Ampixa/user; active test consumer, review2026-11-10; NOT on HF.
