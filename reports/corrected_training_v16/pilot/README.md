# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/corrected_training_v16/pilot`.
Creation and last review (UTC): 2026-11-08T07:47:39.122067+00:00. Status: active.
Next review: 2026-11-15.

Purpose and current consumer: Selected structured candidate. model.pt stores fresh neural weights, train-only normalization and development operating points; history.json compact optimizer trace. report.json metrics distinguish physical known-source targets from original-TTS transfer; predictions.json actual word scores/times and site support. Source labels are not pronunciation correctness. No epoch checkpoints retained. Pilot is an incremental development diagnostic; final is selected after full corrected evidence. Current consumer corrected word inference/review; keep selected models and compact metrics, review2026-11-09.

## File inventory

- `inputs.json`: 6310764 bytes; SHA-256 `6948ce4fde983d0af367c60511e079b7dbc0ecd9027c1695551c19b8fe9dc92b`.
- `source_snapshot.json`: 30176 bytes; SHA-256 `50434633e20758a54787dfa53b3d0f8acc8432124134fc1886e18c532ac9f5da`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Fresh seed44; all complete corrected physical pairs in each selected partition; no old model weights.
- Inputs pin every query/reference/physical waveform and structured feature identity.

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
