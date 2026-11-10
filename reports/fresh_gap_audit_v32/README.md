# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/fresh_gap_audit_v32`.
Creation and last review (UTC): 2026-11-09T00:05:57.661218+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Fresh task-compatible actual waveform edit audit after release freeze. controls/*.wav actual 16kPCM16 same-voice sham/change pairs, every filename/bytes/SHA/source and exact segment geometry in items.jsonl. That file also retains features/decisions for release and legacy baseline; exclusions.json every preflight exclusion, report.json actual counts/metrics/limits. No new weights, NOT on HF. Source copies remain documented in minimal_audit_v29/teacher. Current consumer release validation; review2026-11-09. Keep compact conclusions/source pins; raw controls disposable24h after consumer closes unless explicitly selected release evidence. Inherited rights.

## File inventory

- `protocol.json`: 1059 bytes; SHA-256 `ce2fd1a9e0d256b1adc6458b8fcd06190f6c888a15c6ce7c35122323f8de67fb`.
- `source_snapshot.json`: 19807 bytes; SHA-256 `0b279a8db2d8cafd40efd3b2393fe56b0c51cb97b086964d19bfdbfe2c89c4a8`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.audit_fresh_waveforms

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
