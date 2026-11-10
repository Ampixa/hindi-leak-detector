# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/release_voice_audit_v34`.
Creation and last review (UTC): 2026-11-09T00:14:23.895682+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Frozen release nuisance audit. items.jsonl exact original/control waveform hashes and measured site scores/decisions; controls/*.wav only8 new actual-error transformations, Native84 waveforms reused without copying from vowel_voice_null_v20. report.json gain/reconstruction/pitch-added effects and known-error robustness. No raw frame/model copy, NOT on HF. Current consumer release voice-confound validation, review2026-11-09; compact conclusions retained, diagnostic waves/logs disposable24h after consumer closes. Source/MMS research rights unchanged.

## File inventory

- `protocol.json`: 725 bytes; SHA-256 `cb3f10953617905e65ce33a98fe57eeea41d46d924d1e7921a7760c72a12ab17`.
- `source_snapshot.json`: 10074 bytes; SHA-256 `649a00a6a83636af5e59d29689a7f9d91720f53a83441385823241263bba79d6`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.audit_release_voice

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
