# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/cross_system_swar_content_v84`.
Creation and last review (UTC): 2026-11-10T04:01:31.834758+00:00. Status: active.
Next review: 2026-11-17.

Purpose and current consumer: Lexical-content checks (Nepali Zipformer, Nepali IndicConformer, Hindi audit) on every Swar v84 render, same gate as reports/cross_system_content_v80. items.jsonl per-waveform hypotheses/decision; report.json counts. Created by Claude (claude-opus-5-5) Claude Code session. No pronunciation labels. Status active; consumer reports/cross_system_swar_v84; review 2026-11-17.

## File inventory

- `inputs.json`: 266 bytes; SHA-256 `e9574de77acbacb41ae9fd7e956b97364bdf34af61005a8a8d90cee3785c83ea`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-asr-gate/bin/python -m scripts.screen_swar_cross_system

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
