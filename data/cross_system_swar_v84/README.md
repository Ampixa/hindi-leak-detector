# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/data/cross_system_swar_v84`.
Creation and last review (UTC): 2026-11-10T03:53:20.291513+00:00. Status: active.
Next review: 2026-11-17.

Purpose and current consumer: Swar (Aarjan/swar@8ca6f1b podvoices_ema.pt) renderings of the frozen v80 plan: 26 texts x 4 shared prompts. items.jsonl receipts (plan text, Swar frontend model text, WAV SHA/bytes/duration or explicit quality rejection); audio/*.wav 24k PCM16; runtime.json; run.json. Weights CC BY-SA 4.0; audio NOT on HF. Created by Claude (claude-opus-5-5) Claude Code session. Status active; consumer reports/cross_system_swar_v84; review 2026-11-17; delete audio within 24h after that consumer closes, keep receipts. Reproduce by rerunning the command.

## File inventory

- `lock.json`: 597 bytes; SHA-256 `6034e9c09870d390f05971efce8c0ea4513625206d3a317931df69d5544d0f0c`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.run_swar_cross_system

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
