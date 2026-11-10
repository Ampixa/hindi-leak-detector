# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/cross_system_swar_v84`.
Creation and last review (UTC): 2026-11-10T04:02:18.025640+00:00. Status: active.
Next review: 2026-11-17.

Purpose and current consumer: Frozen v31 detector outputs for Swar podvoices_ema on the 26 v80 texts x 4 prompts; references are the same two other-voice v80 Pocket renders. cases/*.json per-text readings with words/sites/per-reference scores; report.json denominators comparable to reports/cross_system_results_v80/report.json. Flags are candidates, not confirmed errors. Created by Claude (claude-opus-5-5) Claude Code session. Status active; review 2026-11-17. NOT on HF.

## File inventory

- `inputs.json`: 468 bytes; SHA-256 `17f58da6d3b28ca8ddb7b79675a4d2e56d86f19830f37cefa3a407a6f1e62d80`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.score_swar_cross_system

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
