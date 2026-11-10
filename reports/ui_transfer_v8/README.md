# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/reports/ui_transfer_v8`.
Creation and last review (UTC): 2026-11-07T16:18:26.356718+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Verify readable private v8 word/results UI. desktop.png/mobile.png are transient visual QA screenshots (if generated); api_checks.json and browser_checks.json compact semantic results will be retained. Delete images/browser snapshots within24h after QA; no extra audio copies.

## File inventory

- `qa_plan.json`: 172 bytes; SHA-256 `849b0908b25537ce751f9173a91c1c1f1427da26b85a78f861935b20d134e2d4`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- review/wordlab.html
- review/pocket-progress.html
- k2 review/human_word_data_v8

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Actual QA inventory: qa_plan.json plan; browser_checks.json semantic desktop/mobile/audio outcomes; api_checks.json measured live API snapshot; cleanup.json deleted screenshot/browser-snapshot paths, sizes and hashes. Visual QA completed; transient desktop.png/mobile.png were removed. Reproduce screenshots from current private website; original historical pixels are no longer retained.
