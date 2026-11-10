# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/real_word_transfer_summary_v8`.
Creation and last review (UTC): 2026-11-07T16:13:59.622224+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Measure actual-word coverage, crop sensitivity, observed human score support and text-matched Pocket style without inventing Hindi/prosody gold.

## File inventory

- `report.json`: 78613 bytes; SHA-256 `ea24c67e70432307e270c3743a4219b741338e0aff12b25faf3b60e3d2958829`.
- `comparisons.json`: 7710172 bytes; SHA-256 `a2b49d03254de4706f4c443a77253d49648df92cdd8b4ba419ff3a1b9440f0ac`.
- `source_snapshot.json`: 21837 bytes; SHA-256 `ef005e62c5730b19863e6d0cfd16cffe254515d727286947dc172bfaac627841`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/reports/real_word_transfer_v8
- Application items SHA bc92ad35bf12291d0a67a0ccc6dae43376d64ad9bfd783df726cc98b9d6710c1
- .venv-indicf5/bin/python -m scripts.summarize_real_word_transfer --application reports/real_word_transfer_v8 --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
