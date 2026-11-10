# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/reports/reference_word_v3_design`.
Creation and last review (UTC): 2026-11-07T08:53:44.525921+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Lock the reference-conditioned word experiment before fitting.

## File inventory

- `plan.json`: 658 bytes; SHA-256 `91c376d447268b93ed3efec884a7c630e00e829070017be3846d8c2fec399946`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Current inspected 96 development + 96 benchmark word features
- python -m scripts.train_reference_words --training reports/wordlab_training_v3 --test reports/wordlab_test_v3 --out reports/reference_word_v3

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
