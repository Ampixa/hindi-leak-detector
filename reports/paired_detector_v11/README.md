# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/paired_detector_v11`.
Creation and last review (UTC): 2026-11-08T01:45:09.907256+00:00. Status: active.
Next review: 2026-11-15.

Purpose and current consumer: Paired reference-conditioned word-source candidate. model.pt is the single selected final checkpoint; history.json is compact train/dev fitting history, report.json and predictions.json are frozen test results, status.json phase/PID. No human accent labels. Synthetic localized feature replacements have known source identity but are not physical waveforms. Focal lexemes/texts/conditioning disjoint; incidental contrast words excluded from negative supervision. Random hidden projection is pinned inside model. Keep final model and metrics until superseded; no additional epoch checkpoints retained.

## File inventory

- `inputs.json`: 12795618 bytes; SHA-256 `31a8e6cca7076bdbf5f97d4487a52958e54a576459d65489f765947d0cb17464`.
- `source_snapshot.json`: 31227 bytes; SHA-256 `9c32e1528b2fd34bc7164a80bbf87d891da363c6a357535cb70c6bd6da71f8c6`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.train_bulk_detector --features GROUP --out NEW [--wait-complete]
- Original WAV and feature identities in frozen_receipts; native alternative pronunciations remain valid.

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
