# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/reference_word_v3`.
Creation and last review (UTC): 2026-11-07T08:54:24.988239+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Reference-conditioned pronunciation/prosody evidence; evaluate whether same-word differences improve the weak baseline.

## File inventory

- `models.json`: 282487 bytes; SHA-256 `d88b2ed1e2855c11fe24ecb54ac63f3b478fe08995d591bcca70e4ff7ba56999`.
- `report.json`: 6498 bytes; SHA-256 `011cfdc529fc656b1c7bcfc50010707ca999bf4c6cca2fa6f563752ad03426eb`.
- `selection.json`: 2665 bytes; SHA-256 `2e43b689fd4672b1cf6beb586b5031b27b6738f31b3fc438214c842412c9158e`.
- `word_predictions.json`: 169142 bytes; SHA-256 `3a05af6a1312bf05cbe2cdbaddc818358ad528bfb906c5dcfff9181bf8255ab2`.
- `source_snapshot.json`: 24956 bytes; SHA-256 `a8d219ec3e7e5d18972d27720a0912675501e3b8aadfb08c5c92492bc4828d68`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Design SHA 91c376d447268b93ed3efec884a7c630e00e829070017be3846d8c2fec399946
- /Volumes/T9/hindi-detector/reports/wordlab_training_v3
- /Volumes/T9/hindi-detector/reports/wordlab_test_v3
- python -m scripts.train_reference_words --training reports/wordlab_training_v3 --test reports/wordlab_test_v3 --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
