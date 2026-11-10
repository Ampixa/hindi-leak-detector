# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/rhythm_transfer_training_v52`.
Creation and last review (UTC): 2026-11-09T02:52:58.892607+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Rhythm-only40->8->1 candidate, measured emitted MMS word durations and relative pauses; no pitch features. Prosody-only source-style candidate trained on known physical transfers with identical Native carrier audio for both labels. model.pt stores selected state, training-only normalization and frozen threshold; frozen_selection.json proves model identity/selection before reserved score loading. history.json training/dev losses; report.json held-out known-intervention metrics and nuisance controls, scores.json per-context scores. Inputs pin distinct word/conditioning-group partitions. No phone features, voice embeddings, speaker/system IDs or word IDs enter network. NOT a validated Hindi-error detector, never automatically promoted to live vowel model. Consumer causal prosody admission experiment, review2026-11-10. Keep selected state and compact metrics; no historical checkpoints/WAV copies/HF upload; inherited rights.

## File inventory

- `inputs.json`: 4397 bytes; SHA-256 `2b08a743c3cadfbb3b49ed30ed49746b0ec196da8128bcdaad41c6272932737f`.
- `source_snapshot.json`: 14254 bytes; SHA-256 `092243327da2ca1681ab1f5767c28562bfd2542f99d54c352d9e28b995a6189d`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.train_prosody_transfer --rhythm-v52

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. Generated waveforms, if present, are explicitly described above and listed in inventory_v57.json.
