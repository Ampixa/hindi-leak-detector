# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/release_cold_verification_v31/real_yt_yt04_030_explicit`.
Creation and last review (UTC): 2026-11-09T00:06:46.519273+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Cold generic release inference on actual audio/text, explicit or cached verified teacher references. result.json exact inputs/model hashes, per-word/per-reference phonetic evidence, qualifications and conservative consensus. No fixed-case prediction or word-specific threshold. No raw waveform or encoder copies; NOT on HF/inherited source/MMS research rights. Current consumer verification and user inference; review2026-11-09, retain compact result while current model is consumed.

## File inventory

- `result.json`: 621175 bytes; SHA-256 `28847150dcb6c67b3ceccc3f88420e6e2ee1e86fde37c5004a3b333506bd8109`.
- `source_snapshot.json`: 17336 bytes; SHA-256 `5d8058bc28307ec2aa5cf9fcfc1b18e921290ef9b4f30823be0b151e4657187c`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.detect_nepali_leakage --query WAV --text NEPALI --out NEW [--reference SAME_TEXT_WAV ...]

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
