# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/release_cold_verification_v31/real_yt_yt04_030_teacher_cache`.
Creation and last review (UTC): 2026-11-09T00:06:56.691772+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Cold generic release inference on actual audio/text, explicit or cached verified teacher references. result.json exact inputs/model hashes, per-word/per-reference phonetic evidence, qualifications and conservative consensus. No fixed-case prediction or word-specific threshold. No raw waveform or encoder copies; NOT on HF/inherited source/MMS research rights. Current consumer verification and user inference; review2026-11-09, retain compact result while current model is consumed.

## File inventory

- `result.json`: 1105694 bytes; SHA-256 `cc6fb43b42079184ee6b9b1a19e791f648f91b16aae88883015efe9b14e46eed`.
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
