# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/k2_prosody_v1`.
Creation and last review (UTC): 2026-11-06T07:48:06.192350+00:00. Status: active.
Next review: 2026-11-13.

Purpose and current consumer: Prosody smoke check; descriptive only.

## File inventory

- `features.json`: 1405 bytes; SHA-256 `e7e20f9a0dde0cfe6474aa49a732188c4725225469022b0ace80883bb719dcf4`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/hindi_detector/cli.py prosody data/g00005.pocket.16k.wav data/g00005.audio8.16k.wav --out reports/k2_prosody_v1
- /Volumes/T9/hindi-detector/data/g00005.pocket.16k.wav sha256=fcee5d4ce0c4fdb26c5d16caa974da6bb007cc5ea7cda1e9c0bf5a6e1ba317fd
- /Volumes/T9/hindi-detector/data/g00005.audio8.16k.wav sha256=f48a7d34838e5d2f1f375abd5b403d1b364263382f8da71d04d8b1b82482a2fc

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
