# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/wordlab_direct_smoke_v3`.
Creation and last review (UTC): 2026-11-07T09:01:40.669064+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Direct WAV word/prosody inference; exposes timing and separate evidence without inventing word gold.

## File inventory

- `analysis.json`: 32129 bytes; SHA-256 `331bf977265caeff0ae623f6601b3a16c07e8dd0fdf79d264b1798a690cb1816`.
- `source_snapshot.json`: 24853 bytes; SHA-256 `6c0638a61a299bc4b6b2d1440db30732ffd7e66217d8e94a847b96b2bff03936`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/data/indicf5_audio8_v1/audio/aa03252c11ff6089a4c2ec9e.wav
- /Volumes/T9/hindi-detector/reports/wordlab_models_v3/models.json
- python -m scripts.analyze_word_audio --audio mono.wav --text ... --models reports/wordlab_models_v3/models.json --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
