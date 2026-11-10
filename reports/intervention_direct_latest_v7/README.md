# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/intervention_direct_latest_v7`.
Creation and last review (UTC): 2026-11-07T15:27:40.257949+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Direct isolated-word inference for the experimentally trained schwa intervention components; preserves unavailable paths and unknown Hindi/native/style judgments.

## File inventory

- `analysis.json`: 2286 bytes; SHA-256 `9cba62e1ad390898486824afb232329804c283144ef633c149e6b3c42ef12065`.
- `source_snapshot.json`: 28195 bytes; SHA-256 `1c2629f704f1d64024a01e398c0f3a84e6b7ea20389619fc29719f6b2ae85fd7`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/data/schwa_interventions_v7/render/cf7_6d4573a167113a5bb6d463b8.wav
- /Volumes/T9/hindi-detector/reports/intervention_heads_v7/selection/models.json
- .venv-indicf5/bin/python -m scripts.score_word_schwa --audio WORD.wav --candidate-id schwa-ID --models MODELS --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

reproduction.json records numerical agreement after the explicit exception-boundary refactor; its current consumer is latest direct-inference validation. It is not a new independent accuracy test.
