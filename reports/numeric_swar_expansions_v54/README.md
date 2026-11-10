# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/numeric_swar_expansions_v54`.
Creation and last review (UTC): 2026-11-09T02:55:25.206787+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Explicit four-case spoken-number expansions corroborated independently by both Nepali decoders on real and Swar audio. plan.json preserves exact original text, replacement occurrences and audit rules; six ambiguous cases remain excluded. Each id/source directory contains cold model inference result/source receipts; id_source.log child stdout/errors; completed.json final summary. Existing audio, page source text, head weights/thresholds unchanged. Comparison uses real same-text audio, not an unverified generated reference. A source without corroborated number reading is omitted explicitly. No new Hindi-error labels or accuracy estimate. Consumer broaden actual word review, owner Ampixa/user, review2026-11-10; compact metrics retained with active review, logs expire24h after closure. No HF upload; inherited rights.

## File inventory

- `plan.json`: 4187 bytes; SHA-256 `8d7b1728a670309bc0b56ef220998c96469fae5a656f5b537fb29474371439a5`.
- `inputs.json`: 278 bytes; SHA-256 `63c5b1e0ad85c1cfb996284c2975ff6a7d325cbdac57b18dea3dc61e35efbb8e`.
- `source_snapshot.json`: 7784 bytes; SHA-256 `96ea9c8fc1e7d45f196d9b395727b6331997087f65c9569cd331e3836d52dc25`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.score_numeric_swar_expansions

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. Generated waveforms, if present, are explicitly described above and listed in inventory_v57.json.

teacher_check_launch.json/teacher_check.log verify the newly flagged तिरेर against two automatic correct24-layer Native references. Separate reports/numeric_teacher_check_v54 owns actual result; this is corroboration, not an error-gold label.
