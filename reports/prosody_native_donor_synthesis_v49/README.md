# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/prosody_native_donor_synthesis_v49`.
Creation and last review (UTC): 2026-11-09T02:33:09.365144+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Owned best-teacher rendering controller. status.json records actual process/phase; smoke_launch.json/bulk_launch.json pin remote command/PID. report.json counts actual outputs for the exact frozen selection. Remote directory is an explicitly documented mirror, no model-weight copy. Content/phonetic reference suitability is a later separate audit, not assumed from finite WAVs. Review2026-11-09, current consumer automatic-reference detector validation; logs disposable24h after consumer closes. No HF upload/publication; source model/corpus licences apply.

## File inventory

- `inputs.json`: 436 bytes; SHA-256 `56c9425d9004c3ea1ef9f4bdd03da13465c40155d86bb51720be03f0e75d3170`.
- `source_snapshot.json`: 10543 bytes; SHA-256 `e00018c85a10de6ffda76d22ffd6b19cf12a3a367aaf22410baeef0ceaf75d54`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.run_swar_teacher_validation
- Original correct24-layer teacher CUDA adapter, unchanged validated source and exact shared prompt bytes

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. Generated waveforms, if present, are explicitly described above and listed in inventory_v57.json.
