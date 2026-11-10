# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/physical_content_v12`.
Creation and last review (UTC): 2026-11-08T04:55:05.608226+00:00. Status: active.
Next review: 2026-11-15.

Purpose and current consumer: Post-edit content audit of held-out physical controls with the already-fixed recipe and model. items.jsonl stores decoder outputs/decisions, runtime.json exact ASR. report.json retains all decisions and paired-pass-only source metrics. No retraining or waveform deletion. ASR is never an accent label.

## File inventory

- `inputs.json`: 509 bytes; SHA-256 `f723fa3de07c1fe273399b86fd37367ed0669627fd4787abcd8a68b7870bd75d`.
- `source_snapshot.json`: 4491 bytes; SHA-256 `d59346036a4460d916ed5a445447954c02fe5c330ff584b63ed68f44d228a2e1`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-asr-gate/bin/python -m scripts.audit_physical_content --group reports/physical_detector_v12 --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
