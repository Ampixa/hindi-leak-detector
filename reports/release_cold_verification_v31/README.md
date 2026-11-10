# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/release_cold_verification_v31`.
Creation and last review (UTC): 2026-11-09T00:05:56.466600+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Cold CLI checks. plan.json exact sources/text/target expectations before execution; each named subdirectory contains original CLI result/provenance and matching.log contains subprocess output. report.json expected development regression checks, batch/cold feature/time parity and unknown word processing. No raw audio/weight copies. Current consumer release promotion, review2026-11-09; keep compact conclusions, diagnostic logs disposable24h after consumer closes. NOT on HF/inherited rights.

## File inventory

- `plan.json`: 4654 bytes; SHA-256 `e6d74d245855d2bc23b390537f650fb9fdc10beaac023ac38f4484744defdbec`.
- `protocol.json`: 353 bytes; SHA-256 `e5d3b736823ce14a4ef84dec1f6b8f318271ae0ea1b29161b2c46abd6fd17279`.
- `source_snapshot.json`: 7792 bytes; SHA-256 `35b04faf26bc4f72ae0b40875e2cc027c7e02b72c401b2dcdb546907ca690466`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.verify_release_inference

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
