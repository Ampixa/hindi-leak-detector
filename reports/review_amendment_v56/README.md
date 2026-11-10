# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/review_amendment_v56`.
Creation and last review (UTC): 2026-11-09T03:01:22.308516+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Concrete read-only review amendment:90/96 cases and21 candidate word flags, two known development errors and9 two-teacher-supported candidates unchanged. Four number expansions preserve original text; one Audio8 reading omitted without numeric corroboration. Failed prosody heads displayed and remain unserved. previous_* files are compact rollback state for exact prior review/priority reports and only the four changed cases (no55MB full-case backup). applied.json target SHA. Restore those four cases by id and previous reports to reverse; model/service unchanged. Active consumer review provenance/rollback, review2026-11-10; retain compact rollback while amendment consumed, no audio/model copies/HF upload.

## File inventory

- `previous_review_report.json`: 69535 bytes; SHA-256 `1adc8c38f4e54c2b96cbc4543e07117cece21fcbbbe22ce819ccd815ce9ff602`.
- `previous_actual_report.json`: 2461 bytes; SHA-256 `16686a316e8d966d9cc5b449761938f078f434436e7f435d5140eaf4b1fafd8a`.
- `previous_priority_flags.json`: 36841 bytes; SHA-256 `e5744b5e04fec1f857d04c34bb7a79e5a28eac696b83b5b350e744946bd4f587`.
- `previous_numeric_cases.json`: 6638 bytes; SHA-256 `1e966ea38fbf2f41edbd0c2d304c74524e42f4384a7cf30abb1c47aa32cfb47a`.
- `inputs.json`: 822 bytes; SHA-256 `a933aa104ce242be7e94e7b3fa14de99dd55714e9f24dc28c99d6ca50b633f72`.
- `source_snapshot.json`: 12801 bytes; SHA-256 `4ffdfdae915bcdd52d018b5374a39966041045ec2b65196c00d8ef70e5f97105`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.amend_swar_review

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. Generated waveforms, if present, are explicitly described above and listed in inventory_v57.json.
