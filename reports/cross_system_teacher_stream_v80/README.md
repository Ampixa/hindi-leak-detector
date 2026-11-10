# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/cross_system_teacher_stream_v80`.
Creation and last review (UTC): 2026-11-09T09:07:19.680007+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Explicit transport recovery after repeated L40 destination-reserve failures. Approved unchanged CUDA teacher reads existing pinned model/prompts; WAV encoding stays in memory and travels over SSH to k2, where the original20GiB destination guard remains. No reserve lowered and no storage deletion. previous_failure.json exact old failure; inputs.json hashes/budget; worker.log remote stderr; launch.json owned SSH process; runtime.json exact unchanged engine identity; parity.json verifies first old WAV byte-for-byte before any new output; report.json full verified result. No checkpoint/source replacement. Teacher audio remains under original plan with per-row transport provenance and streamed completion in run.json; original lock remains and is not misrepresented as covering the new transport code. Owner Ampixa/user; review2026-11-10; current consumer fresh cross-system listening. NOT on HF, inherited rights; logs disposable24h after consumer closes.

## File inventory

- `previous_failure.json`: 715 bytes; SHA-256 `7178a1650a586d1e9df8b74ffa6cf34212c6086f4b8c8c7528a1e90b595ccc41`.
- `inputs.json`: 503 bytes; SHA-256 `d228fdad96a2a658bba90e2c1094e283ef20233fdd5c504ec38ac6d34e650da2`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.receive_cross_system_teacher

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
