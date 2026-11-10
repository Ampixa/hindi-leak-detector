# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/data/human_g2p_bank_v5`.
Creation and last review (UTC): 2026-11-07T11:42:26.594033+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Human word-realization candidates sampled across reported regions using two explicit G2P predictions; no accent labels inferred.

## File inventory

- `selection.json`: 323187 bytes; SHA-256 `f1a1be6d2db7e27ae90b75f5cd5d90210a46b2486b7681225708585e2682794d`.
- `word_bank.json`: 110421 bytes; SHA-256 `056edc34c0a1788b057c983d1697b35f5504970fd9463e03dead3b1712087e8d`.
- `report.json`: 23513 bytes; SHA-256 `db76805226337d31b1dbd2e804a371edfe059c8f082d66b834aedbc18539112f`.
- `g2p_runtime.json`: 1750 bytes; SHA-256 `39b19bb66c59494267835c9dbe50e7a0d44e8e138e156211dffddfa53048789d`.
- `source_snapshot.json`: 14524 bytes; SHA-256 `9e7dab5f61e9f9facf2663697d61bbde486b700bb7a43be3823a4a76838b4d89`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Human metadata inventory SHA e70ebb4ff5ceb91bc5f09d322b6568c1ca3c58eeae91c633b4380cc39b293989
- sources.lock.json
- Existing /opt/homebrew espeak-ng 1.52.0; no installation or new model download
- python -m scripts.select_human_g2p_bank

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
