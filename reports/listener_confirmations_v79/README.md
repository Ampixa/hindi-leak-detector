# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/listener_confirmations_v79`.
Creation and last review (UTC): 2026-11-09T08:45:34.163461+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Record explicit user confirmations of हुनलाई, चलाउनको and नगरौ. feedback.json exact quote and waveform/word identities; previous_word_checks.json and previous_examples.json compact metadata rollback; inputs.json source/checkpoint hashes; applied.json verified after hashes and unchanged model. No audio, model, threshold, training or held-out-test changes. These are listener-confirmed discovery occurrences, not independent accuracy. Owner Ampixa/user; review2026-11-10, current consumers listening review and announcement draft. NOT on HF; original speech rights apply. Retain compact annotation provenance while confirmed examples are used; rollback snapshots disposable24h after review closes. Known copies Mac and k2.

## File inventory

- `feedback.json`: 1698 bytes; SHA-256 `d507ab99f5ea67ad1ea59aeb84f3fa097c897d07cfe43a6d4a857392ec1e717e`.
- `previous_word_checks.json`: 4013 bytes; SHA-256 `261199d7e5ccf8e94c53ef0a8a2c64f382b7efcf902adf07c1ac48d83892a148`.
- `previous_examples.json`: 6023 bytes; SHA-256 `a1e39fce2090e88f73a3303d462f822e7bafefc8530c7ad332f8e4eeaf65c21f`.
- `inputs.json`: 416 bytes; SHA-256 `8120572be8bb100a2fd4610dc3a280d3e5fdbe2f7dddbdcdd2b8c618dfb717c0`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.record_confirmed_discoveries

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
