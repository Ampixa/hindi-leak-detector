# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/short_vowel_localization_v75`.
Creation and last review (UTC): 2026-11-09T08:25:56.388530+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Train-only short-vowel localization versus phone-resolution audit after rejected frozen candidate. items.jsonl stores source/family identities, retained-interval geometry, phonetic peaks, actual anchor geometry and explicit diagnostics; report.json summarizes both probes. No new audio/encoder/model copies; no fit or threshold change. Owner Ampixa/user; review2026-11-10. Active consumer decide one next measurement experiment. NOT on HF, inherited research rights. Retain compact conclusions, raw diagnostic rows expire24h after consumer closes.

## File inventory

- `protocol.json`: 2158 bytes; SHA-256 `79bfcc41c9d59a16cfaccc0a2ad9f64b2cb419f9edb241a3791ae999ff456666`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.audit_short_vowel_localization

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
