# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/swar_content_audit_v41`.
Creation and last review (UTC): 2026-11-09T01:25:32.764007+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Input-content audit of all actual86 same-text triples. items.jsonl exact WAV/text, three recognizer hypotheses and WER/CER/roman-CER with pass/quarantine/reject; runtime.json verified recognizer/source identities; report.json source counts and known examples. No pronunciation/source labels, new waveform or model copies; no training or detector thresholds changed. Current consumer misalignment/garbling guard and uncertainty; review2026-11-10. Keep compact content receipts with release, transient logs expire24h after consumer closes. NOT on HF; inherited source rights.

## File inventory

- `protocol.json`: 369 bytes; SHA-256 `dadd88f971302d40a2bc1966867d19f0b833af16003ad6f8edafba6ae84f45df`.
- `source_snapshot.json`: 13241 bytes; SHA-256 `892a68815b8090276eff9c57b256e6c94ccb38f50eecc190f2daccbc0feacfc1`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-asr-gate/bin/python -m scripts.audit_swar_content

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
