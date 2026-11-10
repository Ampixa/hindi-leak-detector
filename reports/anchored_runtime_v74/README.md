# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/anchored_runtime_v74`.
Creation and last review (UTC): 2026-11-09T08:20:44.513611+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Persistent runtime measurement on original Swar/real recordings. items.jsonl contains raw source hashes, exact text, word/site evidence, coverage and processing times; report.json contains cold initialization and sequential completed-utterance timing. No WAV/encoder copies. These cases influenced prior development and include listener feedback; they are not held-out test data. Owner Ampixa/user; review2026-11-10, current HindiBench prototype consumer. NOT on HF; source/MMS research usage restrictions inherited. Retain compact results while benchmark consumes them; no automatic accent/prosody accuracy claim.

## File inventory

- `inputs.json`: 922 bytes; SHA-256 `41aebc89f887bbdbddd601ab1e7e3f1be80f16777ee83fa2a87d9c02d49e8285`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.benchmark_anchored_inference --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
