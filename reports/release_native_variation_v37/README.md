# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/release_native_variation_v37`.
Creation and last review (UTC): 2026-11-09T00:55:23.755974+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Frozen release on the existing fixed102 human-word comparison pairs across18 families. Original full-frame/time coordinates and hashes checked; different source voice groups; observed disagreement is variation evidence, not native-error gold. No waveform/model/frame copies, no fitting or threshold update. Consumer native-variation-aware detector interpretation; review2026-11-10; retain compact counts/decisions while release is active. NOT on HF; inherited source/probe/MMS research usage restrictions apply.

## File inventory

- `report.json`: 4662 bytes; SHA-256 `b01e11bde263f0c2513af5ee38f1851a3f63af472a179ae9e5cf79e635abf343`.
- `comparisons.json`: 396135 bytes; SHA-256 `b0c377399ed84f36085f1b67670842e43d98b5c529ecbbc06c215d99310232a9`.
- `source_snapshot.json`: 6631 bytes; SHA-256 `71c0b433a3c3629ab530b524f10786d22d5492fea3f8dd2b1f7167494d20cb8d`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.audit_release_native_variation
- /Volumes/T9/hindi-detector/reports/native_word_variation_v22/comparisons.json

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
