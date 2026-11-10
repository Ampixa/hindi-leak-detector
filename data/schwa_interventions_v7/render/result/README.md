# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/data/schwa_interventions_v7/render/result`.
Creation and last review (UTC): 2026-11-07T15:11:57.963896+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Same-voice one-site controls with lexical and voice partitions frozen before rendering. All audio identities, input IDs and duration correspondence retained.

## File inventory

- `report.json`: 500 bytes; SHA-256 `4b6c1e0647e11adca3f24924519086ab64265a75f44ade5cebbbdcc8bdfa08df`.
- `receipts.json`: 1640717 bytes; SHA-256 `a5a1489768fc3575fc4cd1f4a5d1886358237bead8007bdba4d7ef7698f3f786`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Parent plan.json, source_snapshot.json and items.jsonl
- .venv-kala-controls/bin/python -m scripts.run_intervention_pack

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
