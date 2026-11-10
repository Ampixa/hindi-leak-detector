# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/human_boundary_audit_v6/result`.
Creation and last review (UTC): 2026-11-07T14:50:55.321847+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Quantify sensitivity of human dual-G2P comparisons to fixed word-window changes before calibration.

## File inventory

- `report.json`: 1798 bytes; SHA-256 `f703fb849f27191078f533d715857a6c4d272b28e8c86c34fdb381bf4d751351`.
- `word_summary.json`: 7671 bytes; SHA-256 `e315dfdcf1a8fbeac52a62883ba2b8f327bdae8117f596aac2f9afa3b31eee99`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Parent items.jsonl and source_snapshot.json
- /Volumes/T9/hindi-detector/data/human_g2p_bank_v5/word_bank.json
- /Volumes/T9/hindi-detector/reports/human_g2p_evidence_v5/manifest.json
- python -m scripts.audit_human_boundaries --bank data/human_g2p_bank_v5 --evidence reports/human_g2p_evidence_v5 --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
