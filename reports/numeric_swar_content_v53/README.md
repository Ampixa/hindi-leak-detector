# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/numeric_swar_content_v53`.
Creation and last review (UTC): 2026-11-09T02:52:10.723718+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Lexical-content audit of all10 numeric-containing Swar cases previously unscored. items.jsonl exact source WAV SHA and three actual ASR hypotheses. runtime.json model/source pins. report.json completion; no text expansion/repair or Hindi-error labels automatically inferred. Candidate spoken forms require explicit corroboration across Native and Swar readings; disagreements stay unscored, never guessed from numeric spelling. Consumer widen review coverage safely, review2026-11-10. Compact receipts retained with task; no new audio/model copies/HF upload; inherited rights.

## File inventory

- `inputs.json`: 10716 bytes; SHA-256 `1308925d748d208cf15083cfa577a258fa1de7964064208fda30df3d15ec722b`.
- `source_snapshot.json`: 3316 bytes; SHA-256 `a22621b8bb1e1ce251a4393ea8e55f47809f7f0d62e7922ee60d2126bd911c69`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-asr-gate/bin/python -m scripts.audit_numeric_swar_content

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. Generated waveforms, if present, are explicitly described above and listed in inventory_v57.json.
