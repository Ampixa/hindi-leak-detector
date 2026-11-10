# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/data/human_matched_v9`.
Creation and last review (UTC): 2026-11-07T16:08:08.869083+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Freeze actual human utterances and separate TTS reference groups so the next comparison covers observed word realizations and identical-context prosody, not just artificial carriers. No additional native audio copies.

## File inventory

- `selection.json`: 234733 bytes; SHA-256 `3081371782f34c260d3f6e317f99890827339f3fcc08b2eb13cc8749b5ef783e`.
- `report.json`: 1740 bytes; SHA-256 `e1cb31560e4038262a730d8bf4e00ba8ea7efdee72a9d3e778f557e00de75f82`.
- `inputs.json`: 724 bytes; SHA-256 `e399add2018280c3bce3317decdc8ec268a119a035e299a4f685974154fe4aa4`.
- `source_snapshot.json`: 8715 bytes; SHA-256 `ac1d1cfc5caae46d2c30c596fb75a883ed4e3f0fc4efcf272c19fe2e1dca87f3`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Pinned original human identity report and selected payloads verified at planning time
- .venv-indicf5/bin/python -m scripts.build_human_matched_pack --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Local cdjk copy contains compact JSON/provenance only; WAVs and worker logs remain on k2. Live runtime fields here are snapshots; query the running service for current state.
