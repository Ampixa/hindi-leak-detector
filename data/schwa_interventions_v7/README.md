# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/data/schwa_interventions_v7`.
Creation and last review (UTC): 2026-11-07T15:11:32.062088+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Create one-site same-voice controls with lexemes and voice IDs jointly held out. Exclude all current human-bank and original TTS focus words. Input intervention classes are not acoustic/native/Hindi gold.

## File inventory

- `plan.json`: 749818 bytes; SHA-256 `96c420b5d017631473cd6fd9d27fc4e3bc3a76a71b8ff6bc4db40e3df5e8f12f`.
- `report.json`: 2327 bytes; SHA-256 `c88a7a147cdad430a67e5c7611fa759d6bfb9e7713c81099015691480a74d27d`.
- `source_snapshot.json`: 6295 bytes; SHA-256 `97f07332dda6703839328205c3adb6dc972ff4531ce65d96e9f40592a1fd83f2`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/data/schwa_candidates_v1/candidates.json SHA 21b800f37eb29b0c901063ca34dad587ae54abf3b5e0c5ecee8b6bf282945975
- /Volumes/T9/hindi-detector/data/human_g2p_bank_v5/word_bank.json SHA 056edc34c0a1788b057c983d1697b35f5504970fd9463e03dead3b1712087e8d
- /Volumes/T9/hindi-detector/data/wordmatrix_v3/selection.json SHA c450d7fce0cd96c578edbf13f4e43b1bf2586c0e955b3015692f6cecd4c26942
- /Volumes/T9/hindi-detector/models/kala_controls/config.json SHA 0140dbd5fddaffb52f7fb5252d4e0038848483b71787133893084a2e8f5088b4
- .venv-indicf5/bin/python -m scripts.build_intervention_pack

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
