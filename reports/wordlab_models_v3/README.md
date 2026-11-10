# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/wordlab_models_v3`.
Creation and last review (UTC): 2026-11-07T08:05:26.537481+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Separate phonetic/prosodic local evidence trained with weak utterance-bag labels; no word-level accuracy claim.

## File inventory

- `models.json`: 145503 bytes; SHA-256 `3a56e28de85292664c5902f7ba3146630212efde4e0e0637fb69f200a7c66501`.
- `report.json`: 7226 bytes; SHA-256 `99936d25e84924ccc14d48368bc7462b44d9908403b54fd7c4b7bb5ca50b5cd7`.
- `selection.json`: 3024 bytes; SHA-256 `34fc1caeb7282d02aa44b2a8ae9023e66ee1b06e81cd57c5e881002283b8f189`.
- `word_predictions.json`: 126834 bytes; SHA-256 `4ebf33c0134fa4ba73fae4a54d50edeb250f31534081af4fbbbacccf2646afbf`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/reports/wordlab_training_v3 manifest SHA 3ca832e0217e426abd88f1433ee516dc7e62f344601068e7a520d9b1166eca50
- /Volumes/T9/hindi-detector/reports/wordlab_test_v3 manifest SHA 61c01a6725585f4a8e77ce64924b5d279d5a79402b2502aa7500da196cbe9ba5
- Driver SHA 94ccdb33c5134ae6c693fe675d5a77c2b34677f35c5e59af86f99f6f70054b5f
- python -m scripts.train_word_heads --training reports/wordlab_training_v3 --test reports/wordlab_test_v3 --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
