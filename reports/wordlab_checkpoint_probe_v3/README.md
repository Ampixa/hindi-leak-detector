# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/wordlab_checkpoint_probe_v3`.
Creation and last review (UTC): 2026-11-07T09:28:23.443006+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Check whether local source evidence transfers to Pocket checkpoint pairs sharing architecture and synthesis recipe.

## File inventory

- `report.json`: 925 bytes; SHA-256 `5598f134e94d61176b418417159edb9b6cd57b5908ac5b153d0dffa5b6539154`.
- `predictions.json`: 28386 bytes; SHA-256 `24eea9dc56bbaadeff0ba259802463a3f8916654753ad064b047fffed2260ce1`.
- `source_snapshot.json`: 4578 bytes; SHA-256 `37d6761ed7b86ce87c26d9e7acb8564cee3e497a06d74efa53c53527773568e6`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/reports/wordlab_pocket_probe_v3
- Evidence manifest SHA ae4ea0df2c822d751250070d01030f50cf6ffe1bcde50f061411d4d7f01ac962
- /Volumes/T9/hindi-detector/reports/wordlab_models_v3/models.json
- Model SHA 3a56e28de85292664c5902f7ba3146630212efde4e0e0637fb69f200a7c66501
- python -m scripts.score_checkpoint_probe --evidence reports/wordlab_pocket_probe_v3 --models reports/wordlab_models_v3/models.json --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
