# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/data/prosody_transfer_v49`.
Creation and last review (UTC): 2026-11-09T02:28:44.911032+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Frozen voice-controlled prosody transfer experiment. transfer_plan.json pins164 exact Native carriers/comparison references and stockIndicF5 donors; selection.json requests one extra Native donor per context from a third conditioning group. Word families and conditioning groups remain disjoint across100/24/40 train/dev/reserved contexts. references/*.wav exact verified prompt copies, reference_inventory.json source/bytes/SHA; teacher/ later owns generated Native donor outputs and receipts. No real accent/error labels fabricated and no change to served vowel model. Consumer replacement for failed legacy prosody-source head; review2026-11-10. Raw redundant outputs expire24h after consumer closes; keep compact configs/metrics/selected state. NOT on HF; inherited rights.

## File inventory

- `selection.json`: 165239 bytes; SHA-256 `764feaa6e6362851755c45c830286524c4fcc20eb2bbdc420457dbec69ec03d1`.
- `model_pair.json`: 1575 bytes; SHA-256 `06b469ad88fff6ea655c3c39ba7b70082ed9bb30c607afe61141b61c16b370e6`.
- `transfer_plan.json`: 1964228 bytes; SHA-256 `24839c0f72c51fbb214845c630b99a23f418fe06e92b5c665182b5cbb1618a14`.
- `protocol.json`: 1782 bytes; SHA-256 `5c4c0dab0697e82d9378d7c4f8b7278df9ec126577373a49cb841e05594e02cd`.
- `source_snapshot.json`: 10499 bytes; SHA-256 `5656b6b980563208601ca4fc13f8677593c1b32da96718d5b08633c770a302bf`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.build_prosody_transfer_plan

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. Generated waveforms, if present, are explicitly described above and listed in inventory_v57.json.
