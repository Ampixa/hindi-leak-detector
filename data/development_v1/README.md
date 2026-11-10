# Separate native development selection

Created by Codex GPT-6, session /root, 2026-11-06 UTC. Owner Ampixa/user.
Origin k2 `/Volumes/T9/hindi-detector/data/development_v1`. Status active.
Next review 2026-11-13. Current consumer: separate detector training/development
examples, with actual human labels still required.

Inventory: `selection.json` records 32 native targets, sixteen source speaker
groups, distinct same-speaker reference audio/text, exact paths and SHA-256,
and twelve training versus four development speaker groups. Original audio
remains at `/Volumes/T9/indicvoices_r_ne/audio` and is not copied or modified.
`review_access.json` explicitly permits the review server to read only that
external corpus directory. It is separate from the locked synthesis selection,
so adding review access does not change the generation input checksum.
The input manifest is `/Volumes/T9/indicvoices_r_ne/groups.jsonl`, whose actual
four-field schema was inspected. The local preparation code maps `source` to
the original HF speaker_id; its historical execution revision is unknown.

Source corpus is IndicVoices-R Nepali, declared CC BY 4.0 by its preparation
documentation. Original corpus attribution and voice use limitations still
apply; private internal research only here, no HF/public upload. Selection is
not an assertion of reviewed pronunciation or transcript accuracy. Human native
recording identity never supplies a negative leakage label.

All benchmark sentences are screened out by NFC text normalization, exact and
long-substring matching and fuzzy similarity >=0.90. Speaker IDs are separated
inside this development set. Cross-corpus identity against benchmark opaque
speaker IDs still needs validation; do not claim complete speaker OOD from
different naming schemes. Reproduce using scripts/select_development_data.py
and pinned source manifests. Retain compact selection and human judgments while
the goal is active; remove obsolete generated takes within 24h of consumer
completion. No extra native waveform backup is created.
