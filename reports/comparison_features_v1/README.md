# Compact frozen speech features

Created by Codex GPT-6, session /root, 2026-11-06 UTC. Owner Ampixa/user.
Origin k2 `/Volumes/T9/hindi-detector/reports/comparison_features_v1`.
Status active. Next review 2026-11-13. Current consumer: the active detector goal,
for reference diagnostics and later held-out evaluation with human labels.

Inventory: `<clip_id>.npz` contains only a 4,096-element float32 `speech` vector:
time mean/std for layers 12 and 24 of the pinned XLSR phoneme encoder.
`manifest.json` lists every member, size and SHA-256, original audio checksum,
speaker, sentence, split, encoder pin and feature-code hash. `extract.log` records
progress and tracebacks; disposable within 24h of completed extraction review.
No full frame tensors or posterior dumps are retained. Until the manifest is
written, this group is incomplete and must not be treated as a finished run.

All current clips are reserved evaluation, not training data. No label is derived
from encoder output or TTS model identity. Pooled embeddings can carry speaker
and synthesizer identity; only independent human labels and appropriate held-out
tests can qualify their use in a leakage model. This is not a trained detector.

Inputs: data/indicf5_audio8_v1 with verified native/audio8/IndicF5 audio and
models/phoneme_probe; retain upstream model/audio licence and privacy limits.
Reproduce with `python -m scripts.extract_comparison_features` from the project
root in the existing Pocket runtime (torch/transformers/scipy/numpy/soundfile).
Direct script invocation without installing the project first failed to import
the local package; this is recorded in extract.log and corrected by module entry.
No HF/public upload. No cdjk
feature copy needed; only manifest.json is copied to cdjk. This original v1 cache
does not embed input identities within each NPZ; its immutable manifest binds the
file and audio hashes. The original extractor source was superseded without a
snapshot, so its exact historical source bytes cannot currently be restored.
The current schema-2 extractor reproduces the documented numerical recipe into
a NEW output group and refuses to reuse these older unbound cache files. The
original feature files remain checksum-verified; never relabel them as schema 2.
Delete superseded features once their evaluator finishes,
within 24h; retain compact metrics, source/audio hashes and human labels instead.
