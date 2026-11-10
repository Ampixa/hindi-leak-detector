# Development speech features

Created by Codex GPT-6, session /root, 2026-11-06 UTC. Owner Ampixa/user.
Origin k2 `/Volumes/T9/hindi-detector/reports/development_features_v1`.
Status active. Next review 2026-11-13. Current consumer: first human-supervised
Hindi-leakage head on separate development texts and speakers.

Inventory: `<clip_id>.npz` contains a 4,096D float32 speech vector (time mean/std
from pinned XLSR layers 12 and 24), plus exact source audio, encoder and extraction
code identities. `manifest.json` lists every retained member, size and SHA-256,
audio hash, speaker, normalized text hash and train/development split.
`extract.log` records progress and explicit failures; delete within 24h after
extraction review. Full frame tensors and posterior dumps are not retained.

Inputs: data/development_v1, native recordings referenced read-only by its access
policy, generated IndicF5 examples, and the existing models/phoneme_probe cache.
No source or synthesizer identity supplies a leakage label. Feature extraction
does not train a detector. Caches with another input/model/code identity fail
explicitly instead of being reused. Original evaluation features are retained in
their separate immutable comparison_features_v1 group and are not overwritten.

Reproduce from the project root: python -m scripts.extract_comparison_features
--pack data/development_v1 --out reports/development_features_v1. Model and
corpus licences/privacy restrictions still apply. No public/HF upload; features
stay on k2. Delete superseded feature caches when their consumer finishes, keeping
compact metrics, source identities, model state and irreplaceable human labels.
