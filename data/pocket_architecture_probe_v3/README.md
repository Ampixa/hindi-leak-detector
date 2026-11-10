# Same-architecture checkpoint control

Created by Codex GPT-6 /root, 2026-11-07 UTC. Owner Ampixa/user. Status active.
Origin authored metadata cdjk; k2 /Volumes/T9/hindi-detector/data/pocket_architecture_probe_v3.
Current consumer: testing word/prosody heads across two Pocket checkpoints using
the same runtime, decoder, reference WAV, text and seed. Review by 2026-11-14.

Inventory: selection.json enumerates twelve existing WAVs from six matched pairs,
original paths, SHA, source checkpoint tags, source texts and grouping policy.
No audio copies. Source waveforms/recipes and licences remain documented under
reports/pocket_pilot_v1. This is not a gold native/Hindi accent dataset. Historical
parent hash of the old fine-tune is unknown, and generated voices may change, so
this is an architecture control rather than a proven causal fine-tuning test.
No HF/public upload. Keep compact metadata with diagnostic results; delete when
superseded, preserving referenced irreplaceable inputs and human judgments.
