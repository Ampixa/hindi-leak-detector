# Matched Audio8 proxy acoustic features

Created by Codex GPT-6 /root, 2026-11-06 UTC. Owner Ampixa/user. Status active.
Origin k2 /Volumes/T9/hindi-detector/reports/audio8_proxy_features_v1; metadata
copy on cdjk only. Current consumer: the user-authorized source-proxy baseline.
Last review 2026-11-06; next review 2026-11-13.

Inventory: manifest.json enumerates every <clip_id>.npz member, sizes, SHA-256,
audio hash, speaker, normalized text and split; NPZ contains 4096 float speech
features and the exact audio/encoder/extraction-source identities. extract.log
records progress/errors; delete within 24h of extraction review. Full frame
tensors and posteriors are not retained. Numerical recipe equals the original
IndicF5 feature recipe: pinned XLSR layers 12/24 mean and std, CPU 16 kHz input.

Inputs: data/audio8_proxy_v1 selection/audio and models/phoneme_probe. Reproduce
with python -m scripts.extract_comparison_features --pack data/audio8_proxy_v1
--out reports/audio8_proxy_features_v1. Caches require the same audio/code/model
identity; use a new group for changed inputs. No proxy class is human supervision.
Source/model licences and privacy restrictions apply; internal only, NOT on HF.
Keep while training/inference consumes it; delete superseded features after their
consumer finishes, preserving compact receipts, source/model state and results.
