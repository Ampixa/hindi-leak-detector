# Audio8 proxy development artifacts

Created by Codex GPT-6 /root on 2026-11-06 UTC. Owner Ampixa/user. Status active.
Origin authored metadata cdjk /Users/cdjk/github/hindi-detector/data/audio8_proxy_v1;
synthesis /home/ampixa/hindi-detector-proxy-v1 on khalti-gpu, retained audio k2
/Volumes/T9/hindi-detector/data/audio8_proxy_v1. Review by 2026-11-13.
Current consumer: source-supervised Hindi-leakage baseline requested by the user.

Inventory: manifest.json specifies all 32 development jobs and 16 exact reference
filenames, original paths and SHA-256; README.md is this provenance. On GPU,
references/ contains those exact WAV members temporarily, script.py is the pinned
project scripts/run_audio8_proxy.py, outputs/ holds README.md, run.json, items.jsonl,
synthesis.log and dev_*.wav enumerated with size/SHA in the receipts. On k2,
audio/ retains the generated WAVs; selection.json is the feature/review loader
contract joining the actual receipts to development jobs, and run.json and
items.jsonl retain exact inference provenance. No new checkpoint copy is made.

Inputs: existing IndicVoices-R development selection and pinned Audio8 R3 checkpoint
SHA 0ab717efa1c7359f0006ad59d50855c0560ead2de78c361f82929c8dbd1adfef;
native Audio8 baseline processor/code/codec reused from tts_bench_v0. Native F5
and Audio8 consume the same reference WAV bytes and transcript. No benchmark
sentence enters training. The user authorizes source identities as WEAK proxy
labels; these are not human gold and must never enter annotations/*.jsonl.

Reproduce with GPU benchmark .venv-audio8/bin/python script.py --manifest manifest.json
--out outputs. Code enforces existing GPU/disk/RAM guards and verifies input SHA.
One seed, original native generation, no take selection or signal normalization.
Model/corpus licence and voice privacy restrictions continue to apply, internal
research only; NOT on HF. Delete temporary GPU copies after verifying k2 copies
and no active job uses them; keep compact receipts, source and model hashes.
Audio has an active training/listening consumer until superseded. Logs expire
within 24h of experiment review. Retain only final model and compact results.
