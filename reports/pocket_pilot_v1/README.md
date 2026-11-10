# Pocket Hindi transfer listening pilot

Created by Codex (GPT-6), session /root, on Khatra-Dev-Nepals-Mac-Studio.local, 2026-11-06T07:52:56.870229+00:00.
Owner: Ampixa/user. Origin: /Volumes/T9/hindi-detector/reports/pocket_pilot_v1. Status: active. Next review: 2026-11-13.
Purpose: matched inference examples to review Hindi-to-Nepali transfer, not a training run.
Current consumer: Hindi leakage lab. No native labels or accuracy claim.

Inventory: reference.wav is an explicitly converted copy of /Users/k2/madhyam/madhyam-tts/dashboard/public/audio/pocket/g00005.ref.m4a;
parent.yaml and nepali_v1c.yaml are local-path inference configs; each
<system>.<text_id>.s<seed>.wav is generated mono 24k audio for the three texts and
seeds 7/29, twelve files total; manifest.json records all parameters and results.
inventory.json holds every retained member's size and SHA-256. Partial runs are
recorded by the files and traceback in the project run log; never treat as complete.

Existing fine-tune: /Users/k2/madhyam/runs/nepali_v1c/model.safetensors, exported at step 1309 per its progress/train log.
Historical parent byte hash is unavailable: the current pinned Hindi parent may
not be byte-identical to the historical training initialization. This comparison
cannot by itself establish a causal fine-tuning effect. One voice is a smoke
test only, not speaker-held-out validation. Reference consent/rights are inherited
from the internal source corpus and were not established anew by this session;
keep these outputs internal. No weights or audio are uploaded to HF.

Run: PYTHONPATH=/Volumes/T9/hindi-detector /Users/k2/madhyam/pocket-tts/.venv/bin/python
/Volumes/T9/hindi-detector/scripts/run_pocket_pilot.py. Source code is in this
project; exact Pocket source revision and model hashes are in manifest.json.
Keep listening audio only while actively reviewed; delete within 24h after the
review finishes unless a dated current consumer is recorded. Keep compact
manifest/conclusions. Known duplicates: original reference source above; cdjk
may receive the twelve small listening outputs for the internal review page.
