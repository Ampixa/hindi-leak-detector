# Four-source matched word alignment, phonetic posterior and normalized prosody pilot

Created by Codex GPT-6 /root, 2026-11-07 UTC. Owner Ampixa/user. Status active.
Origin k2 /Volumes/T9/hindi-detector/reports/wordlab_pilot_v3; metadata/review sync on cdjk.
Current consumer: user-requested word-level Hindi-leakage/prosody work.
Next review 2026-11-14. Source/model licences and voice privacy limits apply.
Internal research only, NOT on HF. MMS alignment model is CC BY-NC 4.0.

Inventory: manifest.json and items.jsonl enumerate every analyzed input, original
audio SHA, speaker/text/split, status and explicit errors. <clip_id>.json retains
word times, romanization, alignment scores, phone-probe sequences, separate
central-vowel posterior evidence and normalized pitch/energy tracks. <clip_id>.npz
retains compact word PPG means/std, 29 relative prosody features, time/F0 tracks
for QA, and parent audio identity. Every member has size/SHA in the manifest.
extract.log is disposable within 24h of batch review. No word WAV copies, full
posterior tensors or hidden speech embeddings retained. This README inventories
itself. Original audio remains in its existing documented packs/corpus.

Reproduce via python -m scripts.extract_word_evidence --pack ... [--sentence ...]
--out NEW. Model/runtime inventories pin exact input bytes and packages; source
hashes are in the manifest. Uses uroman by Ulf Hermjakob, USC Information Sciences
Institute (2015–2020). Romanization/alignment does not establish pronunciation
correctness. Central-vowel mass is not a validated schwa detector or Hindi word
label. Relative F0 and energy remove global register/gain offsets by construction;
that is not proof of invariance to full voice conversion. Native/source labels
remain weak assumptions. Keep these compact features while the word/prosody
models and listening lab consume them; delete superseded caches after that
consumer ends, retaining selected model, compact conclusions and human judgments.

source_snapshot.json contains exact UTF-8 driver, engine and prosody-helper source
plus SHA-256, captured before extraction. Restore each member to its recorded
project-relative path to reproduce historical code; do not execute the snapshot
in place. This compact source snapshot is retained with the experiment.
