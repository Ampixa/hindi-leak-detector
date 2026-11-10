# Crossed word, textual context and reference-voice TTS experiment

Created by Codex GPT-6 /root, 2026-11-07 UTC. Owner Ampixa/user. Status active.
Origin k2 /Volumes/T9/hindi-detector/data/wordmatrix_v3; authored source/metadata on cdjk
/Users/cdjk/github/hindi-detector/data/wordmatrix_v3. Current consumer: disentangling word
pronunciation/prosody from voice and text identity. Review by 2026-11-14.

Parent inventory: selection.json records 512 target/reference jobs for IndicF5,
24 disjoint focus lexemes, four authored carrier contexts per lexeme, 8/4/4
reference groups for train/development/evaluation, exact original paths and SHA.
audio8_manifest.json records the same targets and reference bytes in native
Audio8 harness structure. No native reference copies are created on k2.
indicf5/ holds this provenance, run.json, append-only items.jsonl, source snapshot
and wm3_*.wav outputs enumerated by size/SHA in receipts; synthesis.log expires
within 24h after run review. No waveform is a gold per-word Hindi label.

Focus lexemes are drawn from the unreviewed schwa-candidate queue. Authored
carrier text and punctuation are not certified pronunciation or generated-emphasis
labels. Original dictionary contrast uncertainty remains; never infer every
IndicF5 word is incorrect. Same input references condition both systems, but
identity-preserving voice cloning is a measurement to verify, not an assumption.
No benchmark sentence enters this experiment's fitting. Focus lexemes and
source reference groups are separated across splits. Native-corpus speaker IDs
are original source groups; real-person cross-corpus identity remains unverified.

Source corpus/model licences and voice privacy limits apply; internal research
only, NOT on HF. No public Ampixa site change. Original IndicF5 and Audio8 R3
checkpoint hashes/settings remain pinned to the earlier verified native recipes.
Use scripts/build_word_matrix.py then scripts/run_indicf5_matrix.py --device mps
--pack data/wordmatrix_v3. Audio8 runs in an owned temporary GPU directory with
resource checks, checkpoint hashes and exact transfer manifests, then verified
outputs return to k2 before temporary deletion.

Retain only while this crossed-data experiment/training/listening consumer is
active. Remove unused/superseded takes after consumer closure, within 24h; keep
compact configs, source/selection hashes, selected models and necessary evidence.
No extra checkpoints copied. source_snapshot.json preserves the exact runner
bytes; restore to original project-relative path rather than running in place.

Owned GPU workspace /home/ampixa/hindi-detector-wordmatrix-v3 additionally holds
script.py (the exact native Audio8 matrix adapter), queue_worker.py (one-shot
resource waiter), references/<source-basename>.wav (the sixteen manifest-listed
reference copies), manifest.json, queue.pid, queue_status.json plus its atomic
.partial staging file, queue.log, and outputs/{README.md,run.json,items.jsonl,
synthesis.log,wm3_*.wav}. Worker waits for three low-utilization GPU observations
before launch, never kills another workload, and inherits low CPU priority. Native
worker retains its 10 GiB process GPU budget and 18 GiB disk/RAM guards. Queue
files are runtime diagnostics, disposable after job closure; keep receipts and
verified k2 outputs. No automatic janitor is installed.
