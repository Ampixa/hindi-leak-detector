# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/release_handoff_v39`.
Creation and last review (UTC): 2026-11-09T01:11:33.038146+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Current release handoff and verification. scope.json selected model/review consumer; cli_recheck_launch.json/log own final cold regression/parity pass after reference-cache hardening and descriptive Native observation metadata. Later handoff artifacts inventory exact local/remote copies, service, HTTP/browser outcomes and storage. No new training or encoder/audio copies in this group. Review2026-11-10; keep compact source/model/decision receipts, transient logs/screenshots expire24h after consumer closes. NOT on HF; inherited source/MMS usage restrictions.

## File inventory

- `scope.json`: 266 bytes; SHA-256 `8026a2afeec8961fe671211efdf7bb3594f4ac112f6d9300eeefeee7430c116a`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Current frozen word detector and actual user Swar picks; two original development regressions remain clearly distinguished from independent validation

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

actual_content.log/actual_content_launch.json own the independent whole-content audit of all258 actual scored recordings. Original Native content rule fixed; Hindi Conformer retained as a separate lexical audit. No accent labels, model fitting or threshold changes. Output reports/swar_content_audit_v41.

final_cold.log/final_cold_launch.json own final8 cold/parity checks after query-guard summary fix and cached-reference content-pass hardening. Output reports/release_cold_verification_v43. Earlier v42 attempted guard runs produced valid result JSON but failed at the stdout summary due to a local-variable scope bug; original logs preserved. Corrected contract is unit-tested and rerun separately, not counted as success from saved JSON alone.

final_guard.log/final_guard_launch.json own five final exit-code/schema/phonetic-parity checks in reports/query_guard_verification_v44. Includes two known errors with optional content checks, deliberately mismatched transcript and two reference-quality abstentions. No failed attempt is counted as success. Model frozen.

superseded_status_pointers.json preserves exact previous stale v0/v8 status bytes/hashes. reports/active_goal.json and session_status.json are current timestamped snapshots, not live progress or claims of general accuracy. reference_cache_contract.json checks all86 actual cached texts:84 have content-pass references;2 abstain. All existing reference files were SHA/source/tokenizer/settings verified through actual helper; no synthesis or model change. Keep compact source/status/contract receipts while selected model consumed.

runtime_manifest.json inventories exact selected head/parameters, two frozen acoustic backbones, shared alignment, numerical code pins, real frontend resources, correct24-layer/stockIndicF5 pair, Python/dependency versions, owned service and private review page hashes. Full weights remain remote at inventoried paths; this model head alone is not a standalone audio encoder. No commercial-licence clearance inferred.

final_verification_v47.json records151 tests,8 cold checks,5 guard checks, exact first-play seek/stop, desktop/mobile visual and HTTP/range/model-hash verification and storage. runtime_manifest.json now includes current service44652 and preserves previous service snapshot. Selected model unchanged. Superseded screenshots remain disposable evidence; current verified screenshots have seek-verified in their filename. No general Hindi-error or intonation accuracy claim.

Local copy review: local_copy_manifest.json verifies exact sizes/SHA for selected model/source/validation files against k2. Full acoustic weights, complete raw actual-case evidence and new waveform controls remain remote; this local copy is not a standalone installed speech runtime. Root README gives the k2 CLI. Current local consumer user handoff; next review2026-11-10, inherited model/data rights.
