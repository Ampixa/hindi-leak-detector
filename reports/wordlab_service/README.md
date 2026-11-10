# Private read-only word listening service

Created by Codex GPT-6 /root, 2026-11-07 UTC. Owner Ampixa/user. Status active.
Origin k2 /Volumes/T9/hindi-detector/reports/wordlab_service. Consumer: user
listening to actual word-level phonetic/prosody evidence. Review by 2026-11-14.

Inventory: pid stores the owned server PID; server.log contains requests and
explicit read failures, never credentials. connectivity.json records the private
Tailscale mapping and verification. runtime files are disposable when the service
ends; keep compact connectivity/reproduction evidence. No audio/model backup.
Stop the exact owned process only after checking use, disable only its Tailscale
TCP forwarder, and remove logs within 24h of service completion. This service is
a read-only research UI and cannot write or overwrite human annotations.

Reproduce: python -m hindi_detector.wordlab_server --pack data/indicf5_audio8_v1 --pack data/pocket_architecture_probe_v3 --pack data/human_g2p_bank_v5
--page review/wordlab.html --data review/human_word_data_v6 --port 8770. Bind loopback;
a private Tailscale TCP forward exposes the service. No public Funnel or live
Ampixa website change. Detached backend needs restart after host reboot.

Current listener serves 30 cases from 108 verified original benchmark/Pocket audio identities; /progress is the generated readable work/architecture snapshot. Native progress JSON is under reports/wordlab_progress_v3_checkpoint_audit. Current service PID is the remote pid file, not a hardcoded historical value. Reboot requires restart.

2026-11-07 human update: serves 62 cases (32 human word families plus 30 historical TTS comparisons), with 232 additional verified corpus identities. Human contexts differ; only the chosen word is charted and first occurrences are fixed. Current /api/current-run includes completed real-word extraction and supported-path coverage. Source HTML is review/wordlab.html; current progress authoring source is review/pocket-progress.html, copied exactly to wordlab-progress.html. Only this owned readonly service was restarted; annotation services and their labels were unchanged.

Current v6 serving data adds the completed seven-window audit; the progress API also exposes the auxiliary intervention render/feature/head reports. The site-likelihood component metric is synthetic request classification only, never advertised as real Hindi accuracy. Current PID is always the remote pid file.

2026-11-07 v8 update: readonly service now serves review/human_word_data_v8 with110 cases (48 fixed matrix examples,32 human families,30 earlier cases), and adds wordmatrix_v3 plus pocket_wordmatrix_v4 audio packs. Per-word synthetic-request components, crop sensitivity, human score ranks and matched Pocket style differences are separate, unvalidated measurements. /api/current-run exposes complete real-word transfer and current exact-human-context generation. Stop/restart only current pid after command verification; annotation services unchanged.

Corrected comparison v14: /comparison serves review/corrected-pair.html and /api/comparison exposes only pinned new model-pair metadata, synthesis/content coverage and verified paired clips. /comparison-audio resolves clip IDs exclusively through new receipts and hashes every WAV. corrected_pair_deployment.json records current owned restart. No annotation writes, public hosting or live ampixa.com changes.

2026-11-08 training preview refresh: training_refresh_v16.json records exact source hashes, old/new owned PID and unchanged arguments. training_refresh_v16.log is disposable stdout/stderr, retain only through final review; expiry24h after consumer completes. Annotation services untouched. Owner Ampixa/user, Codex /root; reproduce recorded command.

swar_word_review_v19.json/log: 2026-11-08T21:17:12.077249+00:00 read-only /swar, /api/swar, per-case whitelist audio and hash-verified model download. Exact owned process/source pins recorded. Private Tailscale only; original Swar and annotation services untouched. Current consumer actual user-pick review; review2026-11-09, log disposable24h after final verification.

2026-11-09: current entry points / and /swar (also /progress) show actual Swar word results using active_vowel_model.json, selected task_vowel_release_v31. /comparison retains source experiments; /legacy old listener. release_v31_transition.json records checked prior/new PID, exact command and source/model/page hashes; release_v31_server.log current owned request/errors output. Root pointer is documented in project README and this transition. Current consumer user model review, next review2026-11-10; stop only exact owned PID and retain compact deployment evidence, logs disposable24h after service closes. No public Ampixa/Funnel change or annotation-service restart.

range_v47_transition.json/range_v47_server.log and pid identify the current owned read-only service restart for seekable HTTP audio. Source audio SHA is checked before range slicing. Single valid byte ranges return206 and Content-Range; malformed/unsatisfiable ranges416. Existing annotation ports and Tailscale forwarding untouched. Model bytes unchanged. Logs active while service consumed, review2026-11-10; remove redundant request logs24h after service consumer closes, retain compact transition receipts.
