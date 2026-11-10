# Fresh native negative-proxy evaluation, never fitting

Created by Codex GPT-6 /root at 2026-11-07T07:20:25.414834+00:00, owner Ampixa/user. Status active.
Original runtime k2 /Volumes/T9/hindi-detector/reports/native_probe_windows_v2; metadata copy on cdjk
/Users/cdjk/github/hindi-detector/reports/native_probe_windows_v2. Current consumer: user-requested v2 improvement.
Last review 2026-11-07; next review 2026-11-14.

Inventory: manifest.json enumerates 64 full/short feature NPZs from 32 fresh source speaker groups, exact parent hashes/crop indices and member hashes/sizes. extract.log is temporary progress output.
This README documents itself. No full frame tensors, posterior dumps or cropped
WAV backups retained. Short views are deterministic signal windows, not word or
schwa annotations. Parent speaker, text and split remain attached to every view.

Inputs: inspected IndicVoices-R native manifest, existing development_v1 and
audio8_proxy_v1 packs, pinned XLSR model and immutable v1 feature caches.
Selection: python -m scripts.select_native_probe. Features: python -m
scripts.extract_proxy_windows --pack ... [--existing ...] [--probe] --out ... .
Native speech is an explicit weak negative proxy, never human accent gold.
Source/model licence and voice privacy restrictions continue to apply; internal
only, NOT on HF. Retain until this model comparison is superseded, then delete
unused features while retaining compact metrics, exact source/selection pins,
final model and human judgments. Logs expire within 24h of extraction review.
