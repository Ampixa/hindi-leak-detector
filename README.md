# hindi-leak-detector

Detects Hindi pronunciation that survives when a Hindi-pretrained TTS model is
fine-tuned for Nepali. It scores individual words: did a vowel come out the
Hindi way (quality change) or get dropped (deletion), compared with a Nepali
reference reading of the same sentence?

Flags are **candidates, not confirmed errors**. General Hindi-error accuracy has
not been measured against human labels.

## How it works

For each word in a sentence:

1. **Dual G2P** builds the Nepali pronunciation and the Hindi alternative, e.g.
   हैन → Nepali `h ʌi̯ n ʌ` vs Hindi `h ɛː n`. Where they differ is a testable site.
2. **Same-text reference**: two readings of the sentence by the Nepali teacher
   in *other* voices. Each is ASR-checked and must itself show the Nepali vowel
   to count ("qualified").
3. **Alignment** with frozen MMS locates each word in query and reference.
4. **Evidence** from two frozen phone encoders (XLSR-53, LV-60): CTC path
   margins Nepali-vs-Hindi and vowel-region features, 57 per site.
5. **Two 945-parameter MLP heads** score vowel-quality change and deletion.
   A word is flagged only when the qualified references agree; disagreement
   is `uncertain`.

Worked example (हैन, from `reports/cross_system_results_v80/cases/cross80_s00.json`):
IndicF5 scores 0.999 against qualified references in every voice and ASR hears
"hand"; the Nepali teacher mostly scores 0.02–0.17 and ASR hears होइन.

## Example measurement: Swar by Naamche Labs

[Swar](https://huggingface.co/Aarjan/swar) is an open Nepali voice-cloning model
by [Naamche Labs](https://github.com/Naamche-Labs/swar), fine-tuned from
[ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5). We ran Swar's
default checkpoint (`podvoices_ema.pt`, HF revision `8ca6f1b`, SHA-256
`b932ea21…5b53`) through its own text frontend on 26 fresh sentences × 4 voices,
and scored it with the frozen detector next to stock IndicF5 and our Nepali
reference teacher.

| | Swar (Naamche Labs) | IndicF5 (stock) | Nepali teacher |
|---|---|---|---|
| ASR content check pass / quarantine | 100 / 4 | 93 / 11 | 104 / 0 |
| Focal word flagged (content-pass readings) | **36 / 100** | 60 / 93 | 4 / 104 |
| Readings with any flagged word | 41 / 104 | 79 / 104 | 15 / 104 |

Swar has zero focal flags on हैन, नगरौ, सहमति, बैठक and नैतिक, all flagged in IndicF5. Four
words stay flagged in every voice and both sentence contexts: **नेकपा,
हुनलाई, पिढको, तलको**. Consistency across voices points at learned
pronunciation rather than a bad take, but these are unverified until a listener
confirms them. Only 126 of 724 word positions carry a testable vowel contrast.

Details: `reports/cross_system_swar_v84/` (Swar), `reports/cross_system_results_v80/`
(IndicF5 / teacher). Swar weights are CC BY-SA 4.0 and are **not** redistributed
here; no generated audio is included.

## Layout

- `hindi_detector/` – library: G2P, alignment, evidence, detector heads
- `scripts/` – experiment drivers; `detect_nepali_leakage.py` is the CLI;
  `run_swar_cross_system.py`, `screen_swar_cross_system.py`,
  `score_swar_cross_system.py` reproduce the Swar result
- `reports/` – compact JSON results and small checkpoints; live detector:
  `reports/task_vowel_release_v31/model.pt` (`active_vowel_model.json`)
- `data/` – text plans, manifests and receipts (audio excluded)
- `tests/` – unit tests (`pytest`)
- `docs/LAB_NOTES.md` – full experiment log

## Usage

```sh
python -m scripts.detect_nepali_leakage --query QUERY.wav --text 'NEPALI TEXT' \
  --out reports/NEW_RESULT --check-query-content
```

Needs the frozen encoder/aligner resources (MMS, XLSR-53, LV-60, ASR gate) under
`models/`; these are not in the repo. References are generated with the Nepali
teacher or passed with `--reference SAME_TEXT.wav`.

## Limits

- No independent accuracy on natural Hindi errors; two known errors (हैन, किनकि)
  were used as development regressions.
- Controlled-edit audits detect 34/39 deletions and 17/25 quality edits with
  0 sham flags; these are synthetic edits.
- Vowels only; consonants and prosody are not judged.

## Credits

- **Swar** by Naamche Labs – https://github.com/Naamche-Labs/swar, https://huggingface.co/Aarjan/swar (CC BY-SA 4.0)
- **IndicF5** by AI4Bharat – https://huggingface.co/ai4bharat/IndicF5
- MMS, XLSR-53, wav2vec2 LV-60 (Meta), Vocos (charactr)
