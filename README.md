# hindi-leak-detector

Detects Hindi pronunciation that survives when a Hindi-pretrained TTS model is
fine-tuned for Nepali. It scores individual words: did a vowel come out the
Hindi way (quality change) or get dropped (deletion), compared with a Nepali
reference reading of the same sentence?

Flags are **candidates, not confirmed errors**. General Hindi-error accuracy has
not been measured against human labels.

## How it works

The detector never asks "does this sound Hindi?" in the abstract. It asks a
narrower question it can measure: **at a vowel where Nepali and Hindi
pronunciation differ, does this reading move away from a trusted Nepali
reading of the same sentence, in the Hindi direction?**

```
   query.wav  +  Nepali text                     same text, other voices
        │                                                 │
        ▼                                                 ▼
 ┌──────────────┐                              ┌──────────────────────┐
 │ content gate │  Nepali Zipformer +          │ Nepali teacher reads │
 │  (ASR)       │  IndicConformer; wrong       │ the sentence twice   │
 └──────┬───────┘  words → unscored            │ (cached or rendered) │
        │                                      └──────────┬───────────┘
        │                                                 │ ASR content check
        ▼                                                 ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ 1. Dual G2P                                                          │
 │    हैन  Nepali  h  ʌi̯  n  ʌ      Hindi  h  ɛː  n                     │
 │                  ▲         ▲                                         │
 │          quality site   deletion site   (only differing vowels)     │
 └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ 2. Word alignment (frozen MMS)  →  word spans in query and refs      │
 │ 3. Phone evidence (frozen XLSR-53 + wav2vec2 LV-60 CTC)              │
 │      • path margin: Nepali phones vs Hindi phones over the word      │
 │      • vowel-region evidence at 3 boundary settings                  │
 │      → 57 features per site: query, reference, query − reference     │
 └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────┐   ┌──────────────────────────┐
 │ quality head  (MLP, 945) │   │ deletion head (MLP, 945) │
 │ ʌi̯ → ɛː ?                 │   │ final ʌ dropped ?        │
 └────────────┬─────────────┘   └────────────┬─────────────┘
              └──────────────┬───────────────┘
                             ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ 4. Reference consensus, per site                                     │
 │    reference "qualified" only if it clearly shows the Nepali vowel   │
 │    all qualified refs say change   → strong_vowel_change  (flag)     │
 │    all qualified refs say no       → no_strong_vowel_change          │
 │    qualified refs disagree         → uncertain                       │
 │    no qualified ref                → unsupported_reference           │
 └─────────────────────────────────────────────────────────────────────┘
```

Design choices, and why:

- **Relative, not absolute.** Every score is query-vs-reference for the same
  text, so speaker, sentence and recording differences largely cancel. No word
  lookup table, speaker ID or voice embedding is a model input.
- **Two tiny heads on frozen encoders.** 945 parameters each, trained on
  paired same-text examples (418 quality pairs, 238 deletion pairs; 147
  development pairs), so they score "this vowel differs from the reference"
  rather than memorising words.
- **Abstain rather than guess.** Wrong words, an unqualified reference or
  disagreeing references produce `unscored_*` / `uncertain`, never a flag.
- **Narrow coverage, stated.** Only words with a modelled Nepali/Hindi vowel
  contrast are tested (126 of 724 positions in the 26-sentence set); phones
  without an acoustic representation (e.g. `d͡z`, nasal `ã`) are skipped.

### Worked example: हैन

From `reports/cross_system_results_v80/cases/cross80_s00.json`,
*यो बाटो पुरानो हैन, नयाँ पुलतिर जानुपर्छ।*

| Reading | ASR heard | Quality score vs qualified ref | Decision |
|---|---|---|---|
| IndicF5, voice 1 | "hand" | 0.999 | flagged |
| IndicF5, voice 2 | "hand" | 0.999 | flagged |
| Teacher, voice 1 | होइन | 0.021 | not flagged |
| Teacher, voice 2 | होइन | 0.17 | not flagged |
| Teacher, voice 0 | होइन | 0.996 | flagged (candidate false alarm) |

IndicF5 says Hindi *hɛːn*; the Nepali teacher says *haina*. The last row is
why flags stay candidates until a listener confirms them.

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
