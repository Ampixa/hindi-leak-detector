"""Freeze fresh same-text, matched-voice renders outside the Swar corpus."""
from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
import shutil
import unicodedata

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import require_pair, sha
from hindi_detector.dual_g2p import DualG2P
from hindi_detector.pronunciation_sites import compile_word
from scripts.train_paired_local import partition

ROOT = Path(__file__).resolve().parents[1]
TEXTS = [
    ('हैन', 'confirmed_word_new_context', 'यो बाटो पुरानो हैन, नयाँ पुलतिर जानुपर्छ।'),
    ('हैन', 'confirmed_word_new_context', 'आज बिदा हैन, कार्यालय खुलै छ।'),
    ('किनकि', 'confirmed_word_new_context', 'म चाँडै फर्किन्छु किनकि भोलि बिहान काम छ।'),
    ('किनकि', 'confirmed_word_new_context', 'ऊ हिँडेर आयो किनकि बस छुटिसकेको थियो।'),
    ('नेकपा', 'confirmed_word_new_context', 'आज नेकपा र अरू दलका प्रतिनिधि बैठकमा आए।'),
    ('नेकपा', 'confirmed_word_new_context', 'उनले नेकपा भन्ने नाम स्पष्टसँग उच्चारण गरे।'),
    ('तलको', 'confirmed_word_new_context', 'तलको कोठामा आज पढाइ भइरहेको छ।'),
    ('तलको', 'confirmed_word_new_context', 'कृपया तलको सूची एकपटक ध्यान दिएर हेर्नुहोस्।'),
    ('हुनलाई', 'confirmed_word_new_context', 'राम्रो शिक्षक हुनलाई धैर्य र अभ्यास चाहिन्छ।'),
    ('हुनलाई', 'confirmed_word_new_context', 'यो काम सफल हुनलाई सबैको सहयोग चाहिन्छ।'),
    ('चलाउनको', 'confirmed_word_new_context', 'मेसिन चलाउनको लागि पहिले सुरक्षा नियम पढ्नुहोस्।'),
    ('चलाउनको', 'confirmed_word_new_context', 'घर चलाउनको खर्च महिनैपिच्छे बढिरहेको छ।'),
    ('नगरौ', 'confirmed_word_new_context', 'सानो कुरामा झगडा नगरौ, पहिले शान्त भएर कुरा गरौ।'),
    ('नगरौ', 'confirmed_word_new_context', 'अरूको गल्तीमा हतार गरेर निर्णय नगरौ।'),
    ('पिढको', 'listener_similar_word_new_context', 'यो पिढको अनुभव नयाँ पुस्ताका लागि उपयोगी छ।'),
    ('पिढको', 'listener_similar_word_new_context', 'हरेक पिढको सोच र बोल्ने शैली फरक हुन सक्छ।'),
    ('सपना', 'new_focal_word', 'सानो छँदादेखि उसको सपना शिक्षक बन्ने थियो।'),
    ('नगरमा', 'new_focal_word', 'यो नगरमा नयाँ पुस्तकालय खोलिएको छ।'),
    ('तरकारी', 'new_focal_word', 'बजारबाट ल्याएको तरकारी राम्रोसँग पखाल्नुहोस्।'),
    ('सहमति', 'new_focal_word', 'लामो छलफलपछि सबै पक्षबीच सहमति भयो।'),
    ('मकै', 'new_focal_word', 'हामीले खेतमा पाकेको मकै घरमा ल्यायौँ।'),
    ('नैतिक', 'new_focal_word', 'यो निर्णय गर्दा नैतिक पक्ष पनि विचार गर्नुपर्छ।'),
    ('नौलो', 'new_focal_word', 'गाउँमा बनेको पुल सबैका लागि नौलो अनुभव भयो।'),
    ('बैठक', 'new_focal_word', 'भोलिको बैठक बिहान ठीक दस बजे सुरु हुन्छ।'),
    ('सबैलाई', 'new_focal_word', 'यो जानकारी सबैलाई समयमै पुर्‍याउनुपर्छ।'),
    ('पैसाले', 'new_focal_word', 'पैसाले मात्र सबै समस्या समाधान गर्न सकिँदैन।'),
]


def key(text):
    return ' '.join(''.join(c if unicodedata.category(c)[0] in {'L', 'M', 'N'} else ' ' for c in unicodedata.normalize('NFC', text)).split())


def strings(value):
    if isinstance(value, dict):
        for name, item in value.items():
            if name in {'text', 'canonical_text', 'analysis_text'} and isinstance(item, str):
                yield item
            else:
                yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def main():
    out = ROOT / 'data/cross_system_sentences_v80'
    if out.exists():
        raise FileExistsError(out)
    original = ROOT / 'data/teacher_indicf5_v14'
    pair = json.loads((original / 'model_pair.json').read_text())
    require_pair(pair)
    active = json.loads((ROOT / 'active_vowel_model.json').read_text())
    if sha(ROOT / active['model_directory'] / 'model.pt') != active['model_sha256']:
        raise ValueError('Frozen live detector identity differs')
    prior_texts, prior_pins = set(), {}
    for path in sorted((ROOT / 'data').glob('*/selection.json')):
        prior_texts.update(key(t) for t in strings(json.loads(path.read_text())))
        prior_pins[str(path.relative_to(ROOT))] = sha(path)
    used = set()
    for base in ('dual_geometry_features_v30', 'paired_local_features_v27'):
        for domain in ('physical', 'natural', 'indic'):
            for part in ('train', 'development'):
                _, _, rows, _ = partition(ROOT / 'reports' / base, domain, part, True)
                used.update(row['word'] for row in rows)
    swar = {w['word'] for w in json.loads((ROOT / 'data/swar_phone_plan_v18/words.json').read_text())}
    frontend = DualG2P(ROOT)
    vocabulary = [json.loads((ROOT / 'models' / name / 'vocab.json').read_text()) for name in ('phoneme_probe', 'phoneme_probe_lv60_v20')]
    contexts = []
    for number, (word, category, text) in enumerate(TEXTS):
        if key(text) in prior_texts or sum(token == word for token in key(text).split()) != 1:
            raise ValueError('Repeated historical sentence or non-unique target: ' + text)
        row = frontend.word(word)
        compiled = [compile_word(row, frontend.ipa, voc) for voc in vocabulary]
        default = [p for p in row['nepali_default']['phones'] if p != '.']
        sites = []
        for probe in compiled:
            variants = [v for v in probe['variants'] if v['phones'] == default]
            if len(variants) != 1 or not variants[0]['contrasts']:
                raise ValueError('Focal default lacks supported Hindi/Nepali contrast: ' + word)
            sites.append([(s['kind'], s['index']) for s in variants[0]['contrasts']])
        if sites[0] != sites[1]:
            raise ValueError('Probe focal contrast identities differ')
        if category == 'new_focal_word' and (word in used or word in swar):
            raise ValueError('New focal lexeme already in historical fit/dev or Swar: ' + word)
        contexts.append({'id': 'cross80_s' + str(number).zfill(2), 'text': text, 'target_word': word,
            'category': category, 'focal_word_in_prior_fit_or_dev': word in used, 'focal_word_in_swar': word in swar,
            'native_default_phones': default, 'hindi_units': row['hindi_units'], 'supported_focal_contrasts': sites[0],
            'realized_error_label': None, 'new_occurrence_requires_own_confirmation': True})
    if len({key(c['text']) for c in contexts}) != 26:
        raise ValueError('Expected26 unique fresh sentence contexts')
    refs = {}
    for job in json.loads((original / 'selection.json').read_text())['jobs']:
        if job['split'] == 'reserved_evaluation':
            refs[job['speaker_id']] = job
    if len(refs) != 4:
        raise ValueError('Verified four matched conditioning groups unavailable')
    refs = [refs[k] for k in sorted(refs)]
    for ref in refs:
        if sha(original / ref['reference_audio']) != ref['reference_sha256']:
            raise ValueError('Source conditioning bytes changed')
    jobs = []
    for c in contexts:
        for voice, ref in enumerate(refs):
            jobs.append({'id': c['id'] + '_v' + str(voice), 'text_id': c['id'], 'text': c['text'],
                'text_kind': c['category'], 'target_words': [c['target_word']], 'speaker_id': ref['speaker_id'],
                'voice_ordinal': voice, 'split': 'cross_system_discovery', 'reference_audio': ref['reference_audio'],
                'reference_text': ref['reference_text'], 'reference_sha256': ref['reference_sha256'],
                'source_reference_sha256': ref['source_reference_sha256'], 'seed': 0, 'automatic_leakage_label': None,
                'native_comparison_voice_ordinals': [(voice + 1) % 4, (voice + 2) % 4]})
    write_group(out, {'model_pair.json': pair, 'selection.json': {'experiment_id': 'teacher_indicf5_v14',
        'role': 'fresh_cross_system_discovery_v80_no_fitting', 'contexts': contexts, 'jobs': jobs},
        'protocol.json': {'frozen_detector_sha256': active['model_sha256'], 'sentence_contexts': 26,
            'context_categories': dict(Counter(c['category'] for c in contexts)), 'conditioning_groups': 4,
            'systems': list(pair['systems']), 'jobs_per_system': 104, 'outputs_total': 208,
            'comparison': 'Same text/prompts in Original IndicF5 and approved24-layer Nepali Pocket. Each query uses two other Native conditioning groups; no self waveform or own conditioning prompt in references.',
            'novelty': 'All sentence strings absent from prior selection manifests;10 focal spellings absent from both historical v31 task/fallback fit/development metadata and Swar. No morphological-root independence or unseen TTS-family claim.',
            'labels': 'All new realized pronunciation labels null, including known words in new contexts and teacher queries. Prior seven confirmations do not label new utterances. The पिढको condition is a variation check, not automatic-negative gold.',
            'content_policy': 'Retain lexical-content decisions per waveform. Gross rejects receive no pronunciation judgment; quarantine explicit. Unedited ASR checks never supply accent labels.',
            'success': 'Find localized candidates outside Swar on fresh audio, keep complete per-system/word/coverage denominators, offer same-text comparisons for listening. No accuracy claim without independent occurrence labels.',
            'production_changed': False, 'training': False, 'prosody': 'Diagnostic only'},
        'inputs.json': {'prior_selection_sha256': prior_pins, 'historical_fitdev_lexemes': len(used),
            'conditioning_selection_sha256': sha(original / 'selection.json'), 'source_script_sha256': sha(Path(__file__))}},
        'Fresh cross-system sentence discovery beyond Swar. selection.json26 authored sentence contexts/104 matched prompt jobs per system; protocol.json fixed comparison/labels/content/novelty; model_pair.json exact approved checkpoints; inputs.json prior metadata/source hashes. '
        'references/*.wav four verified copied conditioning prompts, reference_inventory.json file roles/source/SHA; teacher/ and indicf5/ generated receipts/audio/runtime with own README. No model weight copies. '
        'Creator Codex GPT-6 /root, owner Ampixa/user; review2026-11-10. Active consumer cross-system listening/generalization check; NOT on HF, inherited prompt/model/MMS research usage rights. '
        'Retain raw audio while this check runs; remove redundant copies within24h after consumer closes, review before7days. Source model families appeared in previous work; this is fresh-text discovery, not unseen-engine accuracy.',
        ['.venv-indicf5/bin/python -m scripts.build_cross_system_sentence_plan'])
    (out / 'references').mkdir()
    inventory = []
    for ref in refs:
        source = original / ref['reference_audio']
        target = out / ref['reference_audio']
        shutil.copyfile(source, target)
        if sha(target) != ref['reference_sha256']:
            raise ValueError('Copied conditioning bytes differ')
        inventory.append({'path': ref['reference_audio'], 'source': str(source), 'sha256': sha(target), 'bytes': target.stat().st_size,
                          'role': 'Shared matched TTS conditioning prompt; not a same-text pronunciation reference'})
    (out / 'reference_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({'contexts': len(contexts), 'jobs_per_system': len(jobs), 'outputs': len(jobs) * 2}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Cross-system plan failed explicitly before any generation')
        raise
