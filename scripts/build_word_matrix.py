"""Lock a crossed word/context/voice experiment, separate from old benchmark text."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.select_development_data import normalized

ROOT = Path(__file__).resolve().parents[1]
FOCUS_WORDS = ['अमन', 'अमर', 'अर्थ', 'असल', 'उपाय', 'कष्ट', 'कर्म', 'कर्मचारी',
               'गमला', 'गरम', 'गृह', 'ग्रह', 'घटना', 'गहना', 'चिन्ह', 'चेतना',
               'चितवन', 'जन्म', 'कृष्ण', 'ज्येष्ठ', 'कसरत', 'कलह', 'कमला', 'उधार']
CARRIERS = [('statement', 'उनले {word} भन्ने शब्द पढे।'),
            ('question', 'के उनले {word} भन्ने शब्द पढे?'),
            ('contrast_text', 'उनले {word} भने, अर्को शब्द भनेनन्।'),
            ('reported_hearing', 'मैले फेरि {word} भन्ने शब्द सुनेँ।')]


def main() -> None:
    out = ROOT / 'data/wordmatrix_v3'
    if not (out / 'README.md').is_file() or (out / 'selection.json').exists():
        raise ValueError('Need a documented new matrix directory')
    candidates = json.loads((ROOT / 'data/schwa_candidates_v1/candidates.json').read_text())
    lookup = {r['word']: r for r in candidates}
    if set(FOCUS_WORDS) - set(lookup):
        raise ValueError('Authored focus list contains words absent from the inspected contrast queue')
    words = sorted(FOCUS_WORDS, key=lambda w: hashlib.sha256(('wm3-word:' + w).encode()).hexdigest())
    development = json.loads((ROOT / 'data/development_v1/selection.json').read_text())
    probe = json.loads((ROOT / 'data/native_probe_v2/selection.json').read_text())
    voice_maps = {'train': {}, 'development': {}}
    for job in development['jobs']:
        voice_maps[job['split']].setdefault(job['speaker_id'], {'speaker_id': job['speaker_id'],
            'reference_audio': job['reference_audio'], 'reference_text': job['reference_text'], 'reference_sha256': job['reference_sha256']})
    voice_groups = {'train': list(voice_maps['train'].values())[:8], 'development': list(voice_maps['development'].values())[:4],
                    'reserved_evaluation': [{'speaker_id': c['speaker_id'], 'reference_audio': c['audio'],
                                            'reference_text': c['text'], 'reference_sha256': c['sha256']} for c in probe['clips'][:4]]}
    if [len(voice_groups[s]) for s in voice_groups] != [8, 4, 4]:
        raise ValueError('Expected 8/4/4 verified reference groups')
    voices = [v for group in voice_groups.values() for v in group]
    if len({v['speaker_id'] for v in voices}) != 16:
        raise ValueError('Voice groups overlap')
    for voice in voices:
        if hashlib.sha256(Path(voice['reference_audio']).read_bytes()).hexdigest() != voice['reference_sha256']:
            raise ValueError('Reference identity mismatch')
    word_groups = dict(zip(voice_groups, [words[:8], words[8:16], words[16:]], strict=True))
    protected = {normalized(r['text_devanagari']) for r in json.loads((ROOT / '.sources/neptts/benchmark/sentences.json').read_text())}
    protected.update(normalized(j['text']) for j in development['jobs'])
    jobs, audio8, transfer = [], [], {}
    for word_index in range(8):
        for carrier_index, (intent, carrier) in enumerate(CARRIERS):
            for split, group in voice_groups.items():
                word = word_groups[split][word_index]; text = carrier.format(word=word)
                if normalized(text) in protected:
                    raise ValueError('Matrix prompt duplicates protected source text')
                for voice_index, voice in enumerate(group):
                    identifier = f'wm3_{split}_{word_index:02d}_{carrier_index}_{voice_index:02d}'
                    job = {'id': identifier, 'text': text, 'split': split, 'seed': 0, **voice,
                           'focus_word': word, 'focus_candidate_id': lookup[word]['candidate_id'],
                           'prompt_id': f'{split}_{word_index:02d}_{carrier_index}', 'intent_text': intent,
                           'pronunciation_gold': False, 'prompt_review_status': 'authored_unreviewed'}
                    jobs.append(job)
                    basename = Path(voice['reference_audio']).name
                    transfer[basename] = {'name': basename, 'source_path': voice['reference_audio'], 'sha256': voice['reference_sha256']}
                    audio8.append({**{k: job[k] for k in ['id', 'text', 'split', 'seed', 'speaker_id', 'focus_word', 'prompt_id', 'intent_text']},
                                   'reference': {'audio': '/home/ampixa/hindi-detector-wordmatrix-v3/references/' + basename,
                                                 'text': voice['reference_text'], 'sha256': voice['reference_sha256']}})
    if len(jobs) != 512:
        raise ValueError('Unexpected matrix size')
    common = {'experiment_id': 'wordmatrix_v3', 'creator': 'Codex GPT-6 /root',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'purpose': 'Crossed voice/text/context source-proxy experiment; not human word-accent gold',
              'counts': {'voices': 16, 'focus_words': 24, 'carriers_per_word': 4, 'jobs_per_tts': 512, 'tts_systems': 2},
              'word_groups': word_groups, 'references': list(transfer.values()),
              'limits': ['Focus words come from unreviewed Hindi/Nepali dictionary contrasts, not certified schwa targets.',
                         'Authored carrier punctuation suggests a style; it does not guarantee generated emphasis/intonation.',
                         'Same reference WAVs condition both systems, but successful preservation of voice identity is not assumed.',
                         'Distinct source speaker IDs do not independently establish cross-corpus real-person identity.']}
    (out / 'selection.json').write_text(json.dumps({**common, 'schema_version': 1, 'jobs': jobs, 'clips': []}, ensure_ascii=False, indent=2) + '\n')
    (out / 'audio8_manifest.json').write_text(json.dumps({**common, 'jobs': audio8}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(common['counts']))


if __name__ == '__main__':
    main()
