"""Select region-diverse human word examples with two explicit G2P predictions."""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata

import soundfile as sf

from hindi_detector.artifacts import write_group

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'data/human_regions_v5'
OUT = ROOT / 'data/human_g2p_bank_v5'
STATE = ROOT / 'reports/human_g2p_selection_v5'
LOG = logging.getLogger(__name__)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(phase, **fields):
    p = STATE / 'status.json'; temporary = p.with_name(p.name + '.partial')
    temporary.write_text(json.dumps({'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(), **fields}, ensure_ascii=False, indent=2) + '\n'); temporary.replace(p)


def main():
    if not (STATE / 'README.md').is_file() or OUT.exists():
        raise ValueError('Need fresh human bank and documented selector state')
    status('waiting_for_human_metadata')
    while True:
        p = META / 'status.json'
        state = json.loads(p.read_text()) if p.exists() else {}
        if state.get('status') == 'complete_metadata_only' and (META / 'inventory.json').is_file():
            break
        if state.get('status', '').startswith('failed'):
            raise RuntimeError('Human metadata recovery failed; inspect before selection')
        time.sleep(15)
    for member in json.loads((META / 'inventory.json').read_text()):
        p = (META / member['path']).resolve(strict=True)
        if not p.is_relative_to(META.resolve()) or p.stat().st_size != member['bytes'] or sha(p) != member['sha256']:
            raise ValueError('Recovered human metadata inventory changed')
    status('indexing_real_human_word_contexts')
    lock = json.loads((ROOT / 'sources.lock.json').read_text())
    for source in lock['sources']:
        if source['id'] not in {'ampixa_frontend', 'hindi_lexicon'}:
            continue
        for member in source['files']:
            p = ROOT / source['directory'] / member['path']; b = p.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
            if len(b) != member['size_bytes'] or blob != member['git_blob_sha1']:
                raise ValueError('Pinned G2P resource changed: ' + str(p))
    frontend = ROOT / '.sources/frontend'
    sys.path.insert(0, str(frontend))
    ph = importlib.import_module('nepali_frontend.g2p.phonemizer')
    if not Path(ph.__file__).resolve().is_relative_to(frontend.resolve()):
        raise ValueError('Wrong Nepali G2P module imported')
    with (frontend / 'data/frontend/ipa_map.tsv').open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != ['phone', 'ipa_candidate', 'status', 'notes']:
            raise ValueError('Nepali IPA candidate map schema changed')
        ipa = {row['phone']: row['ipa_candidate'] for row in reader}
    candidates = json.loads((ROOT / 'data/schwa_candidates_v1/candidates.json').read_text())
    candidate_map = defaultdict(list)
    for row in candidates:
        candidate_map[row['word']].append(row)
    dictionary = defaultdict(list)
    with (ROOT / '.sources/hindi/hindi/data/extra_large.csv').open() as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['hindi', 'phon']:
            raise ValueError('Hindi dictionary schema changed')
        for number, row in enumerate(reader, 2):
            word = unicodedata.normalize('NFC', row['hindi'])
            if word in candidate_map:
                dictionary[word].append({'csv_row': number, 'raw_spelling': row['hindi'], 'phones_ascii': row['phon'].split()})
    excluded = set()
    for name, key in [('development_v1', 'jobs'), ('native_probe_v2', 'clips'), ('native_wordbank_v3', 'clips')]:
        data = json.loads((ROOT / 'data' / name / 'selection.json').read_text())
        excluded.update(r['speaker_id'] for r in data[key])
    speaker_rows = json.loads((META / 'speaker_regions.json').read_text())
    speakers = {r['speaker_id']: r for r in speaker_rows}
    if len(speakers) != len(speaker_rows):
        raise ValueError('Duplicate source speaker region mapping')
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    considered = 0
    for row in json.loads((META / 'single_utterance_candidates.json').read_text()):
        speaker = row['speaker_id']
        if speaker in excluded or speakers[speaker]['status'] != 'unique_reported_region' or not 1 <= row['local_seconds'] <= 15:
            continue
        if not row['state'] or not row['district'] or any(unicodedata.category(c)[0] == 'N' for c in row['text']):
            continue
        words = [''.join(c for c in token if unicodedata.category(c)[0] in {'L', 'M'}) for token in row['text'].split()]
        considered += 1
        for word in set(words) & set(candidate_map):
            groups[word][(row['state'], row['district'])][speaker].append(row)
    coverage = [{'word': w, 'source_speakers': len({s for group in regions.values() for s in group}), 'reported_regions': len(regions),
                 'candidate_contexts': sum(len(rows) for group in regions.values() for rows in group.values())} for w, regions in groups.items()]
    eligible = [r['word'] for r in sorted(coverage, key=lambda r: (-r['reported_regions'], -r['source_speakers'], r['word'])) if r['source_speakers'] >= 6 and r['reported_regions'] >= 2][:40]
    if not eligible:
        raise ValueError('No six-speaker/two-region words; preserve coverage audit rather than invent examples')
    exe = Path('/opt/homebrew/bin/espeak-ng').resolve(strict=True)
    version = subprocess.check_output([str(exe), '--version'], text=True)
    if not version.startswith('eSpeak NG text-to-speech: 1.52.0 '):
        raise ValueError('Hindi G2P version differs from inspected runtime')
    match = re.search(r'Data at:\s*(.+)', version)
    if not match:
        raise ValueError('Cannot locate inspected eSpeak Hindi data')
    data_root = Path(match.group(1).strip()).resolve(strict=True)
    resource_paths = [exe, *[data_root / name for name in ['hi_dict', 'lang/inc/hi', 'phondata', 'phontab', 'phonindex', 'intonations']], *list((exe.parent.parent / 'lib').glob('libespeak-ng*.dylib'))]
    resources = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted({p.resolve(strict=True) for p in resource_paths})]
    bank = []; selected = {}; failures = []
    status('sampling_regions_and_running_both_g2ps', eligible_word_families=len(eligible))
    for word in eligible:
        regions = groups[word]
        region_keys = sorted(regions)
        for group in regions.values():
            for rows in group.values():
                rows.sort(key=lambda r: (r['local_seconds'], r['audio']))
        people = {region: sorted(regions[region], key=lambda s: hashlib.sha256((word + ':' + s).encode()).hexdigest()) for region in region_keys}
        examples = []; used = set()
        per_region = math.ceil(8 / len(region_keys))
        for offset in range(per_region):
            for region in region_keys:
                if len(examples) >= 8:
                    break
                candidates_for_region = [s for s in people[region] if s not in used]
                if not candidates_for_region:
                    continue
                speaker = candidates_for_region[0]; used.add(speaker)
                row = regions[region][speaker][0]
                try:
                    path = Path(row['audio']).resolve(strict=True)
                    if not path.is_relative_to((ROOT.parent / 'indicvoices_r_ne/audio').resolve()):
                        raise ValueError('Human waveform outside declared corpus')
                    info = sf.info(path)
                    if info.channels != 1 or abs(info.duration - row['local_seconds']) > .02:
                        raise ValueError('Human waveform duration/channel mismatch')
                    if str(path) not in selected:
                        selected[str(path)] = {**row, 'audio': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size, 'sample_rate': info.samplerate}
                    examples.append({'audio': str(path), 'speaker_id': speaker, 'state': region[0], 'district': region[1]})
                except (OSError, RuntimeError, ValueError) as error:
                    failures.append({'word': word, 'audio': row['audio'], 'error_type': type(error).__name__, 'error': str(error)})
                    LOG.exception('Human source QC failed; example explicitly excluded')
        if len(examples) < 6 or len({(e['state'], e['district']) for e in examples}) < 2:
            continue
        nepali = ph.phonemize_word(word, style='spoken_nepali')
        hindi = subprocess.run([str(exe), '--path=' + str(data_root.parent), '-q', '-v', 'inc/hi', '--ipa', '--sep=|', word], capture_output=True, text=True, check=True)
        if not hindi.stdout.strip() or hindi.stderr.strip():
            raise ValueError('Hindi G2P returned empty output or an unexplained warning for ' + word)
        bank.append({'word': word, 'nepali_g2p': {'project_phones': nepali.phones, 'source': nepali.source,
                                               'frontend_reported_confidence': nepali.confidence,
                                               'ipa_candidate_units': [ipa.get(p) if p != '.' else '.' for p in nepali.phones]},
                     'hindi_g2p': {'engine': 'espeak-ng 1.52.0 inc/hi', 'raw_ipa_stdout': hindi.stdout,
                                   'units_with_stress_preserved': hindi.stdout.strip().split('|')},
                     'independent_hindi_lexicon_variants': dictionary[word],
                     'screening_schwa_candidates': candidate_map[word], 'human_examples': examples,
                     'actual_realization_labels': None, 'g2p_predictions_are_ground_truth': False,
                     'separate_judgments': {'native_variant_support': None, 'preferred_tts_style_distance': None,
                                            'hindi_phone_hypothesis_evidence': None}})
    if not bank:
        raise ValueError('Human source QC left no usable contrast families')
    used_paths = {e['audio'] for row in bank for e in row['human_examples']}
    jobs, clips = [], []
    for path in sorted(used_paths):
        row = selected[path]; identifier = 'human_g2p_v5_' + hashlib.sha256(path.encode()).hexdigest()[:20]
        jobs.append({'id': identifier, 'speaker_id': row['speaker_id'], 'text': row['text'], 'split': 'development',
                     'target_words': [b['word'] for b in bank if any(e['audio'] == path for e in b['human_examples'])]})
        clips.append({'clip_id': identifier, 'sentence_id': identifier, 'system': 'human_nepali_unlabeled',
                      'speaker_id': row['speaker_id'], 'text': row['text'], 'audio': path, 'sha256': row['sha256'],
                      'bytes': row['bytes'], 'split': 'development', 'automatic_leakage_label': None})
    report = {'status': 'regional_human_g2p_contrast_bank_selected_unlabeled', 'word_families': len(bank),
              'unique_recordings': len(clips), 'source_groups': len({c['speaker_id'] for c in clips}),
              'reported_regions': sorted({(e['state'], e['district']) for b in bank for e in b['human_examples']}),
              'eligible_single_utterance_contexts': considered, 'waveform_qc_failures': failures,
              'excluded_prior_source_groups': sorted(excluded), 'word_coverage': coverage,
              'metadata_revision': json.loads((META / 'audit.json').read_text())['revision'],
              'output_contract': ['Broad native Nepali support, including regional varieties', 'Deviation from the preferred TTS pronunciation style', 'Independent Hindi-phone hypothesis evidence'],
              'limits': ['All human examples are unlabeled for Hindi leakage; region never supplies a pronunciation label.',
                         'Each example matches one original utterance by speaker/text/duration, not verified raw audio checksum identity.',
                         'G2P outputs are alternative priors. Real human audio must determine realized phones and legitimate variants.',
                         'Hindi dictionary aliases used for candidate mining erase distinctions; raw Hindi G2P/lexicon outputs are preserved separately.',
                         'The named Newa repository implements Nepali here; this is not a Nepal Bhasa pronunciation bank.']}
    source = Path(__file__)
    write_group(OUT, {'selection.json': {'schema_version': 1, 'jobs': jobs, 'clips': clips, 'external_audio_roots': [str(ROOT.parent / 'indicvoices_r_ne/audio')]},
                      'word_bank.json': bank, 'report.json': report, 'g2p_runtime.json': {'version': version, 'resources': resources},
                      'source_snapshot.json': {'path': 'scripts/' + source.name, 'sha256': sha(source), 'source_utf8': source.read_text()}},
                'Human word-realization candidates sampled across reported regions using two explicit G2P predictions; no accent labels inferred.',
                ['Human metadata inventory SHA ' + sha(META / 'inventory.json'), 'sources.lock.json',
                 'Existing /opt/homebrew espeak-ng 1.52.0; no installation or new model download',
                 'python -m scripts.select_human_g2p_bank'])
    status('human_bank_selected_unlabeled', word_families=len(bank), recordings=len(clips))
    print(json.dumps({k: report[k] for k in ['word_families', 'unique_recordings', 'source_groups', 'reported_regions']}, ensure_ascii=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Human G2P selection failed; no fabricated people, regions or realizations')
        if STATE.is_dir():
            status('failed_inspect_logs_before_retry')
        raise
