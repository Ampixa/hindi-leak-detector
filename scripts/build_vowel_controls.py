"""Actual same-speaker vowel interventions for a source-independent site head."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.phone_lattice import VOWELS, path_score, units_for_project
from hindi_detector.pronunciation_sites import (FEATURE_NAMES, evidence_for_site, paired_site_features,
    select_reference_structure, windows)
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT = Path(__file__).resolve().parents[1]


def identity(value):
    return hashlib.sha256(value.encode()).hexdigest()


def split_for(voice, word):
    fold = int(identity(word)[:8], 16) % 100
    if voice in {'yt01', 'yt10', 'yt12'} or fold >= 85:
        return 'test'
    if voice == 'yt11' or fold >= 70:
        return 'development'
    return 'train'


def splice(samples, start, end, chunk):
    if not 0 < start < end < len(samples) or not np.isfinite(chunk).all():
        raise ValueError('Invalid actual intervention geometry')
    width = min(32, start, len(samples)-end)
    left, right = samples[:start].copy(), samples[end:].copy()
    left[-width:] *= np.linspace(1, 0, width)
    right[:width] *= np.linspace(0, 1, width)
    chunk = chunk.copy()
    if len(chunk):
        fade = min(width, len(chunk)//4)
        if fade < 2:
            raise ValueError('Replacement nucleus too short')
        chunk[:fade] *= np.linspace(0, 1, fade)
        chunk[-fade:] *= np.linspace(1, 0, fade)
    return np.concatenate([left, chunk, right])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-failed', action='store_true')
    args = parser.parse_args()
    out = ROOT/'reports/vowel_controls_v19'
    base = ROOT/'reports/swar_corpus_features_v18'
    plan = ROOT/'data/swar_phone_plan_v18'
    if out.exists() and not args.resume_failed:
        raise FileExistsError(out)
    if args.resume_failed:
        previous = json.loads((out/'status.json').read_text())
        if previous['phase'] != 'failed_explicitly' or not any(message in previous.get('error', '') for message in ('Site hypotheses cannot both fit measured word frames', 'Invalid measured phone crop')):
            raise ValueError('Only the inspected short-word path failure may resume')
        write_group(out/('resume_invalid_crop' if 'Invalid measured phone crop' in previous.get('error','') else 'resume_short_words'), {'previous_failure.json': previous,
            'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
            'Resume inspected impossible short-word CTC paths. Explicit whole-pair exclusions never become labels. '
            'Frozen intervention plan and split unchanged; previous complete pairs retained. Review2026-11-09.',
            ['--resume-failed; identical source audio, nuclei, donors and partition plan'])
    if not args.resume_failed:
        write_group(out, {'protocol.json': {
            'labels': 'Known intact/replaced or removed vowel segment; not independently labeled Hindi-error speech',
            'negative': 'Same-speaker native-category donor for quality, or sham-reinserted original nucleus for deletion',
            'positive': 'Same-speaker disjoint-vowel donor for quality, or removed observed nucleus for deletion',
            'regression_exclusions': ['हैन', 'किनकि'], 'all_user_pick_texts_excluded': True,
            'speaker_test': ['yt01', 'yt10', 'yt12'], 'speaker_development': ['yt11'],
            'lexeme_fold': 'sha256(word) first8hex modulo100: <70 train,70..84 dev,>=85 test; speaker exclusions take priority',
            'fit_features': FEATURE_NAMES, 'raw_voice_or_generator_identity_features': False,
            'control_boundary': '2ms fades at actual nucleus replacement joins, same recipe for both labels',
            'prior_model_or_actual_swar_error_labels_used_for_fitting': False},
            'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
                for p in [Path(__file__), ROOT/'hindi_detector/phone_lattice.py', ROOT/'hindi_detector/pronunciation_sites.py']]},
            'Known actual same-speaker acoustic vowel controls. plan.json pins original word/site/nucleus, same-speaker '
            'donor, split and intervention geometry. controls/*.wav are actual16kPCM16 paired audio, per-wave SHA/bytes '
            'in items.jsonl. arrays.npz and index.json contain measured site features and intervention labels; numeric '
            'columns exclude word/voice/source IDs. runtime.json pins encoders. report/status show exact coverage. '
            'No ASR filtering of intended changed words. Detection of these controlled changes is not real Hindi-error '
            'accuracy; full-word alignment and probe biases remain. Current consumer vowel-site head; review2026-11-09, '
            'delete redundant control waveforms after evaluation consumer closes, retain model/metrics/config/pins. '
            'No HF upload; inherited source/MMS research licences.',
            ['.venv-indicf5/bin/python -m scripts.build_vowel_controls',
             'Frozen actual same-text corpus, dual-G2P inventory, fixed recorded-speaker/word holdouts'])
        (out/'controls').mkdir()

    def status(phase, **extra):
        p = out/'status.json'; temp = p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(),
            'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2)+'\n'); temp.replace(p)

    try:
        status('waiting_for_complete_actual_features')
        while True:
            state = json.loads((base/'status.json').read_text())
            if state['phase'] == 'failed_explicitly':
                raise RuntimeError('Actual feature source failed')
            if state['phase'] == 'complete_actual_picks_features':
                break
            os.kill(state['pid'], 0); time.sleep(5)
        runtime = json.loads((base/'runtime.json').read_text())
        vocab = runtime['vocabulary']; vowel_ids = frozenset(vocab[t] for t in VOWELS if t in vocab)
        ipa = json.loads((plan/'ipa_map.json').read_text())
        lexical = {r['word']: r for r in json.loads((plan/'words.json').read_text())}
        selection = {r['id']: r for r in json.loads((ROOT/'data/swar_speech_v18/selection.json').read_text())}
        receipts = [r for r in read_rows(base/'items.jsonl') if r['kind'] == 'real']
        cache = {r['id']: load_item(base, r) for r in receipts}
        waves = {r['id']: audio(ROOT/'data/swar_speech_v18', {'audio': r['path'], 'sha256': r['sha256']}) for r in receipts}
        segments, exclusions = [], Counter()
        for key, item in cache.items():
            voice = selection[key]['voice']
            for ordinal, word in enumerate(item['detail']['words']):
                row = lexical.get(word['word'])
                if row is None or row['nepali_default']['source'] != 'lexicon':
                    exclusions['not_lexicon'] += 1; continue
                variants = []
                for number, phones in enumerate(row['nepali_project_variants']):
                    try:
                        units, projections = units_for_project(phones, ipa, vocab)
                    except ValueError:
                        exclusions['unsupported_native_variant'] += 1; continue
                    variants.append({'variant_id': number, 'phones': [p for p in phones if p != '.'],
                                     'units': units, 'projections': projections, 'contrasts': []})
                if not variants:
                    continue
                compiled = {'word': word['word'], 'variants': variants, 'any_vowel': vowel_ids}
                crops = windows(item, ordinal)
                try:
                    variant, reduced = select_reference_structure(compiled, crops)
                    broad = [vowel_ids if p == 'ax' else u for p, u in zip(variant['phones'], variant['units'])]
                    middle = crops[1]
                    trace = path_score(middle[2], broad, trace=True)
                except ValueError:
                    exclusions['unrepresentable_native_word'] += 1; continue
                spans = trace['spans']; dt = item['detail']['phone_frame_seconds']
                for i, unit in enumerate(variant['units']):
                    if not unit <= vowel_ids or i in reduced:
                        continue
                    span = spans[i]
                    left = (spans[i-1]['end_frame']+span['start_frame'])/2 if i else span['start_frame']
                    right = (span['end_frame']+spans[i+1]['start_frame'])/2 if i+1 < len(spans) else span['end_frame']
                    start = int((middle[1]+left)*dt*16000); end = int((middle[1]+right)*dt*16000)
                    if not .04 <= (end-start)/16000 <= .28 or not 0 < start < end < len(waves[key]):
                        exclusions['unsupported_nucleus_geometry'] += 1; continue
                    if span['mean_log_emission'] < -1.5:
                        exclusions['weak_nucleus_anchor'] += 1; continue
                    absolute = middle[1]+span['start_frame'], middle[1]+span['end_frame']
                    posterior = np.exp(item['phone_log_probs'][absolute[0]:absolute[1]])
                    mass = posterior[:, sorted(unit)].sum() / max(1e-12, posterior[:, sorted(vowel_ids)].sum())
                    segments.append({'category_supported': bool(mass >= .6), 'key': key, 'voice': voice, 'word': word['word'], 'ordinal': ordinal,
                        'unit_index': i, 'start': start, 'end': end, 'unit': unit, 'compiled': compiled,
                        'variant': variant, 'reduced': reduced, 'crops': crops})
        segments.sort(key=lambda s: identity(s['key']+':'+str(s['ordinal'])+':'+str(s['unit_index'])))
        donors = defaultdict(list)
        for segment in segments:
            if segment['category_supported']:
                donors[segment['voice']].append(segment)
        interventions = []
        seen_quality_words = set()
        for target in segments:
            if selection[target['key']]['user_pick'] or target['word'] in {'हैन', 'किनकि'}:
                continue
            common = {'key': target['key'], 'word': target['word'], 'ordinal': target['ordinal'],
                'unit_index': target['unit_index'], 'voice': target['voice'], 'start': target['start'], 'end': target['end'],
                'split': split_for(target['voice'], target['word'])}
            if target['variant']['phones'][target['unit_index']] == 'ax':
                interventions.append({'target': target, 'kind': 'vowel_deletion', 'native_donor': target,
                                      'other_donor': None, 'metadata': common})
            if not target['category_supported'] or (target['key'], target['ordinal']) in seen_quality_words:
                continue
            pool = [s for s in donors[target['voice']] if s['key'] != target['key'] or s['ordinal'] != target['ordinal']]
            comparable = [s for s in pool if .5 <= (s['end']-s['start'])/(target['end']-target['start']) <= 2]
            native = [s for s in comparable if s['unit'] == target['unit']]
            other = [s for s in comparable if not s['unit'] & target['unit']]
            if native and other:
                distance = lambda s: abs(np.log((s['end']-s['start'])/(target['end']-target['start'])))
                nd, od = min(native, key=distance), min(other, key=distance)
                interventions.append({'target': target, 'kind': 'vowel_quality', 'native_donor': nd,
                                      'other_donor': od, 'metadata': common})
                seen_quality_words.add((target['key'], target['ordinal']))
        def clean(segment):
            return {k: segment[k] for k in ('key', 'voice', 'word', 'ordinal', 'unit_index', 'start', 'end')}
        frozen = [{**x['metadata'], 'kind': x['kind'], 'native_donor': clean(x['native_donor']),
                   'other_donor': clean(x['other_donor']) if x['other_donor'] else None} for x in interventions]
        if args.resume_failed:
            if frozen != json.loads((out/'plan.json').read_text()):raise ValueError('Frozen intervention plan changed during resume')
        else:(out/'plan.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2)+'\n')
        status('encoding_actual_vowel_controls', planned_pairs=len(interventions), source_nuclei=len(segments))
        engine = WordEvidenceEngine(ROOT, device='mps', pitch_method='dio', temporal_bins=20)
        if engine.phone_inventory != runtime['phone']:
            raise ValueError('Control and actual-query phone encoders differ')
        (out/'runtime.json').write_text(json.dumps({'phone': engine.phone_inventory,
            'device': 'mps', 'frame_recipe': 'Full actual edited utterance, no word-only encoding',
            'feature_recipe': FEATURE_NAMES, 'source_runtime_sha256': sha(base/'runtime.json')}, indent=2)+'\n')
        previous_items = read_rows(out/'items.jsonl') if args.resume_failed else []
        done_pairs = Counter((r['key'], r['ordinal'], r['unit_index'], r['kind']) for r in previous_items)
        if any(n != 2 for n in done_pairs.values()):raise ValueError('Incomplete prior pair requires explicit review')
        features = [np.array(r['features'], np.float32) for r in previous_items]
        labels = [r['label'] for r in previous_items]
        index_keys = ('row_index', 'key', 'word', 'ordinal', 'unit_index', 'voice', 'split', 'kind', 'label', 'audio', 'audio_sha256')
        indices = [{k: r[k] for k in index_keys} for r in previous_items]
        def exclude(number, error, files):
            logging.warning('Explicit whole-pair CTC exclusion %s: %s', frozen[number], error)
            with (out/'pair_exclusions.jsonl').open('a') as f:f.write(json.dumps({'plan_index':number, 'input':frozen[number], 'reason':str(error), 'disposable_audio_files':files}, ensure_ascii=False)+'\n')
        excluded_indices = {r['plan_index'] for r in read_rows(out/'pair_exclusions.jsonl')}
        for number, intervention in enumerate(interventions):
            if number in excluded_indices:continue
            target = intervention['target']; key = target['key']; kind = intervention['kind']
            if (key, target['ordinal'], target['unit_index'], kind) in done_pairs:continue
            orphan_files = [str((out/'controls'/(identity(key+':'+str(target['ordinal'])+':'+str(target['unit_index'])+':'+kind+':'+str(label))[:24]+'.wav')).relative_to(out)) for label in (0,1) if (out/'controls'/(identity(key+':'+str(target['ordinal'])+':'+str(target['unit_index'])+':'+kind+':'+str(label))[:24]+'.wav')).exists()]
            if orphan_files:
                exclude(number, 'Uncommitted pair from inspected prior failure; retained as disposable diagnostics', orphan_files);continue
            original_word = cache[key]['detail']['words'][target['ordinal']]
            if kind == 'vowel_deletion' and original_word['end']-(target['end']-target['start'])/16000 <= original_word['start']+.02:
                exclude(number, 'Nucleus removal would eliminate the aligned word; no valid vowel-only control', []);continue
            i = target['unit_index']; variant = target['variant']; compiled = target['compiled']
            contrast = {'kind': kind, 'index': i, 'expected': variant['projections'][i]['ipa'],
                        'alternative': None if kind == 'vowel_deletion' else 'same-speaker disjoint vowel category'}
            if kind == 'vowel_quality':
                contrast['alternative_unit'] = intervention['other_donor']['unit']
            try:
                reference = evidence_for_site(compiled, variant, target['reduced'], contrast, target['crops'])
            except ValueError as error:
                if str(error) not in {'Site hypotheses cannot both fit measured word frames', 'Invalid measured phone crop'}:raise
                exclude(number, error, []);continue
            pair_records = []; made_files = []
            for label in (0, 1):
                donor = intervention['native_donor'] if not label else intervention['other_donor']
                chunk = waves[donor['key']][donor['start']:donor['end']] if donor else np.empty(0, np.float32)
                edited = splice(waves[key], target['start'], target['end'], chunk)
                delta = len(chunk)-(target['end']-target['start'])
                file = out/'controls'/(identity(key+':'+str(target['ordinal'])+':'+str(i)+':'+kind+':'+str(label))[:24]+'.wav')
                if file.exists():raise FileExistsError('Control output identity already exists')
                sf.write(file, edited, 16000, subtype='PCM_16')
                made_files.append(str(file.relative_to(out)))
                samples = sf.read(file, dtype='float32', always_2d=False)[0]
                with engine.torch.inference_mode():
                    encoded = engine.phone(**{k: v.to(engine.device) for k, v in
                        engine.extractor(samples, sampling_rate=16000, return_tensors='pt').items()})
                    lp = encoded.logits[0].log_softmax(-1).cpu().numpy()
                if not np.isfinite(lp).all():
                    raise ValueError('Nonfinite actual control phone frames')
                original_word = cache[key]['detail']['words'][target['ordinal']]
                modified_word = {**original_word, 'end': original_word['end']+delta/16000}
                item = {'detail': {'words': [modified_word], 'phone_frame_seconds': len(samples)/16000/len(lp)},
                        'phone_log_probs': lp}
                try:
                    evidence = evidence_for_site(compiled, variant, target['reduced'], contrast, windows(item, 0))
                except ValueError as error:
                    if str(error) not in {'Site hypotheses cannot both fit measured word frames', 'Invalid measured phone crop'}:raise
                    exclude(number, error, made_files);break
                vector = paired_site_features(evidence, reference, contrast)
                record = {**frozen[number], 'label': label, 'row_index': len(features)+len(pair_records),
                    'audio': str(file.relative_to(out)), 'audio_sha256': sha(file), 'audio_bytes': file.stat().st_size,
                    'query_evidence': evidence, 'reference_evidence': reference, 'features': vector.tolist(),
                    'actual_hindi_error_label': None}
                pair_records.append(record)
            if len(pair_records) != 2:continue
            for record in pair_records:
                with (out/'items.jsonl').open('a') as f:
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                features.append(np.array(record['features'], np.float32)); labels.append(record['label'])
                indices.append({k: record[k] for k in index_keys})
            status('encoding_actual_vowel_controls', completed_pairs=number+1, planned_pairs=len(interventions),
                   split_pairs=dict(Counter(x['metadata']['split'] for x in interventions[:number+1])))
        np.savez_compressed(out/'arrays.npz', features=np.stack(features), labels=np.array(labels, np.int64))
        (out/'index.json').write_text(json.dumps(indices, ensure_ascii=False)+'\n')
        report = {'status': 'complete_actual_same_speaker_vowel_controls', 'pairs': len(indices)//2, 'planned_pairs': len(interventions),
            'split_pairs': {p: sum(r['split']==p for r in indices)//2 for p in ('train','development','test')},
            'split_lexemes': {part: len({x['metadata']['word'] for x in interventions if x['metadata']['split']==part}) for part in ('train','development','test')},
            'kinds': dict(Counter(x['kind'] for x in interventions)), 'exclusions': dict(exclusions),
            'array_sha256': sha(out/'arrays.npz'), 'index_sha256': sha(out/'index.json'),
            'actual_hindi_error_accuracy': None}
        (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        status('complete_actual_vowel_controls', pairs=len(indices)//2, planned_pairs=len(interventions))
    except Exception:
        status('failed_explicitly', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Actual same-speaker vowel controls failed explicitly')
        raise
