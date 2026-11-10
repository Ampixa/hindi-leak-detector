"""Verified, content-screened 24-layer teacher references for unseen inference text.

Only called on k2 by the inference CLI. It never changes the trained detector,
never treats a requested TTS pronunciation as gold, and refuses partial caches.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import unicodedata

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import require_pair, sha

LOG = logging.getLogger(__name__)
TEACHER_SHA = '2b4636e3aed0e4f2d035630e504be95fc58f494b690a851a91bfe0dd8299516f'


class ReferenceUnavailableError(ValueError):
    """Expected abstention when no generated reference passes content checks."""


def text_key(text: str) -> str:
    value = ' '.join(unicodedata.normalize('NFC', text).split())
    if not value or len(value) > 1000:
        raise ValueError('Need nonempty reference text of at most1000 characters')
    return value


def verified_paths(plan: Path, jobs: list[dict]) -> list[Path]:
    pair = json.loads((plan / 'model_pair.json').read_text())
    require_pair(pair)
    expected = pair['systems']['pocket_teacher_250000_24l']
    runtime = json.loads((plan / 'teacher/runtime.json').read_text())
    if (runtime['loaded_layers'] != 24 or runtime['weights_sha256'] != TEACHER_SHA
            or runtime['system'] != 'pocket_teacher_250000_24l'
            or runtime['tokenizer_sha256'] != expected['tokenizer_sha256']
            or runtime['source_revision'] != expected['source_revision']
            or runtime['settings'] != expected['inference']):
        raise ValueError('Cached reference source is not the approved24-layer teacher')
    rows = [json.loads(line) for line in (plan / 'teacher/items.jsonl').read_text().splitlines()]
    receipts = {row['id']: row for row in rows}
    if len(receipts) != len(rows):
        raise ValueError('Duplicate reference receipt identities')
    paths = []
    for job in jobs:
        row = receipts[job['id']]
        path = (plan / 'teacher' / row['audio']).resolve(strict=True)
        if not path.is_relative_to((plan / 'teacher').resolve()):
            raise ValueError('Reference waveform escaped its owned cache')
        if row['text'] != job['text'] or row['status'] != 'ok_unreviewed' or sha(path) != row['sha256']:
            raise ValueError('Cached reference receipt/audio identity differs')
        paths.append(path)
    return paths


def references_for_text(root: Path, text: str, *, generate: bool = True) -> tuple[list[Path], dict]:
    normalized = text_key(text)
    source = root / 'data/swar_teacher_v21'
    selection = json.loads((source / 'selection.json').read_text())
    matches = [job for job in selection['jobs'] if text_key(job['text']) == normalized]
    if matches:
        if len(matches) > 2:
            raise ValueError('Ambiguous cached text identity; supply explicit references')
        paths = verified_paths(source, matches)
        screening = root / 'reports/swar_teacher_content_v21/items.jsonl'
        rows = [json.loads(line) for line in screening.read_text().splitlines()]
        by_id = {row['id']: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError('Duplicate cached-reference content receipts')
        accepted = []
        for job, path in zip(matches, paths, strict=True):
            row = by_id[job['id']]
            if row['text'] != job['text'] or row['sha256'] != sha(path):
                raise ValueError('Cached-reference content check belongs to different audio/text')
            if row['content_quality']['decision'] == 'pass':
                accepted.append(path)
        if not accepted:
            raise ReferenceUnavailableError('Neither cached teacher reference passed lexical-content checks; no pronunciation judgment is supported')
        return accepted, {'mode': 'cached_verified_24_layer_teacher',
            'plan': str(source), 'new_reference_content_screen': None,
            'cached_reference_content_screen': str(screening), 'content_receipts_sha256': sha(screening),
            'reference_content_passes': len(accepted), 'generated_references': len(matches),
            'excluded_reference_decisions': [{'id': job['id'], 'decision': by_id[job['id']]['content_quality']['decision']}
                for job in matches if by_id[job['id']]['content_quality']['decision'] != 'pass']}
    if not generate:
        raise ValueError('No same-text reference is cached; omit --cached-only to generate or supply --reference')
    pair = json.loads((source / 'model_pair.json').read_text())
    require_pair(pair)
    prompts = []
    for job in selection['jobs']:
        if all(job['speaker_id'] != old['speaker_id'] for old in prompts):
            prompt = (source / job['reference_audio']).resolve(strict=True)
            if not prompt.is_relative_to((source / 'references').resolve()) or sha(prompt) != job['reference_sha256']:
                raise ValueError('Verified reference prompt changed')
            prompts.append(job)
        if len(prompts) == 2:
            break
    if len(prompts) != 2:
        raise ValueError('Need two distinct source conditioning groups')
    key = hashlib.sha256(json.dumps({'text': normalized, 'pair_sha256': sha(source / 'model_pair.json'),
        'prompts': [job['reference_sha256'] for job in prompts]}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    namespace = 'inference_ref_' + key
    plan = root / 'data' / namespace
    report = root / 'reports' / namespace
    if plan.exists():
        if not (report / 'ready.json').is_file():
            raise RuntimeError('Reference cache is incomplete or owned by another process; inspect ' + str(report))
    else:
        # Atomic ownership: a second process may read only the ready receipt.
        plan.mkdir()
        jobs = [{**job, 'id': namespace + '_v' + str(index), 'text_id': namespace,
                 'text': normalized, 'text_kind': 'inference_native_reference',
                 'target_words': [], 'split': 'external_inference_reference',
                 'automatic_leakage_label': None} for index, job in enumerate(prompts)]
        write_group(plan, {'selection.json': {'experiment_id': 'teacher_indicf5_v14',
            'role': 'inference_native_reference', 'jobs': jobs}, 'model_pair.json': pair,
            'reference_inventory.json': [{'path': job['reference_audio'],
                'source': str(source / job['reference_audio']), 'sha256': job['reference_sha256'],
                'bytes': (source / job['reference_audio']).stat().st_size} for job in prompts]},
            'Two generated same-text inference references, not labels. references/*.wav are exact '
            'two prompt copies listed in reference_inventory.json; teacher/ contains generated '
            'WAVs and separate synthesis receipts/README. Known duplicate L40 project plans/data '
            'under this exact namespace. Current consumer CLI inference. Status active; review '
            '2026-11-10; delete redundant generated/prompt copies within24h after consumer closes. '
            'No HF upload; inherited prompt/model/source/MMS usage restrictions apply.',
            [str(source), 'python -m scripts.detect_nepali_leakage --query WAV --text TEXT --out NEW'])
        (plan / 'references').mkdir()
        for job in prompts:
            destination = plan / job['reference_audio']
            shutil.copyfile(source / job['reference_audio'], destination)
            if sha(destination) != job['reference_sha256']:
                raise ValueError('New prompt copy failed SHA verification')
        write_group(report, {'request.json': {'text': normalized, 'plan': str(plan), 'namespace': namespace},
            'source_snapshot.json': [{'path': str(path.relative_to(root)), 'sha256': sha(path),
                'source_utf8': path.read_text()} for path in (Path(__file__), root / 'scripts/screen_teacher_reference.py',
                    root / 'scripts/run_swar_teacher_validation.py')]},
            'Automatic unseen-text reference request. controller/ has owned remote synthesis '
            'processes and receipts; controller.log and content_screen.log are subprocess output; '
            'content_screen/ records exact ASR identities/hypotheses and accepted/rejected references. '
            'ready.json permits reuse only after completed screening; failure.json records explicit '
            'failure. No silent retry, checkpoint fallback or pronunciation gold. Current consumer '
            'generic inference; review2026-11-10; transient logs expire24h after consumer closes.',
            [str(plan), 'Verified24-layer teacher; frozen trained detector unchanged'])
        try:
            commands = [([sys.executable, '-u', '-m', 'scripts.run_swar_teacher_validation',
                          '--plan', str(plan), '--out', str(report / 'controller'),
                          '--namespace', namespace, '--smoke', '2'], 'controller.log'),
                        ([str(root / '.venv-asr-gate/bin/python'), '-u', '-m', 'scripts.screen_teacher_reference',
                          '--plan', str(plan), '--out', str(report / 'content_screen')], 'content_screen.log')]
            for command, name in commands:
                LOG.info('Preparing two best-teacher references: %s; log %s', command[2:], report / name)
                with (report / name).open('xb') as handle:
                    subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=1200)
            screen = json.loads((report / 'content_screen/result.json').read_text())
            if not screen['accepted_ids']:
                raise ReferenceUnavailableError('Both generated Native references failed content screening')
            (report / 'ready.json').write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(),
                'selection_sha256': sha(plan / 'selection.json'), 'screen_sha256': sha(report / 'content_screen/result.json'),
                'accepted_ids': screen['accepted_ids']}, indent=2) + '\n')
        except Exception as exc:
            LOG.exception('Unseen-text reference preparation failed; cache retained for inspection at %s', report)
            (report / 'failure.json').write_text(json.dumps({'error': repr(exc),
                'created_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
            raise
    ready = json.loads((report / 'ready.json').read_text())
    if ready['selection_sha256'] != sha(plan / 'selection.json') or ready['screen_sha256'] != sha(report / 'content_screen/result.json'):
        raise ValueError('Completed reference cache identity changed')
    actual = json.loads((plan / 'selection.json').read_text())['jobs']
    if len(actual) != 2 or any(text_key(job['text']) != normalized for job in actual):
        raise ValueError('Generated reference text/count differs')
    accepted = [job for job in actual if job['id'] in ready['accepted_ids']]
    paths = verified_paths(plan, accepted)
    if not paths:
        raise ValueError('No accepted reference available')
    return paths, {'mode': 'generated_content_screened_24_layer_teacher', 'plan': str(plan),
        'new_reference_content_screen': str(report / 'content_screen/result.json'),
        'reference_content_passes': len(paths), 'generated_references': 2,
        'phonetic_site_qualification_still_required': True}
