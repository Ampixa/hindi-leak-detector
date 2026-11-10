"""Frozen best-teacher reference requests for the actual Swar speech texts."""
from __future__ import annotations

from collections import defaultdict
import json
import logging
from pathlib import Path
import shutil

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import require_pair, sha
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/'data/swar_teacher_v21'
    if out.exists():
        raise FileExistsError(out)
    source = ROOT/'data/swar_speech_v18'
    baseline = ROOT/'data/native_variation_v16'
    pair = json.loads((baseline/'model_pair.json').read_text())
    require_pair(pair)
    old = json.loads((baseline/'selection.json').read_text())
    references = {}
    for job in old['jobs']:
        references[job['speaker_id']] = job
    if len(references) != 2:
        raise ValueError('Inspected two-voice teacher reference plan differs')
    selection = json.loads((source/'selection.json').read_text())
    eligible = {r['id'] for r in read_rows(ROOT/'reports/swar_corpus_features_v18/items.jsonl')}
    contexts = [r for r in selection if r['id'] in eligible]
    focus = ['real_yt_yt01_005','real_yt_yt01_008','real_yt_yt04_030']
    contexts.sort(key=lambda r: (focus.index(r['id']) if r['id'] in focus else len(focus), r['id']))
    words = defaultdict(set)
    for row in read_rows(ROOT/'reports/swar_sites_v18/items.jsonl'):
        words[row['id']].add(row['word'])
    jobs, inventory, sources = [], [], {}
    for context in contexts:
        for ordinal, reference in enumerate(references.values()):
            path = baseline/reference['reference_audio']
            if sha(path) != reference['reference_sha256']:
                raise ValueError('Pinned teacher conditioning prompt changed')
            sources[path.name] = path
            jobs.append({'id': 'swarref21_'+context['id']+'_v'+str(ordinal),
                'text_id': context['id'], 'text': context['text'], 'text_kind': 'actual_swar_reference',
                'target_words': sorted(words[context['id']]), 'speaker_id': reference['speaker_id'],
                'split': 'external_swar_reference', 'reference_audio': 'references/'+path.name,
                'reference_sha256': reference['reference_sha256'],
                'source_reference_sha256': reference['source_reference_sha256'],
                'reference_text': reference['reference_text'], 'seed': 0, 'automatic_leakage_label': None})
    if len(contexts) != 86 or len(jobs) != 172:
        raise ValueError('Frozen inspected eligible Swar dimensions differ')
    write_group(out, {'model_pair.json': pair, 'selection.json': {'experiment_id':'teacher_indicf5_v14',
        'role':'automatic_reference_generalization_not_training', 'jobs': jobs, 'contexts': contexts},
        'inputs.json': {'actual_selection_sha256': sha(source/'selection.json'),
            'model_pair_sha256': sha(baseline/'model_pair.json'), 'conditioning_selection_sha256':sha(baseline/'selection.json'),
            'contexts':len(contexts),'jobs':len(jobs),'smoke_jobs':6,
            'policy':'Best24-layer step250000 only. Compare two reference voices before using generated reference consensus; no teacher reading assumed correct.'},
        'source_snapshot.json': {'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Actual Swar texts rendered by the requested24-layer teacher as two reference voices. This audits '
        'operation without a human recording for every new query. references/*.wav are two exact copied '
        'conditioning prompts from the already verified native audit; reference_inventory.json records '
        'source/bytes/SHA. teacher/ stores generated per-wave receipts and runtime, mirrored from L40. '
        'No new checkpoint copy. Teacher readings are not assumed pronunciation gold; source/content and '
        'phonetic suitability are checked. Current consumer generic reference-backed detector validation; '
        'review2026-11-09, retire redundant generated audio after consumer closes. No HF upload; inherited rights.',
        ['.venv-indicf5/bin/python -m scripts.build_swar_teacher_plan',
         'Explicit24-layer teacher source manifest, exact prior two prompt voices, actual86 nonnumeric speech texts'])
    (out/'references').mkdir()
    for name, path in sources.items():
        target = out/'references'/name
        shutil.copyfile(path, target)
        if sha(target) != sha(path):
            raise ValueError('Conditioning prompt copy mismatch')
        inventory.append({'path':str(target.relative_to(out)),'source':str(path),'bytes':target.stat().st_size,'sha256':sha(target)})
    (out/'reference_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({'contexts':len(contexts),'jobs':len(jobs),'smoke_jobs':6}))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Actual Swar teacher plan failed explicitly')
        raise
