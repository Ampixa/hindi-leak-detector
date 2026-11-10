"""Reuse verified human contexts with the requested teacher and shared prompts."""
from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
import shutil

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import require_pair,sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'data/native_variation_v16'
    if out.exists():raise FileExistsError(out)
    base=ROOT/'data/teacher_indicf5_v14';pair=json.loads((base/'model_pair.json').read_text());require_pair(pair)
    old_path=ROOT/'data/human_matched_v9/selection.json';old=json.loads(old_path.read_text())
    corrected=json.loads((base/'selection.json').read_text())
    refs={j['source_reference_sha256']:j for j in corrected['jobs']}
    training_texts={j['text'] for j in corrected['jobs']};contexts={r['id']:r for r in old['contexts']}
    if any(c['text'] in training_texts for c in contexts.values()):raise ValueError('Native audit text overlaps corrected training corpus')
    jobs=[];sources={};voices=set()
    for oldjob in old['jobs']:
        context=contexts[oldjob['source_context_id']];path=Path(context['native_audio']).resolve(strict=True)
        if sha(path)!=context['native_sha256'] or context['native_speaker_id']==oldjob['speaker_id']:
            raise ValueError('Native source identity/reference group mismatch')
        reference=refs[oldjob['reference_sha256']]
        source=base/reference['reference_audio']
        if sha(source)!=reference['reference_sha256']:raise ValueError('Corrected shared prompt changed')
        sources[source.name]=source;voices.add(reference['speaker_id'])
        jobs.append({'id':'native16_'+oldjob['id'],'text_id':context['id'],'text':context['text'],'text_kind':'external_human_context',
            'target_words':context['target_words'],'speaker_id':reference['speaker_id'],'split':'external_native_variation',
            'reference_audio':'references/'+source.name,'reference_sha256':reference['reference_sha256'],
            'source_reference_sha256':reference['source_reference_sha256'],'reference_text':reference['reference_text'],'seed':0,
            'automatic_leakage_label':None,'native_clip_id':context['native_clip_id']})
    if len(jobs)!=126 or len(contexts)!=63 or len(voices)!=2:raise ValueError('Inspected native audit dimensions differ')
    report={'contexts':len(contexts),'teacher_jobs':len(jobs),'focal_words':len(old['words']),
        'human_speaker_groups':len({c['native_speaker_id'] for c in contexts.values()}),
        'regions':dict(Counter(c['reported_state'] for c in contexts.values())),
        'actual_hindi_error_labels':None,'training_use':False,
        'limits':['Separate actual human recordings from verified source/staging corpus; accents and regional variants are not error labels.',
                  'Text-disjoint from corrected synthesis corpus; focal lexemes can overlap training and are reported separately.',
                  'Corpus metadata reports Indian regions, not verified Hindi exposure or residence in Nepal.',
                  'Selection was explored previously; this is a native-variation audit, not untouched pronunciation-error ground truth.']}
    write_group(out,{'model_pair.json':pair,'selection.json':{'experiment_id':'teacher_indicf5_v14','audit_role':'external_native_variation',
        'jobs':jobs,'contexts':list(contexts.values()),'words':old['words']},'report.json':report,
        'inputs.json':{'human_context_selection_sha256':sha(old_path),'corrected_model_pair_sha256':sha(base/'model_pair.json')},
        'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Recorded-speaker variation audit, not training. references/*.wav are2 verified shared <=5s prompts copied from current corrected plan; reference_inventory.json pins copy bytes/SHA/source. selection.json records63 original human contexts and126 same-text teacher requests. Original human audio stays at its verified source path. Generated teacher/ subgroup has its own per-WAV receipts. Current consumer audit of frozen final corrected candidate on real human variation; no native/hindi-error labels fabricated. Source licences apply; no HF upload. Review2026-11-09; retire extra reference copies/audio after audit consumer closes.',
        ['.venv-indicf5/bin/python -m scripts.build_native_variation_plan','Existing verified human_matched_v9 source contexts; current explicit24-layer teacher only'])
    (out/'references').mkdir();inventory=[]
    for name,path in sources.items():
        target=out/'references'/name;shutil.copyfile(path,target)
        if sha(target)!=sha(path):raise ValueError('Prompt copy mismatch')
        inventory.append({'path':str(target.relative_to(out)),'source':str(path),'bytes':target.stat().st_size,'sha256':sha(target)})
    (out/'reference_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Native-variation plan failed explicitly');raise
