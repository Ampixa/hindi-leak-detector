"""Verify orthographic minimal pairs, then freeze natural same-engine requests."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import unicodedata

from hindi_detector.artifacts import write_group
from hindi_detector.consonant_context import NonSyllabicTable, compile_context_word
from hindi_detector.corrected_synthesis import require_pair, sha
from hindi_detector.dual_g2p import DualG2P
from hindi_detector.phone_lattice import represent, units_for_project
from hindi_detector.pronunciation_sites import align_units

ROOT=Path(__file__).resolve().parents[1]


def rank(text):
    return hashlib.sha256(('minimal-vowel-v23:'+text).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'data/minimal_vowels_v23')
    parser.add_argument('--per-kind',type=int,default=128)
    parser.add_argument('--internal-only',action='store_true')
    parser.add_argument('--quality-ai-only',action='store_true')
    parser.add_argument('--exclude-plan',type=Path,action='append',default=[])
    parser.add_argument('--prefix',default='min23')
    parser.add_argument('--evaluation-only',action='store_true')
    parser.add_argument('--reference-offset',type=int,default=0)
    parser.add_argument('--quality-only',action='store_true',help='Evaluation-only larger quality audit; no deletion quota')
    parser.add_argument('--evaluation-carrier',help='Explicit fresh evaluation template containing exactly one {word}')
    parser.add_argument('--unknown-consonant-context', action='store_true', help='Fresh evaluation only; explicitly unknown Native consonants rather than phoneme substitutions')
    parser.add_argument('--allow-consonant-context', action='store_true', help='Explicit research corpus with supported and unknown nonsyllabic contexts; no consonant-identity claim')
    parser.add_argument('--deletion-quota', type=int, help='Explicit per-cause lexeme quota;0 disables this cause')
    parser.add_argument('--quality-quota', type=int, help='Explicit per-cause lexeme quota;0 disables this cause')
    parser.add_argument('--voices-per-partition', type=int, default=2)
    parser.add_argument('--group-phone-collisions', action='store_true', help='Keep overlapping canonical/edited project-phone strings in one split; not a morphological-family claim')
    parser.add_argument('--training-pilot', action='store_true', help='Pre-fit yield pilot using training conditioning voices only; no reserved evaluation claim')
    parser.add_argument('--edited-test-only', action='store_true', help='Canonical requests in every fold; unspliced edited requests only in the reserved test')
    args=parser.parse_args();out=args.out;quota=args.per_kind
    if not 8<=quota<=512 or not args.prefix.isascii() or not args.prefix.isalnum():raise ValueError('Unsafe size/prefix')
    quotas={'vowel_deletion':args.deletion_quota if args.deletion_quota is not None else quota,
            'vowel_quality':args.quality_quota if args.quality_quota is not None else quota}
    if any(n!=0 and not 8<=n<=512 for n in quotas.values()) or not any(quotas.values()):raise ValueError('Explicit cause quotas require0 or8..512 lexemes and at least one enabled cause')
    if not 2<=args.voices_per_partition<=4:raise ValueError('Verified source has2..4 independent conditioning voices per partition')
    if args.training_pilot and args.evaluation_only:raise ValueError('Training pilot and reserved evaluation are mutually exclusive')
    if args.reference_offset<0 or args.reference_offset and not args.evaluation_only:
        raise ValueError('Different reference offset is restricted to the explicit fresh evaluation plan')
    if (args.quality_only or args.evaluation_carrier) and not args.evaluation_only:
        raise ValueError('Quality-only/custom-carrier plans are restricted to frozen-model evaluation')
    if args.unknown_consonant_context and not args.evaluation_only:
        raise ValueError('New unknown-consonant words are restricted to held-out evaluation, never implicit fitting')
    if args.evaluation_carrier and (args.evaluation_carrier.count('{word}')!=1 or any(c in args.evaluation_carrier.replace('{word}','') for c in '{}')):
        raise ValueError('Evaluation carrier needs exactly one word placeholder and no other format fields')
    if out.exists():raise FileExistsError(out)
    frontend=DualG2P(ROOT)
    ak=importlib.import_module('nepali_frontend.g2p.akshara')
    vocabulary=json.loads((ROOT/'models/phoneme_probe/vocab.json').read_text())
    table=NonSyllabicTable(ROOT/'data/panphon_context_v58') if args.unknown_consonant_context or args.allow_consonant_context else None
    def project_units(phones):
        if table is None:return units_for_project(phones,frontend.ipa,vocabulary)
        units=[];projections=[]
        for phone in phones:
            if phone=='.':continue
            if phone not in frontend.ipa or not frontend.ipa[phone]:raise ValueError('Missing project-phone IPA for '+repr(phone))
            unit,_,projection=table.representation(frontend.ipa[phone],vocabulary)
            units.append(unit);projections.append(projection)
        return units,projections
    swar_words=json.loads((ROOT/'data/swar_phone_plan_v18/words.json').read_text())
    excluded={r['word'] for r in swar_words}
    excluded_phones={tuple(p for p in phones if p!='.') for r in swar_words for phones in r['nepali_project_variants'] if phones}
    cases=json.loads((ROOT/'reports/task_vowel_release_v31/actual_swar/cases.json').read_text())
    excluded.update(word['word'] for case in cases for word in case['analysis'].get('real',{}).get('words',[]))
    for path in args.exclude_plan:
        content=json.loads(path.read_text())
        prior=content['words'] if isinstance(content,dict) else content
        excluded.update(r['word'] for r in prior)
        excluded.update(r['edited_word'] for r in prior if 'edited_word' in r)
        excluded_phones.update(tuple(r[key]) for r in prior for key in ('native_project_phones','edited_project_phones') if key in r)
    lexicon=frontend.ph.data.lexicon_variants()
    pool=sorted([w for w in lexicon if w not in excluded and 2<=len(w)<=16 and
        all('\u0900'<=c<='\u097f' and unicodedata.category(c)[0] in {'L','M'} for c in w)],key=rank)
    selected=[];counts=Counter();exclusions=Counter();seen=set()
    if args.quality_only:quotas['vowel_deletion']=0
    required=[kind for kind in ('vowel_deletion','vowel_quality') if quotas[kind]]
    for word in pool:
        if all(counts[k]>=quotas[k] for k in required):break
        result=frontend.ph.phonemize_word(word);target=[p for p in result.phones if p!='.']
        if tuple(target) in excluded_phones:
            exclusions['prior_or_swar_canonical_phone_collision']+=1;continue
        try:target_units,target_projection=project_units(target)
        except ValueError:
            exclusions['unsupported_native_acoustic_units']+=1;continue
        if args.unknown_consonant_context and not any(p['representation_mode']=='unknown_non_syllabic_context' for p in target_projection):
            exclusions['no_unknown_native_consonant_context']+=1;continue
        edits=[]
        if counts['vowel_quality']<quotas['vowel_quality']:
            for i,char in enumerate(word):
                if char not in {'ै','ौ'}:continue
                if args.quality_ai_only and char!='ै':continue
                edited=word[:i]+('े' if char=='ै' else 'ो')+word[i+1:]
                alternative=[p for p in frontend.ph.phonemize_word(edited).phones if p!='.']
                differences=[j for j,(a,b) in enumerate(zip(target,alternative)) if a!=b]
                if len(target)==len(alternative) and len(differences)==1:
                    j=differences[0]
                    if (target[j],alternative[j]) in {('axj','e'),('axw','o'),('aaj','e'),('aaw','o')}:
                        edits.append(('vowel_quality',edited,alternative,j))
        if counts['vowel_deletion']<quotas['vowel_deletion']:
            for syllable in ak.parse(word):
                if syllable.type!='syllable' or not syllable.components.get('inherent_vowel') or syllable.components.get('modifiers'):
                    continue
                point=syllable.span[1]
                edited=word[:point]+'्'+word[point:]
                alternative=[p for p in frontend.ph.phonemize_word(edited).phones if p!='.']
                indices=[i for i,p in enumerate(target) if p=='ax' and target[:i]+target[i+1:]==alternative]
                if len(indices)==1 and (not args.internal_only or 0<indices[0]<len(target)-1):
                    edits.append(('vowel_deletion',edited,alternative,indices[0]))
        if not edits:exclusions['no_verified_single_site_edit']+=1;continue
        row=frontend.word(word)
        try:
            if table is not None:
                context=compile_context_word(row,frontend.ipa,vocabulary,table)
                matching=[v for v in context['variants'] if v['phones']==target]
                if not matching:raise ValueError('Actual default Native target not represented by lexical variants')
                predicted=[s for variant in matching for s in variant['contrasts']]
                hindi=None
            else:hindi=[represent(t,vocabulary) for t in row['hindi_units']]
        except ValueError:
            exclusions['unsupported_hindi_acoustic_units']+=1;continue
        alignment=align_units(target_units,hindi) if hindi is not None else None
        accepted=[]
        for kind,edited,alternative,index in edits:
            if edited in excluded or tuple(alternative) in excluded_phones:
                exclusions['prior_or_swar_edited_collision']+=1;continue
            try:alternative_units,_=project_units(alternative)
            except ValueError:
                exclusions['unsupported_edited_acoustic_units']+=1;continue
            if table is not None:
                compatible=any(s['kind']==kind and s['index']==index and
                    (kind=='vowel_deletion' or s['alternative_unit']&alternative_units[index]) for s in predicted)
            elif kind=='vowel_deletion':
                compatible=(index,None) in alignment
            else:
                match=[j for i,j in alignment if i==index and j is not None]
                compatible=bool(match and hindi[match[0]]&alternative_units[index] and not target_units[index]&hindi[match[0]])
            if compatible:accepted.append((kind,edited,alternative,index))
        if not accepted:exclusions['edit_not_predicted_by_hindi_prior']+=1;continue
        # Balance causes before choosing any generated audio or outcome.
        accepted.sort(key=lambda e:(counts[e[0]],rank(e[1])))
        kind,edited,alternative,index=accepted[0]
        if word in seen:raise ValueError('Duplicate minimal-pair lexeme')
        ordinal=counts[kind];cause_quota=quotas[kind];part='train' if ordinal<int(cause_quota*.625) else 'development' if ordinal<int(cause_quota*.8125) else 'test'
        if args.evaluation_only:part='test'
        if args.training_pilot:part='train'
        selected.append({'word_id':rank(word)[:20],'word':word,'edited_word':edited,'kind':kind,
            'site_index':index,'native_project_phones':target,'edited_project_phones':alternative,
            'hindi_units':row['hindi_units'],'partition':part,'requested_labels_not_realization_gold':True,
            'nepali_g2p_source':result.source})
        counts[kind]+=1;seen.add(word)
    if {k:v for k,v in counts.items() if v}!={k:quotas[k] for k in required}:
        raise ValueError('Could not verify balanced word plan: '+str(dict(counts)))
    if args.group_phone_collisions:
        # Link exact canonical/edited phonetic collisions before assigning a
        # split. This prevents an edit target from reappearing as a canonical
        # pronunciation in another fold. It does not infer morphological roots.
        parents=list(range(len(selected)))
        def find(i):
            while parents[i]!=i:
                parents[i]=parents[parents[i]];i=parents[i]
            return i
        phone_owner={}
        for i,entry in enumerate(selected):
            for phones in (entry['native_project_phones'],entry['edited_project_phones']):
                key=tuple(phones)
                if key in phone_owner:parents[find(i)]=find(phone_owner[key])
                else:phone_owner[key]=i
        clusters={}
        for i,entry in enumerate(selected):clusters.setdefault(find(i),[]).append(entry)
        for entries in clusters.values():
            identity=hashlib.sha256(('phone-collision-v69:'+':'.join(sorted(e['word_id'] for e in entries))).encode()).hexdigest()
            fraction=int(identity[:12],16)/16**12
            part='train' if fraction<.625 else 'development' if fraction<.8125 else 'test'
            for entry in entries:
                entry['phone_collision_group']=identity
                entry['partition']='test' if args.evaluation_only else 'train' if args.training_pilot else part
        for kind in required:
            if not args.evaluation_only and not args.training_pilot and any(sum(e['kind']==kind and e['partition']==part for e in selected)<8 for part in ('train','development','test')):
                raise ValueError('Frozen phone-collision grouping leaves insufficient lexemes in one cause/fold')
    base=ROOT/'data/teacher_indicf5_v14';pair=json.loads((base/'model_pair.json').read_text());require_pair(pair)
    old=json.loads((base/'selection.json').read_text());by_part={}
    for part,oldpart in [('train','train'),('development','development'),('test','reserved_evaluation')]:
        refs={j['speaker_id']:j for j in old['jobs'] if j['split']==oldpart}
        if len(refs)<args.voices_per_partition+args.reference_offset:raise ValueError('Missing independent conditioning groups')
        by_part[part]=[refs[k] for k in sorted(refs)[args.reference_offset:args.reference_offset+args.voices_per_partition]]
    groups=[{r['speaker_id'] for r in v} for v in by_part.values()]
    if any(a&b for i,a in enumerate(groups) for b in groups[i+1:]):raise ValueError('Conditioning group split overlap')
    carriers={'train':'यो शब्द {word} हो।','development':'मैले {word} भन्ने शब्द सुनेँ।','test':'उहाँले {word} भनेर भन्नुभयो।'}
    if args.evaluation_only:carriers['test']='यस पटक {word} भन्नुहोस्।'
    if args.evaluation_carrier:carriers['test']=args.evaluation_carrier
    jobs=[];sources={}
    for entry in selected:
        part=entry['partition']
        for voice,reference in enumerate(by_part[part]):
            path=base/reference['reference_audio']
            if sha(path)!=reference['reference_sha256']:raise ValueError('Frozen conditioning prompt changed')
            sources[path.name]=path
            for label in (0,1):
                if args.edited_test_only and label and part!='test':continue
                text=carriers[part].format(word=entry['edited_word'] if label else entry['word'])
                jobs.append({'id':args.prefix+'_'+entry['word_id']+'_v'+str(voice)+'_c'+str(label),
                    'text_id':args.prefix+'_'+entry['word_id'],'text':text,'canonical_text':carriers[part].format(word=entry['word']),
                    'text_kind':'same_teacher_minimal_vowel_request','target_words':[entry['word']],
                    'speaker_id':reference['speaker_id'],'voice_ordinal':voice,'split':part,
                    'reference_audio':'references/'+path.name,'reference_sha256':reference['reference_sha256'],
                    'source_reference_sha256':reference['source_reference_sha256'],'reference_text':reference['reference_text'],
                    'seed':0,'automatic_leakage_label':None,'requested_vowel_change_class':label,'word_id':entry['word_id']})
    jobs.sort(key=lambda j:(0 if j['split']=='train' else 1 if j['split']=='development' else 2,j['word_id'],j['voice_ordinal'],j['requested_vowel_change_class']))
    write_group(out,{'model_pair.json':pair,'selection.json':{'experiment_id':'teacher_indicf5_v14','role':'same_engine_phonetic_requests',
        'jobs':jobs,'words':selected,'carriers':carriers},'inputs.json':{'words':len(selected),'clips':len(jobs),
        'kind_words':dict(counts),'partition_words':dict(Counter(e['partition'] for e in selected)),
        'explicit_cause_quotas':quotas,'voices_per_partition':args.voices_per_partition,
        'group_phone_collisions':args.group_phone_collisions,'allow_consonant_context':args.allow_consonant_context,
        'training_pilot':args.training_pilot,
        'edited_test_only':args.edited_test_only,
        'internal_deletions_only':args.internal_only,'quality_ai_only':args.quality_ai_only,
        'evaluation_only':args.evaluation_only,'conditioning_group_offset':args.reference_offset,
        'quality_only':args.quality_only,'evaluation_carrier_override':args.evaluation_carrier,
            'unknown_consonant_context':args.unknown_consonant_context,
            'context_resource_inventory_sha256':sha(ROOT/'data/panphon_context_v58/inventory.json') if table is not None else None,
        'additional_excluded_plans':[{'path':str(p),'sha256':sha(p)} for p in args.exclude_plan],
        'excluded_canonical_and_edited_phone_strings':len(excluded_phones),
        'excluded_original_swar_word_plan_sha256':sha(ROOT/'data/swar_phone_plan_v18/words.json'),
        'phonetic_validation':'Edited G2P differs only at one declared vowel site; Hindi prior predicts that contrast',
        'word_selection_exclusions':dict(exclusions),'native_source_model_sha256':pair['systems']['pocket_teacher_250000_24l']['sha256'],
        'labels':'Requested canonical/edited pronunciation, never generator identity or automatically asserted realization gold'},
        'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Natural same-teacher minimal vowel requests: see inputs.json for actual word/clip counts, two voices per partition, '
        'canonical/edited readings from the same24-layer checkpoint/conditioning/seed. No splice artifacts. '
        'Train/dev/test words, conditioning groups and carrier templates are disjoint; all actual Swar '
        'vocabulary excluded. Requested phonetic changes are verified through actual frontend/Hindi outputs '
        'but realized audio remains unreviewed until acoustic audit. references/*.wav are exact six copied '
        'prompts, reference_inventory.json pins sources/SHA. teacher/ later stores the planned generated outputs. '
        'Current consumer improved vowel-head training and fresh request-class evaluation; review2026-11-09, '
        'delete redundant generated copies after consumer closes. No HF upload; source licences apply.',
        ['.venv-indicf5/bin/python -m scripts.build_minimal_vowel_plan'])
    (out/'references').mkdir();inventory=[]
    for name,path in sources.items():
        target=out/'references'/name;shutil.copyfile(path,target)
        inventory.append({'path':str(target.relative_to(out)),'source':str(path),'bytes':target.stat().st_size,'sha256':sha(target)})
    (out/'reference_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({'words':len(selected),'jobs':len(jobs),'kinds':dict(counts),'partitions':dict(Counter(e['partition'] for e in selected))}))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Verified minimal-pair plan creation failed explicitly');raise
