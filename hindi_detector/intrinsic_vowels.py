"""Reference-free phonetic evidence for one declared vowel counterfactual.

Nuisance reductions outside the focal site are shared by both hypotheses.
These are acoustic-model representations, never native-acceptability gold.
"""
from __future__ import annotations

import math

import numpy as np

from hindi_detector.ctc import log_likelihood
from hindi_detector.phone_lattice import path_score

FEATURE_NAMES = [
    'margin_mean','margin_min','margin_max','margin_std','positive_margin_fraction',
    'native_fit_mean','alternative_fit_mean','best_fit_minus_free_decode',
    'native_site_log_emission','alternative_site_log_emission','native_site_conditional_mass',
    'alternative_site_conditional_mass','native_site_frame_fraction',
    'left_anchor_log_emission','right_anchor_log_emission','left_anchor_available','right_anchor_available',
    'gap_vowel_mass','gap_vowel_peak','gap_word_fraction','gap_available',
    'log_word_frames','native_unit_count','site_position','native_vowel_fraction','background_reductions',
    'site_initial','site_final','is_vowel_deletion','is_vowel_quality',
    'native_path_possible_fraction','alternative_path_possible_fraction']
FLOOR = -30.


def make_paths(compiled, variant, contrast, retained):
    at=retained.index(contrast['index'])
    target=[variant['units'][i] for i in retained]
    if contrast['kind']=='vowel_deletion':
        target[at]=compiled['any_vowel'];alternative=target[:at]+target[at+1:]
    else:
        alternative=target.copy();alternative[at]=contrast['alternative_unit']
    if not target or not alternative:raise ValueError('A vowel contrast requires two nonempty phonetic hypotheses')
    return target,alternative,at


def choose_context(compiled,variant,contrast,crop):
    retained=list(range(len(variant['units'])))
    candidates={i for i,p in enumerate(variant['phones']) if i!=contrast['index'] and p in {'ax','h'}}
    def fit(indices):
        a,b,_=make_paths(compiled,variant,contrast,indices)
        return max(path_score(crop,a),path_score(crop,b))
    current=fit(retained)
    while candidates:
        trials=[]
        for index in sorted(candidates):
            proposal=[i for i in retained if i!=index]
            try:value=fit(proposal)
            except ValueError:continue
            trials.append((value,index,proposal))
        if not trials:break
        value,index,proposal=max(trials,key=lambda r:r[0])
        if value<=current+.01:break
        retained=proposal;current=value;candidates.remove(index)
    return retained


def conditional_mass(values, unit, span):
    block=values[span['start_frame']:span['end_frame']]
    if not len(block):return 0.
    p=np.exp(block);mass=p[:,4:].sum()
    return float(p[:,sorted(unit)].sum()/max(1e-12,mass))


def measure(compiled,variant,contrast,crops):
    if len(crops)!=3:raise ValueError('Exactly three fixed boundary measurements required')
    retained=choose_context(compiled,variant,contrast,crops[1][2])
    native,alternate,at=make_paths(compiled,variant,contrast,retained)
    measurements=[]
    for pad,start,values in crops:
        a,b=path_score(values,native),path_score(values,alternate)
        possible_a,possible_b=math.isfinite(a),math.isfinite(b)
        if not possible_a:a=FLOOR*len(values)
        if not possible_b:b=FLOOR*len(values)
        measurements.append({'padding_seconds':pad,'frames':len(values),'start_frame':start,
            'native_fit':a/len(values),'alternative_fit':b/len(values),'margin':(b-a)/len(values),
            'native_possible':possible_a,'alternative_possible':possible_b})
    if not any(r['native_possible'] or r['alternative_possible'] for r in measurements):
        raise ValueError('Neither declared pronunciation can fit the measured word')
    middle=crops[1][2];n=len(middle)
    def trace(units):
        try:return path_score(middle,units,trace=True)
        except ValueError:return None
    native_trace=trace(native);alt_trace=trace(alternate)
    nspan=native_trace['spans'][at] if native_trace else None
    aspan=alt_trace['spans'][at] if alt_trace and contrast['kind']=='vowel_quality' else None
    ns=float(nspan['mean_log_emission']) if nspan else FLOOR
    als=float(aspan['mean_log_emission']) if aspan else 0.
    nm=conditional_mass(middle,native[at],nspan) if nspan else 0.
    am=conditional_mass(middle,alternate[at],aspan) if aspan else 0.
    left_score=right_score=FLOOR;left_available=right_available=0.
    start=end=0;gap_available=0.
    if contrast['kind']=='vowel_quality' and aspan:
        start,end=aspan['start_frame'],aspan['end_frame'];gap_available=1.
    elif alt_trace:
        left_index=at-1;right_index=at
        core_start=max(0,crops[0][1]-crops[1][1]);core_end=min(n,core_start+len(crops[0][2]))
        start,end=core_start,core_end
        if left_index>=0 and not alternate[left_index]<=compiled['any_vowel']:
            left=alt_trace['spans'][left_index];left_score=float(left['mean_log_emission'])
            left_available=1.;start=left['end_frame']
        if right_index<len(alternate) and not alternate[right_index]<=compiled['any_vowel']:
            right=alt_trace['spans'][right_index];right_score=float(right['mean_log_emission'])
            right_available=1.;end=right['start_frame']
        gap_available=float(bool(left_available or right_available))
    gap_mass=gap_peak=0.
    if gap_available and end>start:
        block=np.exp(middle[max(0,start):min(n,end)])
        if len(block):
            vp=block[:,sorted(compiled['any_vowel'])].sum(1);nonblank=block[:,4:].sum(1)
            gap_mass=float(vp.sum()/max(1e-12,nonblank.sum()));gap_peak=float(vp.max())
    ids=middle.argmax(1)
    free=[int(t) for i,t in enumerate(ids) if t>3 and (i==0 or t!=ids[i-1])]
    free_score=log_likelihood(middle,free)
    m=np.array([x['margin'] for x in measurements]);nf=np.array([x['native_fit'] for x in measurements]);af=np.array([x['alternative_fit'] for x in measurements])
    native_vowels=sum(u<=compiled['any_vowel'] for u in native)
    values=[m.mean(),m.min(),m.max(),m.std(),np.mean(m>0),nf.mean(),af.mean(),
        max(measurements[1]['native_fit'],measurements[1]['alternative_fit'])-free_score/n,
        ns,als,nm,am,(nspan['end_frame']-nspan['start_frame'])/n if nspan else 0.,
        left_score,right_score,left_available,right_available,gap_mass,gap_peak,max(0,end-start)/n,gap_available,
        np.log1p(len(crops[0][2])),len(native),at/max(1,len(native)-1),native_vowels/len(native),
        len(variant['units'])-len(retained),float(at==0),float(at==len(native)-1),
        float(contrast['kind']=='vowel_deletion'),float(contrast['kind']=='vowel_quality'),
        np.mean([x['native_possible'] for x in measurements]),np.mean([x['alternative_possible'] for x in measurements])]
    features=np.asarray(values,np.float32)
    if len(features)!=len(FEATURE_NAMES) or not np.isfinite(features).all():raise ValueError('Invalid intrinsic phonetic evidence')
    detail={'windows':measurements,'retained_native_indices':retained,
        'native_site_span':nspan,'alternative_site_span':aspan,
        'gap':{'start_frame':crops[1][1]+start,'end_frame':crops[1][1]+end,'available':bool(gap_available)},
        'native_target_phones':variant['phones'],'free_decode_ids':free,
        'impossible_path_censor_floor_per_frame':FLOOR,'no_reference_audio_used':True}
    return features,detail
