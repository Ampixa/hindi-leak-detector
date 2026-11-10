"""Pinned local two-phone-model runtime; no hidden downloads or substitutions."""
from __future__ import annotations

import json

import numpy as np
import torch
from transformers import AutoFeatureExtractor,AutoModelForCTC

from hindi_detector.corrected_synthesis import sha


class PhonePair:
    def __init__(self,root,state,device='mps',primary_engine=None):
        if device not in {'cpu','mps'} or device=='mps' and not torch.backends.mps.is_available():raise ValueError('Unavailable phone device')
        self.device=device;self.models=[];self.vocabularies=[]
        inventories=[state['phone_inventory'],state['secondary_phone_inventory']]
        for index,directory in enumerate([root/'models/phoneme_probe',root/'models/phoneme_probe_lv60_v20']):
            actual=json.loads((directory/'inventory.json').read_text())
            if actual!=inventories[index]:raise ValueError('Requested checkpoint inventory differs')
            for member in actual['files']:
                path=directory/member['path']
                if path.stat().st_size!=member['bytes'] or sha(path)!=member['sha256']:raise ValueError('Phone checkpoint bytes changed')
            vocabulary=json.loads((directory/'vocab.json').read_text());self.vocabularies.append(vocabulary)
            if {k:vocabulary[k] for k in ('<pad>','<s>','</s>','<unk>')}!={'<pad>':0,'<s>':1,'</s>':2,'<unk>':3}:raise ValueError('Unexpected phone special IDs')
            if index==0 and primary_engine is not None:
                if primary_engine.phone_inventory!=actual or primary_engine.device!=device:raise ValueError('Shared primary runtime differs')
                extractor,model=primary_engine.extractor,primary_engine.phone
            else:
                extractor=AutoFeatureExtractor.from_pretrained(directory,local_files_only=True)
                model=AutoModelForCTC.from_pretrained(directory,local_files_only=True).eval().to(device)
            if extractor.sampling_rate!=16000 or model.config.pad_token_id!=0:raise ValueError('Unexpected phone input/blank')
            self.models.append((extractor,model))
        torch.set_num_threads(2)

    def encode_one(self,samples,index):
        samples=np.asarray(samples,np.float32)
        if samples.ndim!=1 or not 3200<=len(samples)<=30*16000 or not np.isfinite(samples).all():raise ValueError('Need finite16k mono speech between0.2 and30sec')
        if self.device=='mps' and len(samples)>20*16000:raise ValueError('Long utterance requires whole-group CPU execution')
        extractor,model=self.models[index]
        with torch.inference_mode():
            output=model(**{k:v.to(self.device) for k,v in extractor(samples,sampling_rate=16000,return_tensors='pt').items()})
            lp=output.logits[0].log_softmax(-1).cpu().numpy()
        if lp.ndim!=2 or lp.shape[1]!=len(self.vocabularies[index]) or not np.isfinite(lp).all():raise ValueError('Invalid measured phone posterior')
        return lp

    def encode(self,samples):return [self.encode_one(samples,index) for index in (0,1)]
