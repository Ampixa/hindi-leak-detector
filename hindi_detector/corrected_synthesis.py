"""Strict adapters for the explicitly requested24-layer teacher/IndicF5 pair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

SYSTEMS = ('pocket_teacher_250000_24l', 'indicf5_stock')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def require_pair(pair: dict) -> None:
    if pair.get('schema') != 'teacher_indicf5_pair_v14' or set(pair.get('systems', {})) != set(SYSTEMS):
        raise ValueError('Only the user-requested teacher/IndicF5 pair is accepted; no legacy fallback')
    teacher, f5 = (pair['systems'][s] for s in SYSTEMS)
    if teacher['layers'] != 24 or teacher['step'] != 250000 or f5['repo'] != 'ai4bharat/IndicF5':
        raise ValueError('Wrong teacher architecture/step or IndicF5 source')
    if teacher['sha256'] != '2b4636e3aed0e4f2d035630e504be95fc58f494b690a851a91bfe0dd8299516f' or f5['sha256'] != 'ba7f3671180fb7784e24bd1dafc96e729a38ce02e7f6d3877cdef32525a1865c':
        raise ValueError('Unapproved checkpoint bytes')


class QualityError(ValueError):
    pass


def validate_wave(wave, rate):
    if rate != 24000 or wave.ndim != 1 or not np.isfinite(wave).all() or not .2 <= len(wave)/rate <= 30 or np.max(np.abs(wave)) < 1e-5:
        raise QualityError('Invalid generated waveform; no substitute or trimming')


class TeacherEngine:
    def __init__(self, spec):
        import torch
        import yaml
        self.torch = torch
        source=Path(spec['source']); config=Path(spec['config']); weights=Path(spec['weights']); tokenizer=Path(spec['tokenizer'])
        if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=spec['source_revision']:
            raise ValueError('Teacher source revision differs')
        subprocess.run(['git','-C',str(source),'diff','--quiet','HEAD','--','pocket_tts','training/modules','training/args.py'],check=True)
        if sha(weights)!=spec['sha256'] or sha(tokenizer)!=spec['tokenizer_sha256']:
            raise ValueError('Teacher weights/tokenizer identity differs')
        cfg=yaml.safe_load(config.read_text())
        if cfg['flow_lm']['transformer']['num_layers']!=24 or cfg['mimi']['sample_rate']!=24000:
            raise ValueError('Teacher config is not the inspected24-layer24k model')
        if not torch.cuda.is_available():
            raise RuntimeError('Explicit L40 CUDA teacher placement unavailable')
        free,total=torch.cuda.mem_get_info()
        if free<10*1024**3:
            raise RuntimeError('Insufficient free GPU memory; do not disrupt other workloads')
        sys.path.insert(0,str(source))
        from training.args import TrainArgs
        from training.modules.builders import build_models
        from pocket_tts.modules.stateful_module import init_states
        args=TrainArgs(model_config=str(config),model_overrides={'weights_path':str(weights),
            'flow_lm.lookup_table.tokenizer_path':str(tokenizer)},start_from_pretrained=True,
            reset_text_embedding=False,compile=False)
        torch.set_num_threads(4)
        model,mimi,resolved=build_models(args)
        if resolved.flow_lm.transformer.num_layers!=24 or len(model.flow_lm.transformer.layers)!=24:
            raise ValueError('Loaded teacher does not contain24 transformer layers')
        self.model=model.eval().to('cuda');self.mimi=mimi.eval().to('cuda')
        self.model.requires_grad_(False);self.mimi.requires_grad_(False)
        self.spec=spec;self.prompts={};self.init_states=init_states
        self.ratio=round(mimi.encoder_frame_rate/mimi.frame_rate)
        paths=[source/'training/modules/builders.py',source/'training/modules/model.py',source/'training/args.py',
            source/'pocket_tts/modules/stateful_module.py',config]
        self.runtime={'system':SYSTEMS[0],'weights_sha256':sha(weights),'tokenizer_sha256':sha(tokenizer),
            'loaded_layers':24,'settings':spec['inference'],'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'source_revision':spec['source_revision'],'inference_source':[{'path':str(p),'sha256':sha(p),'source_utf8':p.read_text()} for p in paths]}

    def generate(self, job):
        import soundfile as sf
        from scipy.signal import resample_poly
        torch=self.torch;key=job['reference_sha256'];settings=self.spec['inference']
        if key not in self.prompts:
            wave,rate=sf.read(job['reference_audio'],dtype='float32',always_2d=True)
            if wave.shape[1]!=1 or not np.isfinite(wave).all() or not 1<=len(wave)/rate<=5.01:
                raise ValueError('Teacher requires the verified shared <=5s mono prompt')
            divisor=int(np.gcd(rate,24000));wave=resample_poly(wave[:,0],24000//divisor,rate//divisor).astype('float32')
            with torch.inference_mode():
                self.prompts[key]=self.mimi.encode_to_latent(torch.from_numpy(wave.copy())[None,None].to('cuda'))[0]
        torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
        tokenize=self.model.flow_lm.conditioner.tokenizer.sp.encode
        tokens=torch.tensor(tokenize(job['text']),dtype=torch.long)
        with torch.inference_mode():
            latents=self.model.generate([tokens],[self.prompts[key]],max_frames=settings['max_frames'],
                temp=settings['temperature'],n_steps=settings['n_steps'],cfg_coef=settings['cfg_coef'],eos_threshold=settings['eos_threshold'])[0]
            if len(latents)<8 or len(latents)>=settings['max_frames']:
                raise QualityError('Teacher emitted too few frames or reached frame cap')
            states=self.init_states(self.mimi,1,(len(latents)+8)*self.ratio)
            wave=self.mimi.decode_from_latent(latents[None],states)[0,0].float().cpu().numpy()
        return wave,24000,{'frames':len(latents),'frame_cap':False}


class IndicF5Engine:
    def __init__(self,spec):
        import torch
        from safetensors.torch import load_file
        from vocos import Vocos
        self.torch=torch;self.spec=spec;root=Path(spec['model_root'])
        inventory=json.loads((root/'inventory.json').read_text())
        if inventory['repo']!=spec['repo'] or inventory['revision']!=spec['revision']:
            raise ValueError('IndicF5 bundle source differs')
        for row in inventory['files']:
            p=root/row['path']
            if p.stat().st_size!=row['size_bytes'] or sha(p)!=row['sha256']:
                raise ValueError('IndicF5 bundle file differs: '+row['path'])
        vocos_inventory=json.loads((root/'vocos/inventory.json').read_text())
        for row in vocos_inventory['files']:
            if sha(root/'vocos'/row['path'])!=row['sha256']:
                raise ValueError('Vocos recipe differs')
        if sha(root/'model.safetensors')!=spec['sha256'] or not torch.backends.mps.is_available():
            raise ValueError('Wrong IndicF5 checkpoint or explicit MPS unavailable')
        sys.path.insert(0,str(root))
        from f5_tts.model import CFM, DiT
        from f5_tts.model.utils import get_tokenizer, seed_everything
        from f5_tts.infer.utils_infer import infer_batch_process
        torch.set_num_threads(2);vocab,size=get_tokenizer(str(root/'checkpoints/vocab.txt'),'custom')
        model=CFM(transformer=DiT(dim=1024,depth=22,heads=16,ff_mult=2,text_dim=512,conv_layers=4,text_num_embeds=size,mel_dim=100),
            mel_spec_kwargs={'n_fft':1024,'hop_length':256,'win_length':1024,'n_mel_channels':100,'target_sample_rate':24000,'mel_spec_type':'vocos'},
            odeint_kwargs={'method':'euler'},vocab_char_map=vocab)
        vocoder=Vocos.from_hparams(str(root/'vocos/config.yaml'));state=load_file(str(root/'model.safetensors'),device='cpu')
        a,b='ema_model._orig_mod.','vocoder._orig_mod.'
        if any(not k.startswith((a,b)) for k in state):
            raise ValueError('Unexpected IndicF5 state prefixes')
        model.load_state_dict({k.removeprefix(a):v for k,v in state.items() if k.startswith(a)},strict=True)
        vocoder.load_state_dict({k.removeprefix(b):v for k,v in state.items() if k.startswith(b)},strict=True)
        self.model=model.eval().to('mps');self.vocoder=vocoder.eval().to('mps');self.seed=seed_everything;self.infer=infer_batch_process
        self.runtime={'system':SYSTEMS[1],'repo':spec['repo'],'revision':spec['revision'],'weights_sha256':sha(root/'model.safetensors'),
            'settings':spec['inference'],'torch':torch.__version__,'inventory_sha256':sha(root/'inventory.json'),
            'vocos_inventory_sha256':sha(root/'vocos/inventory.json'),'strict_state_loading':True}

    def generate(self,job):
        import soundfile as sf
        wave,rate=sf.read(job['reference_audio'],dtype='float32',always_2d=True)
        if wave.shape[1]!=1 or not np.isfinite(wave).all():
            raise ValueError('Invalid shared IndicF5 prompt')
        self.seed(job['seed']);settings=self.spec['inference']
        audio,rate,_=self.infer((self.torch.from_numpy(wave.T),rate),job['reference_text'],[job['text']],
            self.model,self.vocoder,mel_spec_type='vocos',target_rms=.1,cross_fade_duration=.15,
            nfe_step=settings['nfe_step'],cfg_strength=settings['cfg_strength'],sway_sampling_coef=settings['sway_sampling_coef'],
            speed=settings['speed'],fix_duration=None,device='mps')
        return audio,rate,{}
