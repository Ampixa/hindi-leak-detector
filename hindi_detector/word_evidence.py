"""Aligned phonetic posteriors and normalized prosody, without hidden embeddings."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from hindi_detector.local_features import PROSODY_NAMES, relative_tracks, word_prosody
from hindi_detector.g2p_evidence import compile_word_hypotheses, word_path_evidence
from hindi_detector.schwa_probe import compare_site, site_paths


def normalized_words(text: str, romanizer) -> list[dict]:
    words = []
    for match in re.finditer(r'\S+', text):
        raw = match.group()
        lexical = ''.join(c for c in raw if unicodedata.category(c)[0] in {'L', 'M', 'N'})
        if not lexical:
            continue
        if any(unicodedata.category(c)[0] == 'N' for c in lexical):
            raise ValueError('Numeric token needs an explicit spoken expansion; never silently drop it')
        roman = romanizer.romanize_string(lexical, lcode='nep').lower().replace('’', "'")
        if re.search(r"[^a-z' ]", roman):
            raise ValueError(f'Romanization has unsupported symbols: {raw!r} -> {roman!r}')
        roman = roman.replace(' ', '')
        if not roman:
            raise ValueError(f'Word vanished during romanization: {raw!r}')
        words.append({'word': lexical, 'display': raw, 'roman': roman,
                      'text_start': match.start(), 'text_end': match.end()})
    if not words:
        raise ValueError('No alignable words')
    return words


class WordEvidenceEngine:
    def __init__(self, root: Path, *, g2p_word_bank: list[dict] | None = None,
                 device: str = 'cpu', pitch_method: str = 'harvest', hidden_layers: tuple[int, ...] = (),
                 temporal_bins: int = 0, schwa_candidates: list[dict] | None = None,
                 lexical_ipa: dict[str, str] | None = None,
                 retain_frame_evidence: bool = False):
        import torch
        import torchaudio
        import pyworld
        import uroman
        from transformers import AutoFeatureExtractor, AutoModelForCTC

        self.torch, self.world = torch, pyworld
        if device not in {'cpu', 'mps'} or device == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('Requested word encoder device is unavailable/unsupported')
        if pitch_method not in {'harvest', 'dio'} or any(type(i) is not int or i < 1 for i in hidden_layers):
            raise ValueError('Need explicit supported pitch recipe and positive hidden-layer indices')
        self.device, self.pitch_method, self.hidden_layers = device, pitch_method, hidden_layers
        if temporal_bins not in {0, 20} or (schwa_candidates is None) != (lexical_ipa is None):
            raise ValueError('Need explicit supported temporal geometry and complete optional site inputs')
        self.temporal_bins = temporal_bins
        torch.set_num_threads(2)
        directory = root / 'models/mms_fa'
        inventory = json.loads((directory / 'inventory.json').read_text())
        path = directory / inventory['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != inventory['sha256']:
            raise ValueError('Alignment checkpoint identity changed')
        bundle = torchaudio.pipelines.MMS_FA
        if bundle._path != inventory['url'] or torchaudio.__version__ != inventory['torchaudio']:
            raise ValueError('Alignment bundle/runtime changed')
        self.alignment = bundle.get_model(dl_kwargs={'model_dir': str(directory), 'file_name': path.name, 'weights_only': True}).eval().to(device)
        self.tokenizer, self.aligner = bundle.get_tokenizer(), bundle.get_aligner()
        self.romanizer = uroman.Uroman()
        self.aligner_inventory = inventory
        probe = root / 'models/phoneme_probe'
        phone_inventory = json.loads((probe / 'inventory.json').read_text())
        for item in phone_inventory['files']:
            p = probe / item['path']
            if p.stat().st_size != item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Phone checkpoint inventory mismatch')
        self.extractor = AutoFeatureExtractor.from_pretrained(probe, local_files_only=True)
        self.phone = AutoModelForCTC.from_pretrained(probe, local_files_only=True).eval().to(device)
        if any(i > self.phone.config.num_hidden_layers for i in hidden_layers):
            raise ValueError('Requested hidden layer exceeds actual phone encoder depth')
        self.vocab = json.loads((probe / 'vocab.json').read_text())
        self.tokens = {i: token for token, i in self.vocab.items()}
        self.special = {self.vocab[token] for token in ['<pad>', '<s>', '</s>', '<unk>']}
        self.phone_inventory = phone_inventory
        self.retain_frame_evidence = retain_frame_evidence
        self.g2p_hypotheses = compile_word_hypotheses(g2p_word_bank, self.vocab, self.special) if g2p_word_bank is not None else {}
        self.site_hypotheses = {}
        for candidate in schwa_candidates or []:
            try:
                record = {'candidate_id': candidate['candidate_id'], 'status': 'supported',
                          'paths': site_paths(candidate, lexical_ipa, self.vocab)}
            except ValueError as error:
                record = {'candidate_id': candidate['candidate_id'], 'status': 'unsupported', 'reason': str(error)}
            self.site_hypotheses.setdefault(candidate['word'], []).append(record)
        self.other_vowels = sorted({self.vocab[token] for token in ('a','aː','e','eː','i','iː','o','oː','u','uː','ɪ','ʊ','ɛ','ɔ','æ') if token in self.vocab})

    def analyze(self, samples: np.ndarray, text: str) -> tuple[dict, dict[str, np.ndarray]]:
        torch = self.torch
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all() or not 3200 <= len(samples) <= 30 * 16000 or np.max(np.abs(samples)) < 1e-5:
            raise ValueError('Need finite non-silent 16 kHz mono speech between .2 and 30 seconds')
        words = normalized_words(text, self.romanizer)
        with torch.inference_mode():
            emission, _ = self.alignment(torch.from_numpy(samples)[None].to(self.device))
            log_alignment = emission[0].log_softmax(-1).cpu()
            if not torch.isfinite(log_alignment).all():
                raise ValueError('Nonfinite alignment model output')
            aligned = self.aligner(log_alignment, self.tokenizer([w['roman'] for w in words]))
            encoded = self.phone(**{k: v.to(self.device) for k, v in self.extractor(samples, sampling_rate=16000, return_tensors='pt').items()},
                                 output_hidden_states=bool(self.hidden_layers))
            phone_logits = encoded.logits[0]
            if not torch.isfinite(phone_logits).all():
                raise ValueError('Nonfinite phone model output')
        if len(aligned) != len(words) or any(not spans for spans in aligned):
            raise ValueError('Alignment did not retain every expected word')
        duration = len(samples) / 16000
        frame_seconds = duration / len(log_alignment)
        for word, spans in zip(words, aligned, strict=True):
            start, end = spans[0].start * frame_seconds, spans[-1].end * frame_seconds
            if not 0 <= start < end <= duration + 1e-6:
                raise ValueError('Impossible aligned word times')
            lengths = np.array([s.end - s.start for s in spans])
            score = float(np.average([s.score for s in spans], weights=lengths))
            word.update(start=float(start), end=float(end), alignment_score=score)
        # Fixed shared analysis rate and explicit broad adult-speech pitch range.
        pitch_function = self.world.harvest if self.pitch_method == 'harvest' else self.world.dio
        f0, times = pitch_function(samples.astype(np.float64), 16000, f0_floor=50., f0_ceil=650., frame_period=5.)
        f0 = self.world.stonemask(samples.astype(np.float64), f0, times, 16000)
        smooth_energy = np.sqrt(np.maximum(np.convolve(samples.astype(float) ** 2, np.ones(320) / 320, mode='same'), 0))
        rms = np.interp(times, np.arange(len(samples)) / 16000, smooth_energy)
        prosody = word_prosody(times, f0, rms, words)
        relative_pitch, relative_energy = relative_tracks(f0, rms)
        probabilities = torch.softmax(phone_logits, -1).cpu().numpy()
        log_probabilities = torch.log_softmax(phone_logits, -1).cpu().numpy() if self.g2p_hypotheses or self.site_hypotheses or self.retain_frame_evidence else None
        hidden = {i: encoded.hidden_states[i][0].cpu().numpy() for i in self.hidden_layers}
        if any(v.ndim != 2 or len(v) != len(probabilities) or not np.isfinite(v).all() for v in hidden.values()):
            raise ValueError('Hidden encoder geometry/values disagree with actual phone frames')
        hidden_words = {i: [] for i in hidden}
        phone_vectors = []; temporal_vectors = []
        for word, pvector in zip(words, prosody, strict=True):
            start = max(0, int(np.floor(word['start'] / duration * len(probabilities))))
            end = min(len(probabilities), max(start + 1, int(np.ceil(word['end'] / duration * len(probabilities)))))
            values = probabilities[start:end].copy()
            for layer, states in hidden.items():
                part = states[start:end]
                hidden_words[layer].append(np.r_[part.mean(0), part.std(0)])
            values[:, sorted(self.special)] = 0
            mass = values.sum(axis=1)
            if self.temporal_bins:
                central = probabilities[start:end][:, [self.vocab[t] for t in ('ʌ','ə','ɐ')]]
                channels = np.column_stack([central, central / np.maximum(mass[:,None],1e-8), mass,
                                           probabilities[start:end][:, self.other_vowels].sum(axis=1)])
                positions = np.linspace(0,len(channels)-1,self.temporal_bins)
                temporal_vectors.append(np.stack([np.interp(positions,np.arange(len(channels)),channels[:,i]) for i in range(channels.shape[1])],axis=1))
                word['temporal_phone_frames'] = len(channels)
            available = float(mass.sum()) > 1e-5
            if available:
                mean = values.sum(axis=0) / mass.sum()
                conditional = values / np.maximum(mass[:, None], 1e-20)
                spread = np.sqrt(((conditional - mean) ** 2 * mass[:, None]).sum(0) / mass.sum())
            else:
                mean, spread = np.zeros(len(self.vocab)), np.zeros(len(self.vocab))
            path = probabilities[start:end].argmax(axis=1)
            collapsed = [int(token) for i, token in enumerate(path) if (i == 0 or token != path[i - 1]) and token not in self.special]
            word.update(phone_evidence_available=available, phone_probe_sequence=' '.join(self.tokens[t] for t in collapsed),
                        central_vowel_posteriors={token: float(mean[self.vocab[token]]) for token in ['ə', 'ɐ', 'ʌ']},
                        prosody={name: float(value) for name, value in zip(PROSODY_NAMES, pvector, strict=True)})
            if word['word'] in self.g2p_hypotheses:
                left = max(0, int(np.floor(max(0., word['start'] - .03) / duration * len(probabilities))))
                right = min(len(probabilities), int(np.ceil(min(duration, word['end'] + .03) / duration * len(probabilities))))
                word['g2p_path_evidence'] = word_path_evidence(log_probabilities[left:right], self.g2p_hypotheses[word['word']])
                word['g2p_path_evidence']['boundary_context_seconds'] = .03
            if word['word'] in self.site_hypotheses:
                measured = []
                for candidate in self.site_hypotheses[word['word']]:
                    if candidate['status'] != 'supported':
                        measured.append({k:v for k,v in candidate.items() if k!='paths'}); continue
                    sites = []
                    for path in candidate['paths']:
                        windows = []
                        for padding in (0., .03, .06):
                            left=max(0,int(np.floor(max(0.,word['start']-padding)/duration*len(probabilities))))
                            right=min(len(probabilities),int(np.ceil(min(duration,word['end']+padding)/duration*len(probabilities))))
                            try:
                                evidence=compare_site(log_probabilities[left:right],path)
                                windows.append({'padding_seconds':padding,'frames':right-left,'status':'supported',
                                    'alternative_minus_lexicon_per_frame':evidence['marginal_alternative_minus_lexicon']/(right-left)})
                            except ValueError as error:
                                windows.append({'padding_seconds':padding,'status':'impossible','reason':str(error)})
                        sites.append({'gap_index':path['gap_index'],'lexicon_schwa_count':path['lexicon_schwa_count'],
                            'alternative_schwa_count':path['alternative_schwa_count'],
                            'skeleton_length':sum(p!='ax' for p in path['lexicon_project_phones']), 'windows':windows})
                    measured.append({'candidate_id':candidate['candidate_id'],'status':'site_paths_measured_not_error_labels','sites':sites})
                word['schwa_site_evidence']=measured
            phone_vectors.append(np.r_[mean, spread])
        phonetics = np.asarray(phone_vectors, dtype=np.float32)
        if not np.isfinite(phonetics).all():
            raise ValueError('Nonfinite phone features')
        result = {'status': 'aligned_word_evidence_not_validated_hindi_labels', 'text': text,
                  'seconds': duration, 'words': words,
                  'tracks': {'times': times.tolist(),
                             'pitch_relative_st': [float(v) if np.isfinite(v) else None for v in relative_pitch],
                             'energy_relative_db': relative_energy.tolist()},
                  'normalization': {'absolute_pitch_not_a_feature': True, 'absolute_loudness_not_a_feature': True,
                                    'global_tempo_not_a_relative_word_feature': True,
                                    'hidden_speech_embeddings_not_used': not bool(self.hidden_layers)},
                  'encoder_device': self.device, 'pitch_method': self.pitch_method, 'hidden_layers': list(self.hidden_layers),
                  'temporal_bins': self.temporal_bins,
                  'warnings': ['Alignment scores are not calibrated pronunciation correctness.',
                               'Phone probe is not validated as a Nepali schwa recognizer.',
                               'Central-vowel posterior differences are evidence to inspect, not automatic Hindi word labels.',
                               'Pitch-register normalization does not prove invariance to complete voice conversion.']}
        if self.retain_frame_evidence:
            result['phone_frame_seconds'] = duration / len(probabilities)
        return result, {'phonetics': phonetics, 'prosody': prosody,
                        'times': times.astype(np.float32), 'f0_hz_for_qa_only': f0.astype(np.float32),
                        **({'temporal_phone':np.asarray(temporal_vectors,dtype=np.float32)} if self.temporal_bins else {}),
                        **({'phone_log_probs':log_probabilities.astype(np.float32)} if self.retain_frame_evidence else {}),
                        **{f'hidden_word_{layer}': np.stack(values).astype(np.float16) for layer, values in hidden_words.items()}}
