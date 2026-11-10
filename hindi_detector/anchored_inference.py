"""Persistent post-utterance local-deletion runtime for the research bench."""
from __future__ import annotations

import time

import numpy as np

from hindi_detector.anchored_detector import load, score
from hindi_detector.anchored_vowels import measure
from hindi_detector.aspirate_context import AspirateContextTable
from hindi_detector.consonant_context import compile_context_word
from hindi_detector.dual_g2p import DualG2P
from hindi_detector.dual_runtime import PhonePair
from hindi_detector.pronunciation_sites import windows
from hindi_detector.word_evidence import WordEvidenceEngine


class AnchoredInference:
    def __init__(self, root, model, device='mps'):
        self.network, self.state = load(model)
        self.frontend = DualG2P(root)
        self.table = AspirateContextTable(root / 'data/panphon_context_v58')
        self.engine = WordEvidenceEngine(root, device=device, pitch_method='dio', retain_frame_evidence=True)
        if self.engine.aligner_inventory != self.state['aligner_inventory']:
            raise ValueError('Bench aligner differs from frozen training runtime')
        self.pair = PhonePair(root, self.state, device=device, primary_engine=self.engine)

    def analyze(self, samples, text):
        started = time.perf_counter()
        detail, arrays = self.engine.analyze(samples, text)
        aligned = time.perf_counter()
        log_probs = [arrays['phone_log_probs'], self.pair.encode_one(samples, 1)]
        encoded = time.perf_counter()
        words = []
        for ordinal, word in enumerate(detail['words']):
            record = {key: word[key] for key in ('word', 'start', 'end')}
            record.update(sites=[], unavailable_reasons=[])
            try:
                row = self.frontend.word(word['word'])
                compiled = [compile_context_word(row, self.frontend.ipa, vocabulary, self.table) for vocabulary in self.pair.vocabularies]
                default = [phone for phone in row['nepali_default']['phones'] if phone != '.']
                candidates = [variant for variant in compiled[0]['variants'] if variant['phones'] == default]
                if not candidates:
                    raise ValueError('Preferred spoken-Nepali variant not represented in the pinned lexical hypotheses')
                variant = candidates[0]
                for contrast in variant['contrasts']:
                    if contrast['kind'] != 'vowel_deletion':
                        record['unavailable_reasons'].append('Separate vowel-quality contrast; this head covers internal deletion only')
                        continue
                    try:
                        values, evidence = [], []
                        for probe in (0, 1):
                            current = next(v for v in compiled[probe]['variants'] if v['phones'] == variant['phones'])
                            site = next(c for c in current['contrasts'] if c['kind'] == contrast['kind'] and c['index'] == contrast['index'])
                            item = {'detail': {**detail, 'phone_frame_seconds': detail['seconds'] / len(log_probs[probe])}, 'phone_log_probs': log_probs[probe]}
                            vector, local = measure(compiled[probe], current, site, windows(item, ordinal))
                            values.extend(vector.tolist())
                            evidence.append(local)
                        probability = float(score(self.network, self.state, np.array(values)[None])[0])
                        record['sites'].append({'kind': 'vowel_deletion', 'native_phone_index': contrast['index'],
                            'expected': contrast['expected'], 'native_phones': variant['phones'],
                            'evidence_score': probability * 100, 'candidate': probability >= self.state['threshold'],
                            'features': values, 'evidence': evidence, 'actual_hindi_error_probability': None})
                    except StopIteration:
                        record['unavailable_reasons'].append('Unsupported local contrast: matching preferred variant/site absent from a probe vocabulary')
                    except ValueError as error:
                        record['unavailable_reasons'].append('Unsupported local contrast: ' + str(error))
            except ValueError as error:
                record['unavailable_reasons'].append(str(error))
            record['tested'] = bool(record['sites'])
            record['candidate'] = any(site['candidate'] for site in record['sites'])
            record['evidence_score'] = max((site['evidence_score'] for site in record['sites']), default=None)
            words.append(record)
        elapsed = time.perf_counter() - started
        tested = [word for word in words if word['tested']]
        return {'status': 'post_utterance_local_deletion_evidence', 'text': text, 'seconds': detail['seconds'], 'words': words,
            'score': max((word['evidence_score'] for word in tested), default=None),
            'score_definition': 'Strongest tested word-level physical-deletion evidence,0..100; uncalibrated for Hindi influence.',
            'coverage': {'tested_words': len(tested), 'total_words': len(words), 'candidate_words': sum(word['candidate'] for word in words)},
            'timing_seconds': {'primary_alignment_encoder_prosody': aligned - started, 'secondary_encoder': encoded - aligned,
                'g2p_local_measurement_head': time.perf_counter() - encoded, 'total': elapsed, 'real_time_factor': elapsed / detail['seconds']},
            'streaming': False, 'actual_hindi_error_probability': None,
            'limits': ['Internal-vowel deletion head only; no validated intonation or general accent score.',
                'Bidirectional encoders analyze the completed utterance; real-time factor is processing time/audio duration, not streaming latency.',
                'No tested contrast does not establish correctness; Native variation and G2P/transcript errors can cause candidates.']}
