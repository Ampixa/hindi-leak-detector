import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from hindi_detector.teacher_reference_cache import ReferenceUnavailableError, TEACHER_SHA, references_for_text, text_key, verified_paths


class ReferenceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = self.root / 'data/swar_teacher_v21'
        (self.plan / 'teacher/audio').mkdir(parents=True)
        self.audio = self.plan / 'teacher/audio/example.wav'
        self.audio.write_bytes(b'identity-test-only')
        spec = {'sha256': TEACHER_SHA, 'layers': 24, 'step': 250000,
                'tokenizer_sha256': 'test-tokenizer', 'source_revision': 'test-source',
                'inference': {'temperature': .3}}
        self.pair = {'schema': 'teacher_indicf5_pair_v14', 'systems': {
            'pocket_teacher_250000_24l': spec,
            'indicf5_stock': {'repo': 'ai4bharat/IndicF5', 'sha256': 'ba7f3671180fb7784e24bd1dafc96e729a38ce02e7f6d3877cdef32525a1865c'}}}
        self.runtime = {'system': 'pocket_teacher_250000_24l', 'loaded_layers': 24,
                        'weights_sha256': TEACHER_SHA, 'tokenizer_sha256': 'test-tokenizer',
                        'source_revision': 'test-source', 'settings': {'temperature': .3}}
        self.row = {'id': 'test', 'text': 'किनकि', 'status': 'ok_unreviewed',
                    'audio': 'audio/example.wav', 'sha256': hashlib.sha256(self.audio.read_bytes()).hexdigest()}
        self.job = {'id': 'test', 'text': 'किनकि'}
        self.save()

    def save(self):
        for name, data in [('model_pair.json', self.pair), ('teacher/runtime.json', self.runtime)]:
            (self.plan / name).write_text(json.dumps(data))
        (self.plan / 'teacher/items.jsonl').write_text(json.dumps(self.row) + '\n')

    def test_identity_requires_weights_tokenizer_source_and_settings(self):
        self.assertEqual(verified_paths(self.plan, [self.job]), [self.audio.resolve()])
        for key in ('weights_sha256', 'tokenizer_sha256', 'source_revision', 'settings'):
            old = self.runtime[key]
            self.runtime[key] = 'different'
            self.save()
            with self.assertRaisesRegex(ValueError, 'approved24-layer'):
                verified_paths(self.plan, [self.job])
            self.runtime[key] = old

    def test_changed_audio_or_different_text_is_rejected(self):
        self.audio.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            verified_paths(self.plan, [self.job])
        self.audio.write_bytes(b'identity-test-only')
        self.row['text'] = 'अर्को'
        self.save()
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            verified_paths(self.plan, [self.job])

    def test_receipt_cannot_escape_owned_teacher_cache(self):
        (self.plan / 'outside.wav').write_bytes(b'identity-test-only')
        self.row['audio'] = '../outside.wav'
        self.save()
        with self.assertRaisesRegex(ValueError, 'escaped'):
            verified_paths(self.plan, [self.job])

    def test_text_cache_key_normalizes_whitespace_and_unicode(self):
        self.assertEqual(text_key('  यस\n पटक  '), 'यस पटक')
        for text in (' ', 'अ' * 1001):
            with self.assertRaises(ValueError):
                text_key(text)

    def test_cached_quarantined_reference_never_reenters_automatic_inference(self):
        jobs = [self.job, {**self.job, 'id': 'other'}]
        (self.plan / 'selection.json').write_text(json.dumps({'jobs': jobs}))
        other = {**self.row, 'id': 'other'}
        (self.plan / 'teacher/items.jsonl').write_text('\n'.join(json.dumps(row) for row in [self.row, other]) + '\n')
        screening = self.root / 'reports/swar_teacher_content_v21'
        screening.mkdir(parents=True)
        rows = [{**self.row, 'content_quality': {'decision': 'quarantine'}},
                {**other, 'content_quality': {'decision': 'pass'}}]
        (screening / 'items.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
        paths, context = references_for_text(self.root, 'किनकि')
        self.assertEqual(paths, [self.audio.resolve()])
        self.assertEqual(context['reference_content_passes'], 1)
        self.assertEqual(context['excluded_reference_decisions'], [{'id': 'test', 'decision': 'quarantine'}])
        rows[1]['content_quality']['decision'] = 'quarantine'
        (screening / 'items.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
        with self.assertRaises(ReferenceUnavailableError):
            references_for_text(self.root, 'किनकि')


if __name__ == '__main__':
    unittest.main()
