import unittest

from hindi_detector.wordlab_server import audio_byte_range


class AudioRangeTests(unittest.TestCase):
    def test_full_and_partial_requests_preserve_original_bytes(self):
        body = b'0123456789'
        self.assertEqual(audio_byte_range(body, None), (200, body, {'Accept-Ranges': 'bytes'}))
        for header, expected, content_range in [
                ('bytes=0-', body, 'bytes 0-9/10'),
                ('bytes=3-5', b'345', 'bytes 3-5/10'),
                ('bytes=7-999', b'789', 'bytes 7-9/10'),
                ('bytes=-3', b'789', 'bytes 7-9/10')]:
            status, result, headers = audio_byte_range(body, header)
            self.assertEqual((status, result, headers['Content-Range']), (206, expected, content_range))

    def test_invalid_or_multi_ranges_are_explicitly_unsatisfiable(self):
        for header in ('bytes=10-', 'bytes=5-2', 'bytes=-0', 'bytes=-', 'bytes=0-1,3-4', 'units=0-1', 'bytes=' + '9' * 130 + '-'):
            status, body, headers = audio_byte_range(b'0123456789', header)
            self.assertEqual(status, 416)
            self.assertEqual(body, b'')
            self.assertEqual(headers['Content-Range'], 'bytes */10')


if __name__ == '__main__':
    unittest.main()
