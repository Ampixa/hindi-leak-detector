import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.detect_nepali_leakage import emit_result


class InferenceResultContractTests(unittest.TestCase):
    def test_scored_and_abstained_results_emit_a_valid_summary_without_scope_state(self):
        # The summary must be derived from the saved result, not local variables
        # in main(); otherwise inference writes a result but exits with failure.
        for status in ('scored_vowel_evidence', 'unscored_content_mismatch', 'unscored_reference_quality'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / 'result'
                result = {'status': status, 'word_flags': [], 'reference_mode': 'contract_test',
                          'model_sha256': 'contract-test-digest'}
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    emit_result(argparse.Namespace(out=out), result)
                summary = json.loads(stream.getvalue())
                self.assertEqual(summary['status'], status)
                self.assertEqual(summary['reference_mode'], result['reference_mode'])
                self.assertEqual(json.loads((out / 'result.json').read_text()), result)


if __name__ == '__main__':
    unittest.main()
