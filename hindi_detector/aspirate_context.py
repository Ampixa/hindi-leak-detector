"""Explicit unknown context for a verified nonsyllabic voiced aspirate.

This preserves only consonant-versus-vowel class. It never equates aspiration,
breathy voice, or consonant identity in the actual acoustic hypothesis.
"""
from __future__ import annotations

from hindi_detector.consonant_context import NonSyllabicTable


class AspirateContextTable(NonSyllabicTable):
    def representation(self, token, vocabulary, *, mask_all=False):
        try:
            return super().representation(token, vocabulary, mask_all=mask_all)
        except ValueError:
            # The pinned table does not segment modifier-letter voiced
            # aspiration U+02B1. Admit a class-only context iff the base,
            # explicitly aspirated base, and explicitly breathy base are ALL
            # completely parsed as nonsyllabic in that same pinned table.
            if not token.endswith('ʱ') or not token[:-1]:
                raise
            base = token[:-1]
            witnesses = [base, base + 'ʰ', base + '\u0324']
            if any(self.classify(value)['status'] != 'pure_non_syllabic' for value in witnesses):
                raise
            pool, metadata = self.pool(vocabulary)
            return pool, None, {'ipa': token,
                'representation_mode': 'verified_nonsyllabic_voiced_aspirate_unknown_context',
                'class_only_witnesses': witnesses, 'consonant_identity_tested': False,
                'context_pool_sha256': metadata['sha256'], 'context_pool_size': len(pool)}
