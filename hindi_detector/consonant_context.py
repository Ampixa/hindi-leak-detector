"""Research-only unknown consonant contexts; never assert consonant equivalence.

The acoustic pool excludes every token with a syllabic or unparsed segment.
G2P alignment keeps lexical identities separate from those pooled emissions.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from hindi_detector.corrected_synthesis import sha
from hindi_detector.phone_lattice import VOWELS, represent
from hindi_detector.pronunciation_sites import align_units


def canonical(token):
    return unicodedata.normalize('NFD', re.sub('[ˈˌ]', '', token))


class NonSyllabicTable:
    def __init__(self, directory: Path):
        inventory = json.loads((directory / 'inventory.json').read_text())
        for member in inventory['files']:
            path = (directory / member['path']).resolve(strict=True)
            if not path.is_relative_to(directory.resolve()) or sha(path) != member['sha256']:
                raise ValueError('Pinned upstream phonological feature data changed')
        signs = defaultdict(set)
        with (directory / 'ipa_all.csv').open(newline='') as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != inventory['columns'] or 'syl' not in reader.fieldnames:
                raise ValueError('Observed upstream feature schema changed')
            count = 0
            for row in reader:
                if not row['ipa'] or row['syl'] not in {'+', '-', '0'}:
                    raise ValueError('Invalid observed feature-table segment')
                signs[canonical(row['ipa'])].add(row['syl'])
                count += 1
        if count != inventory['rows']:
            raise ValueError('Pinned upstream feature row count changed')
        self.signs = dict(signs)
        self.by_start = defaultdict(list)
        for segment in signs:
            self.by_start[segment[0]].append(segment)
        for candidates in self.by_start.values():
            candidates.sort(key=lambda s: (-len(s), s))
        self.inventory = inventory
        self.pools = {}

    @lru_cache(maxsize=4096)
    def classify(self, token):
        token = canonical(token)
        # Exhaust all complete segmentations. A permissive parser that silently
        # skips an unknown character must never admit a token to this pool.
        states = {len(token): {True}}
        best = {len(token): []}
        for at in range(len(token) - 1, -1, -1):
            possibilities = set()
            for segment in self.by_start.get(token[at], []):
                end = at + len(segment)
                if not token.startswith(segment, at) or end not in states:
                    continue
                nonsyllabic = self.signs[segment] == {'-'}
                possibilities.update(nonsyllabic and rest for rest in states[end])
                if at not in best:
                    best[at] = [segment, *best[end]]
            if possibilities:
                states[at] = possibilities
        if not token or 0 not in states:
            return {'status': 'unparsed', 'canonical': token, 'segments': []}
        pure = states[0] == {True}
        return {'status': 'pure_non_syllabic' if pure else 'syllabic_mixed_or_ambiguous',
                'canonical': token, 'segments': best[0]}

    def pool(self, vocabulary):
        identity = tuple(sorted(vocabulary.items()))
        if identity in self.pools:
            return self.pools[identity]
        tokens = sorted(t for t, i in vocabulary.items() if i > 3 and self.classify(t)['status'] == 'pure_non_syllabic')
        ids = frozenset(vocabulary[t] for t in tokens)
        if not ids or ids & {vocabulary[t] for t in VOWELS if t in vocabulary}:
            raise ValueError('Unknown-context pool empty or contaminated with declared vowel tokens')
        result = ids, {'tokens': tokens, 'ids': sorted(ids),
                       'sha256': hashlib.sha256(json.dumps(tokens, ensure_ascii=False).encode()).hexdigest()}
        self.pools[identity] = result
        return result

    def representation(self, token, vocabulary, *, mask_all=False):
        normalized = canonical(token)
        strict = None
        try:
            strict = represent(token, vocabulary)
        except ValueError:
            exact = frozenset(i for t, i in vocabulary.items() if i > 3 and canonical(t) == normalized)
            if exact:
                strict = exact
        classification = self.classify(token)
        if classification['status'] == 'pure_non_syllabic' and (strict is None or mask_all):
            pool, metadata = self.pool(vocabulary)
            return pool, strict, {'ipa': token, 'canonical_ipa': normalized,
                'representation_mode': 'unknown_non_syllabic_context', 'consonant_identity_tested': False,
                'context_pool_sha256': metadata['sha256'], 'context_pool_size': len(pool)}
        if strict is None:
            raise ValueError('No supported vowel/strict representation; unknown consonant context forbidden for ' + repr(token))
        mode = 'declared_representation'
        try:
            represent(token, vocabulary)
        except ValueError:
            mode = 'exact_Unicode_canonical_representation'
        return strict, strict, {'ipa': token, 'canonical_ipa': normalized,
            'representation_mode': mode, 'acoustic_tokens': sorted(t for t, i in vocabulary.items() if i in strict),
            'consonant_identity_tested': None}


def compile_context_word(row, ipa, vocabulary, table, *, mask_all=False):
    any_vowel = frozenset(vocabulary[t] for t in VOWELS if t in vocabulary)
    identities = {}

    def unit(token):
        emission, strict, projection = table.representation(token, vocabulary, mask_all=mask_all)
        normalized = canonical(token)
        if normalized not in identities:
            identities[normalized] = -1000 - len(identities)
        # Exact lexical identity is shared; broad acoustic C emissions never
        # make two different unsupported consonants match in G2P alignment.
        alignment = (strict or frozenset()) | {identities[normalized]}
        return emission, alignment, projection

    empty = [i for i, token in enumerate(row['hindi_units']) if token == '']
    hindi_tokens = [token for token in row['hindi_units'] if token != '']
    if not hindi_tokens:
        raise ValueError('Hindi G2P has no nonempty phonetic tokens')
    hindi = [unit(token) for token in hindi_tokens]
    variants = []
    for number, phones in enumerate(row['nepali_project_variants']):
        phones = [phone for phone in phones if phone != '.']
        if not phones or any(phone not in ipa or not ipa[phone] for phone in phones):
            raise ValueError('Native G2P contains missing/empty project-phone IPA')
        values = [unit(ipa[phone]) for phone in phones]
        alignment = align_units([value[1] for value in values], [value[1] for value in hindi])
        contrasts = []
        for ni, hi in alignment:
            if ni is None:
                continue
            if hi is None:
                if phones[ni] == 'ax':
                    contrasts.append({'kind': 'vowel_deletion', 'index': ni, 'expected': ipa[phones[ni]], 'alternative': None})
                continue
            a, b = values[ni][0], hindi[hi][0]
            if not a & b and a <= any_vowel and b <= any_vowel:
                contrasts.append({'kind': 'vowel_quality', 'index': ni, 'expected': ipa[phones[ni]],
                                  'alternative': hindi_tokens[hi], 'alternative_unit': b})
        variants.append({'variant_id': number, 'phones': phones, 'units': [v[0] for v in values],
            'projections': [v[2] for v in values], 'contrasts': contrasts,
            'lexical_alignment': alignment, 'unknown_context_count': sum(v[2]['representation_mode'] == 'unknown_non_syllabic_context' for v in values)})
    return {'word': row['word'], 'variants': variants, 'any_vowel': any_vowel, 'hindi_units': hindi_tokens,
            'hindi_projections': [v[2] for v in hindi],
            'input_normalization': {'removed_empty_separator_positions': empty, 'original_hindi_units': row['hindi_units']},
            'context_policy': 'Consonant identity explicitly unknown where pooled; vowel-presence/quality hypothesis only.',
            'hypotheses_are_realization_gold': False}
