"""Declared single-vowel controls with explicit unknown consonant emissions."""
from __future__ import annotations

from hindi_detector.phone_lattice import VOWELS


def template(word, ipa, vocabulary, table):
    native, edited, at = word['native_project_phones'], word['edited_project_phones'], word['site_index']
    if word['kind'] == 'vowel_deletion':
        if not 0 <= at < len(native) or native[at] != 'ax' or native[:at] + native[at + 1:] != edited:
            raise ValueError('Declared control is not exactly one Native inherent-vowel deletion')
    elif word['kind'] == 'vowel_quality':
        if len(native) != len(edited) or [i for i, (a, b) in enumerate(zip(native, edited, strict=True)) if a != b] != [at]:
            raise ValueError('Declared control is not exactly one vowel-quality substitution')
    else:
        raise ValueError('Unsupported physical vowel control kind')

    def values(phones):
        units, projections = [], []
        for phone in phones:
            if phone not in ipa or not ipa[phone]:
                raise ValueError('Missing actual Native project-phone IPA: ' + repr(phone))
            unit, _, projection = table.representation(ipa[phone], vocabulary)
            units.append(unit)
            projections.append(projection)
        return units, projections

    units, projections = values(native)
    alternative, _ = values(edited)
    vowels = frozenset(vocabulary[t] for t in VOWELS if t in vocabulary)
    if not units[at] <= vowels:
        raise ValueError('Focal declared site is not a supported vowel')
    contrast = {'kind': word['kind'], 'index': at}
    if word['kind'] == 'vowel_quality':
        if not alternative[at] <= vowels or units[at] & alternative[at]:
            raise ValueError('Quality contrast has unsupported or overlapping vowel categories')
        contrast['alternative_unit'] = alternative[at]
    return {'any_vowel': vowels}, {'phones': native, 'units': units, 'projections': projections}, contrast
