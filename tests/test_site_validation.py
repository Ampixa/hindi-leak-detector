import unittest

import numpy as np

from hindi_detector.site_validation import validate_site


class LocalSiteValidationTests(unittest.TestCase):
    def test_ratio_without_an_observed_reference_vowel_is_not_a_diagnosis(self):
        # A poorly fitting native reference must not validate an alternative
        # merely because that alternative fits even worse.
        x = np.full((12, 6), .001)
        x[:, 0] = .2; x[:4, 1] = .79; x[4:8, 4] = .79; x[8:, 3] = .79
        x /= x.sum(1, keepdims=True)
        values = np.log(x)
        compiled = {'any_vowel': frozenset({2, 4, 5})}
        variant = {'units': [{1}, {2}, {3}]}
        contrast = {'kind':'vowel_quality','index':1,'alternative_unit':{4}}
        evidence = {'retained_native_indices':[0,1,2]}
        crops = [(p, 0, values) for p in (0,.03,.06)]
        result = validate_site(compiled, variant, contrast, evidence, crops, crops)
        self.assertEqual(result['status'], 'weak_local_phonetic_support')
        self.assertTrue(any(r['role']=='reference_expected_vowel' for r in result['failed_checks']))

    def test_deletion_requires_surviving_consonant_anchors(self):
        def frames(sequence):
            x = np.full((len(sequence),6),.001)
            x[np.arange(len(sequence)),sequence]=.995
            return np.log(x/x.sum(1,keepdims=True))
        ref=frames([0,1,0,2,2,0,3,0]);query=frames([0,1,0,0,0,0,3,0])
        compiled={'any_vowel':frozenset({2,4,5})};variant={'units':[{1},{2},{3}]}
        contrast={'kind':'vowel_deletion','index':1};evidence={'retained_native_indices':[0,1,2]}
        q=[(p,0,query) for p in (0,.03,.06)];r=[(p,0,ref) for p in (0,.03,.06)]
        result=validate_site(compiled,variant,contrast,evidence,q,r)
        self.assertEqual(result['status'],'supported_local_vowel_diagnosis')
        self.assertEqual(sum(c['role']=='query_surviving_consonant' for c in result['checks']),2)
