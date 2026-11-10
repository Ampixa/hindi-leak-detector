import unittest

import numpy as np

from hindi_detector.intrinsic_vowels import FEATURE_NAMES,make_paths,measure
from hindi_detector.intrinsic_detector import supported


def frames(sequence,size=10):
    values=np.full((len(sequence),size),.00001)
    values[np.arange(len(sequence)),sequence]=.99991
    values/=values.sum(1,keepdims=True)
    return np.log(values)


class IntrinsicVowelTests(unittest.TestCase):
    def setUp(self):
        self.compiled={'any_vowel':frozenset({6,7})}
        self.variant={'phones':['k','ax','n'],'units':[frozenset({4}),frozenset({6}),frozenset({5})]}

    def test_realized_vowel_and_deleted_vowel_reverse_the_margin(self):
        contrast={'kind':'vowel_deletion','index':1}
        native=frames([0,4,0,6,0,5,0]);deleted=frames([0,4,0,0,0,5,0])
        a,_=measure(self.compiled,self.variant,contrast,[(pad,0,native) for pad in (0,.03,.06)])
        b,evidence=measure(self.compiled,self.variant,contrast,[(pad,0,deleted) for pad in (0,.03,.06)])
        self.assertEqual(a.shape,(len(FEATURE_NAMES),))
        self.assertLess(a[0],0);self.assertGreater(b[0],0)
        self.assertGreater(a[8],b[8]);self.assertGreater(a[17],b[17])
        self.assertTrue(evidence['no_reference_audio_used'])

    def test_vowel_quality_requires_observed_alternative_category(self):
        contrast={'kind':'vowel_quality','index':1,'alternative_unit':frozenset({7})}
        changed=frames([0,4,0,7,0,5,0])
        values,_=measure(self.compiled,self.variant,contrast,[(pad,0,changed) for pad in (0,.03,.06)])
        self.assertTrue(supported(values[None])[0])
        values[9]=-4
        self.assertFalse(supported(values[None])[0])

    def test_nuisance_reduction_does_not_remove_focal_site(self):
        variant={'phones':['h','ax','k','ax','n'],'units':[frozenset({8}),frozenset({6}),frozenset({4}),frozenset({6}),frozenset({5})]}
        values=frames([0,4,0,6,0,5,0]);contrast={'kind':'vowel_deletion','index':3}
        _,detail=measure(self.compiled,variant,contrast,[(pad,0,values) for pad in (0,.03,.06)])
        self.assertIn(3,detail['retained_native_indices'])
        self.assertNotIn(1,detail['retained_native_indices'])

    def test_impossible_native_path_is_explicit_and_finite(self):
        values=frames([4,5]);contrast={'kind':'vowel_deletion','index':1}
        result,detail=measure(self.compiled,self.variant,contrast,[(pad,0,values) for pad in (0,.03,.06)])
        self.assertTrue(np.isfinite(result).all());self.assertEqual(result[30],0)
        self.assertTrue(all(not w['native_possible'] for w in detail['windows']))

    def test_completely_unrepresentable_site_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Neither declared pronunciation'):
            measure(self.compiled,self.variant,{'kind':'vowel_deletion','index':1},[(pad,0,frames([4])) for pad in (0,.03,.06)])

    def test_directional_gate_cannot_be_overridden_by_high_model_score(self):
        values=np.zeros((1,len(FEATURE_NAMES)),np.float32);values[0,[1,6,31]]=[.2,-.5,1]
        self.assertTrue(supported(values)[0]);values[0,1]=-.001
        self.assertFalse(supported(values)[0])


if __name__=='__main__':unittest.main()
