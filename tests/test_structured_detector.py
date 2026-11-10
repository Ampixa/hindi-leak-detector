import unittest

import numpy as np

from hindi_detector.structured_detector import context,paired_inputs,site_vector
from scripts.train_corrected_detector import matched_layout


class StructuredEvidenceTests(unittest.TestCase):
    def test_background_length_cannot_change_focal_class_weights(self):
        for length in (4,100):
            rows=[{'text_id':'paired','mask':np.ones(length,dtype=bool),'focal':[0],
                'labels':np.r_[float(label),np.zeros(length-1)]} for label in (0,1)]
            weights,groups=matched_layout(rows)
            self.assertEqual(float(weights[0]),float(weights[length]))
            self.assertAlmostEqual(float(weights.sum()),2.25,places=5)
            self.assertEqual(set(groups[0]['focal'][0]),{0,1})

    def test_incomplete_focal_source_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            matched_layout([{'text_id':'x','mask':np.array([True]),'focal':[0],'labels':np.array([1.])}])

    def test_unsupported_is_distinct_from_measured_zero_margin(self):
        absent=site_vector({'schwa_site_evidence':[{'status':'unsupported','reason':'missing dental phone'}]})
        measured=site_vector({'schwa_site_evidence':[{'sites':[{'gap_index':2,'skeleton_length':4,
            'lexicon_schwa_count':1,'alternative_schwa_count':0,'windows':[
                {'status':'supported','alternative_minus_lexicon_per_frame':0.} for _ in range(3)]}]}]})
        self.assertEqual(absent[0],0);self.assertEqual(measured[0],1)
        self.assertEqual(measured[1],0);self.assertEqual(measured[6],.5)

    def test_incomplete_boundary_support_does_not_imply_native_evidence(self):
        result=site_vector({'schwa_site_evidence':[{'sites':[{'gap_index':1,'skeleton_length':2,
            'lexicon_schwa_count':1,'alternative_schwa_count':0,'windows':[
                {'status':'supported','alternative_minus_lexicon_per_frame':3.},{'status':'impossible'}]}]}]})
        self.assertEqual(result[0],0)

    def test_context_boundaries_have_no_wraparound(self):
        result=context(np.array([[1.],[2.],[3.]],np.float32))
        np.testing.assert_array_equal(result[0,:4],0)
        np.testing.assert_array_equal(result[-1,-4:],0)
        self.assertEqual(result[1,2],1.)

    def test_temporal_change_survives_equal_pooled_phone_means(self):
        def item():
            return {'detail':{'words':[{'word':'कर्म'}]},'phonetics':np.ones((1,4),np.float32),
                    'prosody':np.zeros((1,3),np.float32),'temporal_phone':np.zeros((1,20,8),np.float32)}
        query,reference=item(),item();query['temporal_phone'][0,5,0]=1.;reference['temporal_phone'][0,15,0]=1.
        p,_=paired_inputs(query,reference)
        self.assertTrue(np.any(p!=0))
        with self.assertRaises(ValueError):
            reference['detail']['words'][0]['word']='अर्थ';paired_inputs(query,reference)
