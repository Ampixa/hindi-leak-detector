import copy
import unittest

import numpy as np

from hindi_detector.corrected_synthesis import QualityError, require_pair, validate_wave


class CorrectedSourceTests(unittest.TestCase):
    def pair(self):
        return {'schema':'teacher_indicf5_pair_v14','systems':{
            'pocket_teacher_250000_24l':{'layers':24,'step':250000,'sha256':'2b4636e3aed0e4f2d035630e504be95fc58f494b690a851a91bfe0dd8299516f'},
            'indicf5_stock':{'repo':'ai4bharat/IndicF5','sha256':'ba7f3671180fb7784e24bd1dafc96e729a38ce02e7f6d3877cdef32525a1865c'}}}

    def test_legacy_models_cannot_be_selected(self):
        pair=self.pair();pair['systems']['pocket_nepali_v1c']=pair['systems'].pop('pocket_teacher_250000_24l')
        with self.assertRaises(ValueError):require_pair(pair)

    def test_wrong_depth_step_or_hash_rejected(self):
        for key,value in [('layers',6),('step',1309),('sha256','unknown')]:
            pair=copy.deepcopy(self.pair());pair['systems']['pocket_teacher_250000_24l'][key]=value
            with self.assertRaises(ValueError):require_pair(pair)
        require_pair(self.pair())

    def test_wave_quality_cannot_silently_accept_bad_data(self):
        for wave,rate in [(np.zeros(24000),24000),(np.full(24000,np.nan),24000),(np.ones(24000),16000)]:
            with self.assertRaises(QualityError):validate_wave(wave,rate)
