import unittest

from hindi_detector.edit_geometry import map_sample,map_word


class EditGeometryTests(unittest.TestCase):
    def test_internal_removal_shortens_only_word_end(self):
        w=map_word({'start':1.,'end':2.},20000,24000,0)
        self.assertEqual(w,{'start':1.,'end':1.75})

    def test_edit_before_word_shifts_both_boundaries(self):
        w=map_word({'start':1.,'end':2.},10000,14000,0)
        self.assertEqual(w,{'start':.75,'end':1.75})

    def test_edit_after_word_cannot_shorten_word(self):
        word={'start':1.,'end':2.}
        self.assertEqual(map_word(word,33000,35000,0),word)

    def test_boundary_inside_replacement_maps_affinely(self):
        self.assertEqual(map_sample(15,10,20,4),12)
        self.assertEqual(map_sample(15,10,20,0),10)

    def test_sham_has_identity_geometry(self):
        for value in (0,10,15,20,30):self.assertEqual(map_sample(value,10,20,10),value)

    def test_whole_word_removal_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(ValueError,'entire aligned word'):
            map_word({'start':1.,'end':2.},10000,35000,0)


if __name__=='__main__':unittest.main()
