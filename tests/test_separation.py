import unittest

from zero_index import normalized_separation


class SeparationScoreTests(unittest.TestCase):
    def test_equal_leaf_signal_and_parent_reference_has_no_separation_gain(self):
        result=normalized_separation(.75,.25,.5,threshold=.3)
        self.assertEqual(result['score'],1)
        self.assertFalse(result['accept_split'])

    def test_lower_score_favors_a_split_and_threshold_is_inclusive(self):
        weak=normalized_separation(.75,.25,.5,threshold=.5)
        strong=normalized_separation(.5,.25,.5,threshold=.5)
        self.assertGreater(weak['score'],strong['score'])
        self.assertEqual(strong['score'],.5)
        self.assertTrue(strong['accept_split'])

    def test_clipping_preserves_raw_components(self):
        high=normalized_separation(.9,.1,.4,threshold=.3)
        low=normalized_separation(.1,.9,.4,threshold=.3)
        self.assertEqual(high['score'],1)
        self.assertEqual(high['raw_ratio'],2)
        self.assertEqual(low['score'],0)
        self.assertEqual(low['raw_ratio'],-2)
        self.assertTrue(high['clipped'] and low['clipped'])
        self.assertTrue(low['accept_split'])

    def test_weak_parent_reference_abstains_instead_of_dividing_by_epsilon(self):
        for parent in (0,.001,.049):
            result=normalized_separation(.1,.9,parent,threshold=.3)
            self.assertEqual(result['status'],'insufficient_parent_reference')
            self.assertIsNone(result['accept_split'])
            self.assertIsNone(result['score'])
            self.assertIsNone(result['raw_ratio'])

    def test_invalid_inputs_and_accept_everything_threshold_are_rejected(self):
        for value in (True,-1,1.1,float('nan'),float('inf'),'0.9'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                normalized_separation(value,.2,.5,threshold=.3)
        with self.assertRaises(ValueError):normalized_separation(.5,.2,.5,threshold=1)
        with self.assertRaises(ValueError):normalized_separation(.5,.2,.5,threshold=.3,minimum_parent_probability=0)


if __name__=='__main__':unittest.main()
