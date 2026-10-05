import unittest
from scripts.retrieval_v3_metrics import align_references,paragraph_metrics,oracle_at_k


class ReferenceAlignmentTests(unittest.TestCase):
    def test_whitespace_normalization_does_not_turn_heading_into_paragraph(self):
        doc={'text':'A fact.  \n\nOther fact.', 'units':[
            {'paragraph':0,'start':0,'end':9,'heading':'Results'},
            {'paragraph':1,'start':11,'end':22,'heading':'Results'}]}
        refs,diagnostics,raw=align_references(doc,[
            {'unanswerable':False,'evidence':['A fact.']},
            {'unanswerable':False,'evidence':['Results']},
            {'unanswerable':False,'evidence':['A fact.','missing']}])
        self.assertEqual(refs,[['p0']])
        self.assertEqual(diagnostics[1]['issues'],['heading_annotation'])
        self.assertEqual(diagnostics[2]['valid_paragraph_reference'],False)
        self.assertEqual(raw[1],['Results'])

    def test_duplicates_are_ambiguous_and_empty_is_not_perfect_retrieval(self):
        doc={'text':'Fact.Fact.', 'units':[{'paragraph':0,'start':0,'end':5,'heading':''},
                                        {'paragraph':1,'start':5,'end':10,'heading':''}]}
        refs,d,_=align_references(doc,[{'unanswerable':False,'evidence':['Fact.']}])
        self.assertFalse(refs);self.assertEqual(d[0]['issues'],['ambiguous_duplicate_paragraph'])
        self.assertIsNone(paragraph_metrics([],[])['f1'])

    def test_alternative_sets_and_budget_oracle(self):
        self.assertEqual(paragraph_metrics(['p0'],[['p0'],['p1','p2']])['f1'],1)
        self.assertAlmostEqual(oracle_at_k([list('abcdef')],5),10/11)
        self.assertEqual(paragraph_metrics([], [['p0']])['complete'],0)


if __name__=='__main__':unittest.main()
