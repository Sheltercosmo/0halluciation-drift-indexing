import unittest
from scripts.jev_evidence_comparison import EvidenceComparisonJev, pairwise_rerank, whole_paragraph_cards
from scripts.retrieval_v4_search import final_paragraph_rerank
from tests import test_jev_scoped_client as scope_tests


def document(n=3):
    text = ''
    units = []
    for i in range(n):
        start = len(text)
        text += f'First {i}. Specific evidence deep in paragraph {i}. Last {i}.'
        units.append({'paragraph': i, 'heading': 'Results', 'start': start, 'end': len(text)})
        text += '\n\n'
    return {'text': text, 'units': units}


class EvidenceComparisonTests(unittest.TestCase):
    def test_old_rank_prior_cannot_promote_best_last_paragraph(self):
        doc = document(30)
        r = final_paragraph_rerank(doc, 'question', [f'p{i}' for i in range(30)],
                                  lambda q,n,c: [0.] * 29 + [1.], score_weight=.25)
        self.assertEqual(r['ranking'].index('p29'), 20)

    def test_pairwise_recovers_evidence_without_rank_prior(self):
        doc = document(30)
        def compare(q,pairs):
            return [.95 if int(a['node_id'][1:]) > int(b['node_id'][1:]) else .05 for a,b in pairs]
        r = pairwise_rerank(doc,'question',[f'p{i}' for i in range(30)],compare)
        self.assertEqual(r['ranking'][0], 'p29')
        self.assertEqual(r['decisions'], 30*29)
        self.assertEqual(len(r['comparisons']), 435)

    def test_orientation_conflict_ties_instead_of_favoring_A(self):
        r = pairwise_rerank(document(),'q',['p0','p1','p2'],lambda q,pairs: [.9]*len(pairs))
        self.assertEqual(r['ranking'], ['p0','p1','p2'])
        self.assertEqual(r['scores'], [1.,1.,1.])
        self.assertTrue(all(p['a_win']==.5 for p in r['comparisons']))

    def test_full_paragraphs_identity_and_empty_pool(self):
        doc=document();cards=whole_paragraph_cards(doc,['p1','p1','p2'])
        self.assertEqual(len(cards),2)
        for c in cards:
            u=doc['units'][int(c['node_id'][1:])]
            self.assertEqual(c['excerpts'][0]['text'],doc['text'][u['start']:u['end']])
        self.assertEqual(pairwise_rerank(doc,'q',[],lambda q,p: self.fail('Empty call'))['ranking'],[])
        with self.assertRaises(ValueError):pairwise_rerank(doc,'q',['p1','p2'],lambda q,p:[float('nan')]*2)

    def test_per_pair_inputs_do_not_depend_on_other_comparisons(self):
        harness=scope_tests.JevInputScopeRegression()
        self.addCleanup(harness.doCleanups)
        client=harness.harness(EvidenceComparisonJev)
        cards=whole_paragraph_cards(document(),['p0','p1','p2'])
        client.compare('original question',[(cards[0],cards[1]),(cards[0],cards[2])])
        state,questions=client.requests[0]
        self.assertEqual(state,{'question':'original question'})
        self.assertEqual(questions['q0']['instructions']['A'],cards[0])
        self.assertEqual(questions['q0']['instructions']['B'],cards[1])
        self.assertNotIn(cards[2]['excerpts'][0]['text'], str(questions['q0']))


if __name__=='__main__':unittest.main()
