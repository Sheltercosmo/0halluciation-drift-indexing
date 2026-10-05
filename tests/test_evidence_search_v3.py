import unittest
from zero_index import build_index
from zero_index.evidence_search import EvidenceSearchConfig,global_embedding_search,promising_search,source_card,pack_paragraphs
from scripts.retrieval_v3 import evidence_metrics


class EvidenceSearchV3Tests(unittest.TestCase):
    def setUp(self):
        self.index=build_index('# Biology\n\nPlants grow. Their roots absorb water.\n\n# Finance\n\nBanks lend. Interest costs money.')

    def test_global_embedding_reaches_leaf_with_bad_ancestors(self):
        def score(need,cards):
            return [1. if c['kind']=='sentence' and 'water' in str(c['excerpts']) else -1. for c in cards]
        result=global_embedding_search(self.index,['water'],score,EvidenceSearchConfig(global_hits=1))
        self.assertEqual(len(result['ranking']),1)
        node=self.index._node(result['ranking'][0])
        self.assertEqual(node.kind,'paragraph')
        self.assertIn('roots absorb water',self.index.source[node.start:node.end])

    def test_jev_never_jumps_across_pruned_branch_and_reads_full_paragraph(self):
        seen=[]
        def decide(question,need,cards):
            seen.extend(cards)
            return [.9 if 'Biology' in c['heading_path'] else .01 for c in cards]
        result=promising_search(self.index,'?', ['water'],decide,EvidenceSearchConfig(beam=1))
        allowed={'root'}
        for row in result['trace']:
            self.assertEqual(set(row['parents']),allowed)
            for node_id in row['candidates']:
                self.assertIn(self.index._parents[node_id],allowed)
            allowed={n for n in row['selected'] if self.index._node(n).kind!='paragraph'}
        paragraph_cards=[c for c in seen if c['kind']=='paragraph']
        self.assertEqual(len(paragraph_cards),1)
        self.assertIn('Their roots absorb water.',paragraph_cards[0]['excerpts'][0]['text'])
        packed=pack_paragraphs(self.index,result['ranking'],len,1000)
        self.assertEqual(packed['paragraphs'],result['ranking'])
        self.assertEqual(packed['source_tokens'],len('Plants grow. Their roots absorb water.'))

    def test_additional_content_can_rescue_low_initial_score(self):
        inspected=[]
        def decide(q,need,cards):
            inspected.extend(cards)
            return [.9 if c['kind']=='paragraph' or 'source_sentences_total' in c else .05 for c in cards]
        result=promising_search(self.index,'?', ['water'],decide,EvidenceSearchConfig(beam=1,acceptance=.2))
        self.assertTrue(result['ranking'])
        self.assertTrue(any(c.get('source_sentences_total') for c in inspected))

    def test_packing_never_counts_partial_paragraph(self):
        paragraph=next(n for n in self.index.root.walk() if n.kind=='paragraph')
        result=pack_paragraphs(self.index,[paragraph.node_id],len,5)
        self.assertEqual(result['paragraphs'],[])
        self.assertEqual(result['source_tokens'],0)
        self.assertEqual(result['skipped'],[paragraph.node_id])

    def test_alternative_evidence_sets_are_not_unioned(self):
        class Official:
            @staticmethod
            def paragraph_f1_score(pred,gold):
                common=len(set(pred)&set(gold))
                return 2*common/(len(pred)+len(gold)) if common else 0
        result=evidence_metrics(['a'],[['a'],['b','c']],Official)
        self.assertEqual(result['f1'],1.)
        self.assertEqual(result['complete'],1)
        self.assertEqual(evidence_metrics([], [[],[]],Official)['f1'],None)
        self.assertEqual(evidence_metrics(['unrelated'],[['a']],Official)['f1'],0)

    def test_invalid_configuration_and_model_outputs_fail(self):
        with self.assertRaises(ValueError):EvidenceSearchConfig(beam=0)
        with self.assertRaises(ValueError):
            promising_search(self.index,'?', ['water'],lambda *x:[float('nan')]*len(x[-1]))


if __name__=='__main__':unittest.main()
