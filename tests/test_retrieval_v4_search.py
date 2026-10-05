import unittest
from tests.test_retrieval_v4_structure import flat_index
from scripts.retrieval_v4_structure import restore_heading_hierarchy
from scripts.retrieval_v4_search import search_tree,final_paragraph_rerank,navigation_card,navigation_text
from zero_index.evidence_search import EvidenceSearchConfig,card_text


class RepairedEvidenceSearch(unittest.TestCase):
    def test_rejected_branch_is_never_opened(self):
        index=flat_index(['Relevant','Unrelated'])
        seen=[]
        def decide(question,need,cards):
            seen.extend(c['node_id'] for c in cards)
            return [0 if c['node_id']=='h1' else 1 for c in cards]
        result=search_tree(index,'question',['need'],decide)
        self.assertEqual(result['ranking'],['p0'])
        self.assertNotIn('b1',seen);self.assertNotIn('p1',seen)

    def test_terminals_have_no_beam_or_confidence_filter(self):
        index=flat_index(['Methods']*8)
        blocks=[h.children[0] for h in index.root.children]
        first=blocks[0]
        first.children=[p for b in blocks for p in b.children]
        first.end=first.children[-1].end
        index.root.children=index.root.children[:1]
        index.root.children[0].end=first.end
        # Rebuild parent lookup after constructing the wide terminal fixture.
        index=type(index).from_dict(index.to_dict())
        result=search_tree(index,'q',['n'],lambda q,n,c:[.01 if x['kind']=='paragraph' else 1 for x in c],
                           EvidenceSearchConfig(beam=1))
        self.assertEqual(set(result['ranking']),{f'p{i}' for i in range(8)})

    def test_full_paragraph_final_rerank_uses_original_question(self):
        text='First. Important detail in the middle. Last.\nAnother paragraph.'
        end=text.index('\n')
        doc={'text':text,'units':[{'paragraph':0,'start':0,'end':end,'heading':'A'},
                                {'paragraph':1,'start':end+1,'end':len(text),'heading':'B'}]}
        def decide(question,need,cards):
            self.assertEqual(question,'Complete original question')
            self.assertEqual(need,question)
            self.assertEqual(cards[0]['central_sentence'],'')
            self.assertEqual(cards[0]['excerpts'][0]['text'],text[:end])
            return [.1,.9]
        result=final_paragraph_rerank(doc,'Complete original question',['p0','p1','p0'],decide)
        self.assertEqual(result['candidate_pool'],['p0','p1'])
        self.assertEqual(result['ranking'],['p1','p0'])

    def test_missing_parent_exposes_child_headings_and_layer_order(self):
        index=restore_heading_hierarchy(flat_index(['Methods ::: Features','Results']))
        config=EvidenceSearchConfig()
        card=navigation_card(index,index.root.children[0],config)
        self.assertEqual(card['child_headings'][0]['title'],'Features')
        self.assertIn('Features',navigation_text(card))
        result=search_tree(index,'q',['n'],lambda q,n,c:[1]*len(c),config)
        seen={index.root.node_id}
        for step in result['trace']:
            self.assertTrue(set(step['parents'])<=seen)
            seen.update(step['selected_internal'])
        self.assertIn('p0',result['ranking'])

    def test_leaf_text_and_empty_pool(self):
        index=flat_index(['A']);card=navigation_card(index,index._node('p0'),EvidenceSearchConfig())
        self.assertEqual(navigation_text(card),card_text(card))
        result=final_paragraph_rerank({},'q',[],lambda *args:self.fail('Unexpected call'))
        self.assertEqual(result['ranking'],[])

    def test_refinement_reserves_budget_for_other_requests_at_layer(self):
        index=flat_index(['A','B']);called=[]
        def decide(q,n,c):
            called.extend(c)
            return [.5]*len(c)
        result=search_tree(index,'q',['n1','n2'],decide,EvidenceSearchConfig(max_decisions=5))
        self.assertLessEqual(len(called),5)
        self.assertEqual(len(called),result['decisions'])
        self.assertEqual(result['status'],'truncated')

    def test_common_final_rule_preserves_recorded_model_scores(self):
        doc={'text':'abc','units':[{'paragraph':i,'start':i,'end':i+1,'heading':'A'} for i in range(3)]}
        result=final_paragraph_rerank(doc,'q',['p0','p1','p2'],lambda *args:[.2,.9,.8],score_weight=.25)
        self.assertEqual(result['decision_scores'],[.2,.9,.8])
        self.assertEqual(result['scores'],[.8,.6,.2])
        self.assertEqual(result['ranking'],['p0','p1','p2'])


if __name__=='__main__':unittest.main()
