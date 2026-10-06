import unittest
from scripts.jev_importance import ImportanceJev,promote_important_evidence,contextual_paragraph_cards
from tests.test_jev_evidence_comparison import document
from tests import test_jev_scoped_client as scope_tests


class ImportanceTests(unittest.TestCase):
    def test_no_strong_promotion_preserves_every_candidate_and_whole_paragraph(self):
        doc=document(6);rank=[f'p{i}' for i in range(6)]
        cards=contextual_paragraph_cards(doc,rank,context=False)
        result=promote_important_evidence('q',cards,lambda q,s,p:[.5]*len(p))
        self.assertEqual(result['ranking'],rank)
        for c in cards:
            unit=doc['units'][int(c['node_id'][1:])]
            self.assertEqual(c['excerpts'][0]['text'],doc['text'][unit['start']:unit['end']])

    def test_only_agreed_important_challenger_moves_ahead_of_incumbent(self):
        cards=contextual_paragraph_cards(document(4),['p0','p1','p2','p3'],context=False)
        def compare(q,selected,pairs):
            return [.95 if a['node_id']=='p2' else .05 if b['node_id']=='p2' else .5 for a,b in pairs]
        result=promote_important_evidence('q',cards,compare,k=2)
        self.assertEqual(result['ranking'],['p0','p2','p1','p3'])
        self.assertTrue(result['selection_trace'][0]['promoted'])
        self.assertEqual(result['selection_trace'][0]['incumbent'],'p1')

    def test_orientation_conflict_does_not_promote(self):
        cards=contextual_paragraph_cards(document(3),['p0','p1','p2'],context=False)
        result=promote_important_evidence('q',cards,lambda q,s,p:[.95]*len(p),k=2)
        self.assertEqual(result['ranking'],['p0','p1','p2'])

    def test_exact_probability_threshold_accepts_both_orientations(self):
        cards=contextual_paragraph_cards(document(3),['p0','p1','p2'],context=False)
        result=promote_important_evidence('q',cards,lambda q,s,p:[.8,.2],k=2,threshold=.8)
        self.assertEqual(result['ranking'],['p0','p2','p1'])

    def test_scope_tracks_nearest_introduction_without_crossing_section(self):
        texts=['For detection, we evaluate the following models.','Classifier details.',
               'For generation, we evaluate the following models.','Generator details.',
               'Different section.']
        text='';units=[]
        for i,t in enumerate(texts):
            units.append({'paragraph':i,'start':len(text),'end':len(text)+len(t),
                          'heading':'Repeated heading','section':0 if i<4 else 1})
            text+=t+'\n\n'
        doc={'text':text,'units':units}
        a,b,c=contextual_paragraph_cards(doc,['p1','p3','p4'])
        self.assertEqual(a['context'][0]['node_id'],'p0')
        self.assertEqual(b['context'][0]['node_id'],'p2')
        self.assertEqual(c['context'],[])
        self.assertEqual(b['excerpts'][0]['text'],texts[3])

    def test_selected_context_and_source_scope_are_in_payload_and_cache_key(self):
        h=scope_tests.JevInputScopeRegression();self.addCleanup(h.doCleanups)
        client=h.harness(ImportanceJev)
        pair=({'node_id':'p2','context':[{'text':'scope'}]},{'node_id':'p1'})
        client.promote('q',[{'node_id':'p0'}],[pair])
        self.assertEqual(client.requests[0][0]['selected_evidence'],[{'node_id':'p0'}])
        self.assertEqual(client.requests[0][1]['q0']['instructions']['A'],pair[0])
        client.promote('q',[{'node_id':'p3'}],[pair])
        self.assertEqual(len(client.requests),2)


if __name__=='__main__':unittest.main()
