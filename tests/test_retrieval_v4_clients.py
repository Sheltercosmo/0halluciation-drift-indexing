import unittest
from scripts.retrieval_v4_clients import EvidenceJevV4


class CapturedBatch:
    def _batch(self,keys,items,payload):
        self.keys=keys
        self.state,self.questions=payload([items[k] for k in keys])
        return [.8]*len(keys)


class FinalEvidenceDecision(unittest.TestCase):
    def test_final_decision_receives_complete_source_and_original_question(self):
        scorer=CapturedBatch()
        text='Opening. '+('A source fact in the middle. '*100)+' Closing.'
        cards=[{'node_id':'p4','kind':'paragraph','heading_path':['Methods'],
                'central_sentence':'','excerpts':[{'text':text,'start':0,'end':len(text),'role':'full_source'}]}]
        scores=EvidenceJevV4.rank_paragraphs(scorer,'What did this paper use?','What did this paper use?',cards)
        self.assertEqual(scores,[.8])
        self.assertEqual(scorer.state['question_about_this_paper'],'What did this paper use?')
        self.assertEqual(scorer.state['paragraphs']['p0'],cards[0])
        self.assertEqual(scorer.keys[0][0],'direct-evidence-rank-v4')
        self.assertIn('complete source text',scorer.questions['q0']['instructions'])

    def test_narrowed_final_request_is_rejected(self):
        with self.assertRaises(ValueError):
            EvidenceJevV4.rank_paragraphs(CapturedBatch(),'Original question','Narrow need',[])


if __name__=='__main__':unittest.main()
