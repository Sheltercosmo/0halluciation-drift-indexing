import unittest
from scripts.retrieval_v3_jev_controls import rerank_pool


class SamePoolJevTests(unittest.TestCase):
    def test_reranker_reads_whole_paragraphs_and_preserves_candidate_pool(self):
        text='Opening. Important middle evidence. Closing.\nUnrelated paragraph.'
        split=text.index('\n')
        doc={'text':text,'units':[{'start':0,'end':split,'heading':'Methods'},
                                {'start':split+1,'end':len(text),'heading':'Background'}]}
        observed=[]
        def decide(question,need,cards):
            observed.append((question,need,cards))
            return [.1,.9]
        result=rerank_pool(doc,'question',['question'],['p1','p0'],decide)
        self.assertEqual(result['candidate_pool'],['p1','p0'])
        self.assertEqual(result['ranking'],['p0','p1'])
        card=observed[0][2][1]
        self.assertEqual(card['excerpts'][0]['text'],text[:split])
        self.assertEqual(card['heading_path'],['Methods'])
        self.assertEqual(card['central_sentence'],'')

    def test_shared_needs_keep_same_pool_for_every_ranking(self):
        doc={'text':'A\nB','units':[{'start':0,'end':1,'heading':'h'},{'start':2,'end':3,'heading':'h'}]}
        seen=[]
        def decide(q,need,cards):
            seen.append((need,[c['node_id'] for c in cards]));return [.8,.2]
        result=rerank_pool(doc,'q',['need1','need2'],['p0','p1'],decide)
        self.assertEqual(seen,[('need1',['p0','p1']),('need2',['p0','p1'])])
        self.assertEqual(result['ranking'],['p0','p1'])

    def test_duplicate_candidates_are_rejected(self):
        with self.assertRaises(ValueError):rerank_pool({},'q',['q'],['p0','p0'],lambda *x:[])


if __name__=='__main__':unittest.main()
