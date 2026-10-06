import json
import unittest
from scripts.retrieval_v4_clients import EvidenceJevV4
from zero_index.jev import JevScorer


class JevInputScopeRegression(unittest.TestCase):
    def harness(self, cls):
        class Capture(cls):
            def __init__(self):
                JevScorer.__init__(self,api_key='test-only',provider='typesafe',batch_size=64)
                self.requests=[]
            def _cached_request(self,state,questions,key,path):
                self.requests.append((state,questions))
                # Invariant test: output depends on every source object actually
                # visible to this question, as a model is permitted to do.
                if 'paragraphs' in state:return {q:.35 if len(state['paragraphs'])>1 else .73 for q in questions}
                return {q:.73 for q in questions}
        from tempfile import TemporaryDirectory
        from pathlib import Path
        tmp=TemporaryDirectory();self.addCleanup(tmp.cleanup)
        c=Capture();c.cache=Path(tmp.name)
        return c

    def test_old_client_reuses_score_from_different_visible_context(self):
        a={'node_id':'p0','text':'target'};b={'node_id':'p1','text':'distractor'}
        c=self.harness(EvidenceJevV4)
        self.assertEqual(c.rank_paragraphs('q','q',[a,b])[0],.35)
        self.assertEqual(c.rank_paragraphs('q','q',[a])[0],.35)
        fresh=self.harness(EvidenceJevV4)
        self.assertEqual(fresh.rank_paragraphs('q','q',[a])[0],.73)

    def test_visible_target_is_identical_across_batch_and_cache_history(self):
        from scripts.jev_scoped_client import ScopedEvidenceJev
        a={'node_id':'p0','text':'target'};b={'node_id':'p1','text':'distractor'}
        c=self.harness(ScopedEvidenceJev)
        mixed=c.rank_paragraphs('q','q',[b,a])[1]
        cached=c.rank_paragraphs('q','q',[a])[0]
        d=self.harness(ScopedEvidenceJev);single=d.rank_paragraphs('q','q',[a])[0]
        self.assertEqual((mixed,cached,single),(.73,.73,.73))
        state,questions=c.requests[0];other_state,other_questions=d.requests[0]
        self.assertEqual(state,other_state)
        self.assertNotIn('paragraphs',state)
        self.assertEqual(questions['q1'],other_questions['q0'])
        self.assertEqual(questions['q1']['instructions']['target'],a)
        self.assertEqual(len(questions),2)  # Still one parallel multi-question request.
        self.assertEqual(len(c.requests),1)  # The cache is safe and still works.

    def test_routing_uses_same_isolation_without_losing_source_text(self):
        from scripts.jev_scoped_client import ScopedEvidenceJev
        text='Opening. '+('Exact middle evidence. '*100)+' End.'
        card={'node_id':'p7','kind':'paragraph','heading_path':['Methods'],
              'excerpts':[{'text':text,'start':0,'end':len(text)}]}
        c=self.harness(ScopedEvidenceJev)
        c.route_content('full question','evidence need',[card,dict(card,node_id='p8')])
        state,questions=c.requests[0]
        self.assertEqual(state,{'question':'full question','requested_content':'evidence need'})
        self.assertEqual(questions['q0']['instructions']['target'],card)
        self.assertIn('`target`',questions['q0']['instructions']['decision'])
        self.assertNotIn('nodes.n',json.dumps(questions))


if __name__=='__main__':unittest.main()
