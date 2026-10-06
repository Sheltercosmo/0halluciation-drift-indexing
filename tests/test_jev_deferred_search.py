import unittest
from scripts.jev_deferred_search import resume_frontier
from scripts.retrieval_v4_search import search_tree
from tests.test_retrieval_v4_structure import flat_index
from scripts.retrieval_v4_structure import restore_heading_hierarchy
from zero_index.evidence_search import EvidenceSearchConfig


def starvation_fixture():
    return restore_heading_hierarchy(flat_index(['A ::: A1','A ::: A2','B']))


class DeferredSearchTests(unittest.TestCase):
    def test_retains_and_explores_accepted_branch_after_initial_beam(self):
        idx=starvation_fixture()
        def decide(q,n,cards):return [.7 if c['node_id']=='b2' else .95 for c in cards]
        old=search_tree(idx,'q',['n'],decide,EvidenceSearchConfig(beam=2))
        self.assertNotIn('p2',old['ranking'])
        new=resume_frontier(idx,'q',['n'],decide,old,20)
        self.assertIn('p2',new['ranking'])
        self.assertEqual(new['trace'],old['trace'])
        self.assertEqual(new['status'],'complete')
        self.assertEqual(new['extra_decisions'],1)
        self.assertEqual(new['deferred_trace'][0]['parent'],'b2')

    def test_rejected_branch_is_not_forced_and_zero_budget_has_no_calls(self):
        idx=starvation_fixture()
        def decide(q,n,cards):return [.01 if c['node_id']=='b2' else .95 for c in cards]
        old=search_tree(idx,'q',['n'],decide,EvidenceSearchConfig(beam=2))
        new=resume_frontier(idx,'q',['n'],lambda *args:self.fail('No accepted remainder'),old,20)
        self.assertNotIn('p2',new['ranking'])
        old=search_tree(idx,'q',['n'],lambda q,n,c:[.95]*len(c),EvidenceSearchConfig(beam=2))
        new=resume_frontier(idx,'q',['n'],lambda *args:self.fail('No allowance'),old,0)
        self.assertEqual(new['ranking'],old['ranking'])
        self.assertEqual(new['status'],'budget_exhausted')

    def test_respects_shared_budget_and_never_expands_without_scored_parent(self):
        idx=starvation_fixture();calls=[]
        def decide(q,n,c):calls.extend(c);return [.95]*len(c)
        old=search_tree(idx,'q',['n'],decide,EvidenceSearchConfig(beam=2,max_decisions=20))
        before=len(calls)
        new=resume_frontier(idx,'q',['n'],decide,old,1)
        self.assertLessEqual(new['extra_decisions'],1)
        self.assertEqual(new['extra_decisions'],len(calls)-before)
        known={nid for s in old['trace'] for nid,p in zip(s['candidates'],s['scores']) if p>=.2}
        for s in new['deferred_trace']:
            self.assertIn(s['parent'],known)
            self.assertEqual(s['candidates'],[n.node_id for n in idx._node(s['parent']).children if n.kind!='sentence'])
            known.update(s['queued_internal'])


if __name__=='__main__':unittest.main()
