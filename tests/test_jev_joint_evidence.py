import unittest
from copy import deepcopy
from scripts.jev_joint_evidence import SharedSetJev,evidence_packet,joint_select
from tests import test_jev_scoped_client as scope_tests


def document():
    texts=['General summary.','We introduce a corrected measure,','$x = y$','where y is the baseline.',
           'A separate experiment.','The following approaches are tested:','First method.','Other section.']
    units=[];text=''
    for i,t in enumerate(texts):
        units.append({'paragraph':i,'start':len(text),'end':len(text)+len(t),'section':int(i==7),'heading':'Methods'})
        text+=t+'\n\n'
    return {'title':'Research document','text':text,'units':units}


class JointEvidenceTests(unittest.TestCase):
    def test_following_definition_is_visible_and_can_become_selectable_without_crossing_section(self):
        doc=document();rank=['p1','p6'];plain=evidence_packet(doc,rank,base_limit=2,context=True)
        expanded=evidence_packet(doc,rank,base_limit=2,context=True,expand=True)
        self.assertIn('p2',[p['node_id'] for p in plain['passages']])
        self.assertNotIn('p2',plain['target_ids'])
        self.assertIn('p2',expanded['target_ids'])
        self.assertNotIn('p7',[p['node_id'] for p in expanded['passages']])
        self.assertEqual(plain['target_ids'],['p1','p6'])
        for p in expanded['passages']:
            self.assertEqual(p['text'],doc['text'][p['start']:p['end']])
        self.assertTrue(any(x['context']=='p2' and x['target']=='p1' for x in expanded['links']))

    def test_joint_selection_can_replace_any_slot_but_never_returns_auxiliary_only_passage(self):
        packet=evidence_packet(document(),['p0','p1','p4','p5','p6'],base_limit=5)
        seen=[]
        def decide(q,p,ids):
            self.assertEqual(q,'original request');seen.append([x['node_id'] for x in p['passages']])
            return [{'p0':.1,'p1':.9,'p4':.2,'p5':.8,'p6':.7}[i] for i in ids]
        r=joint_select('original request',packet,['p0','p1','p4','p5','p6'],decide,k=3)
        self.assertEqual(r['ranking'][:3],['p1','p5','p6'])
        self.assertEqual(seen[0],list(reversed(seen[1])))
        self.assertEqual(r['decisions'],10)
        self.assertNotIn('p2',r['ranking'])

    def test_common_context_membership_and_order_are_part_of_cache_identity(self):
        h=scope_tests.JevInputScopeRegression();self.addCleanup(h.doCleanups)
        c=h.harness(SharedSetJev);packet=evidence_packet(document(),['p0','p1'],base_limit=2)
        c.score_pool('q',packet,['p0']);c.score_pool('q',packet,['p0'])
        self.assertEqual(len(c.requests),1)
        changed=deepcopy(packet);changed['passages'].reverse()
        c.score_pool('q',changed,['p0'])
        self.assertEqual(len(c.requests),2)
        changed=deepcopy(packet);changed['passages'][1]['text']='Changed peer context'
        c.score_pool('q',changed,['p0'])
        self.assertEqual(len(c.requests),3)
        self.assertEqual(c.requests[0][0]['evidence_packet'],packet)

    def test_question_batching_never_changes_shared_packet(self):
        h=scope_tests.JevInputScopeRegression();self.addCleanup(h.doCleanups)
        c=h.harness(SharedSetJev);c.batch_size=1
        packet=evidence_packet(document(),['p0','p1'],base_limit=2)
        c.score_pool('q',packet,packet['target_ids'])
        self.assertEqual(len(c.requests),2)
        self.assertEqual(c.requests[0][0],c.requests[1][0])
        self.assertEqual(c.requests[1][1]['q0']['instructions']['target_id'],'p1')

    def test_duplicate_unknown_and_invalid_scores_fail_closed(self):
        with self.assertRaises(ValueError):evidence_packet(document(),['p1','p1'])
        packet=evidence_packet(document(),['p0','p1'])
        with self.assertRaises(ValueError):joint_select('q',packet,['p0','p1'],lambda q,p,t:[float('nan')]*len(t))


if __name__=='__main__':unittest.main()
