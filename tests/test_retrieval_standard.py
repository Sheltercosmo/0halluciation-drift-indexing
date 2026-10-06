import unittest
from scripts.bounded_eval import build_document
from scripts.retrieval_standard import STANDARD_POLICY,select_standard_evidence


class StandardPolicyTests(unittest.TestCase):
    def test_defaults_preserve_measured_policy(self):
        self.assertEqual((STANDARD_POLICY['beam'],STANDARD_POLICY['deferred_search_fraction']), (5,.25))
        self.assertEqual((STANDARD_POLICY['pairwise_candidates'],STANDARD_POLICY['shared_targets'],STANDARD_POLICY['selection_k']), (30,12,5))
        self.assertFalse(STANDARD_POLICY['expand_targets'])

    def test_shared_context_is_retained_and_both_presentations_select_five(self):
        doc=build_document('paper','Title',[('Section',[f'Paragraph {i}.' for i in range(16)])])
        rank=[f'p{i}' for i in range(16)];presentations=[]
        def decide(q,packet,targets):
            self.assertEqual(q,'Original question')
            self.assertGreater(len(packet['passages']),len(targets))
            self.assertEqual(set(targets),set(rank[:12]))
            self.assertNotIn('p12',targets)
            presentations.append([p['node_id'] for p in packet['passages']])
            return [.9 if p in rank[7:12] else .1 for p in targets]
        result=select_standard_evidence(doc,'Original question',rank,decide)
        self.assertEqual(presentations[0],presentations[1][::-1])
        self.assertEqual(result['result']['selected'],rank[7:12])
        for p in result['packet']['passages']:
            self.assertEqual(p['text'],doc['text'][p['start']:p['end']])

if __name__=='__main__':unittest.main()
