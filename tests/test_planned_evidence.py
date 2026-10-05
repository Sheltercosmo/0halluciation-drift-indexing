import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.bounded_clients import Budget
from scripts.experiment_clients import StructuredCodex
from scripts.planned_evidence_trial import PlannedJev,validate_plan
from zero_index.evidence import select_evidence


class EvidenceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.doc={'title':'T','text':'a'*20+'\n\n'+'b'*20+'\n\n'+'c'*20}
        self.chunks=[{'id':str(i),'start':i*22,'end':i*22+20,'text':letter*20,'heading':'H'}
                     for i,letter in enumerate('abc')]

    def test_complementary_evidence_displaces_redundant_high_relevance_passage(self):
        result=select_evidence(self.doc,self.chunks,[.9,.89,.8],[[.9,0],[.9,0],[0,.9]],len,budget=70)
        self.assertEqual(result['selected_ids'],['0','2'])
        self.assertEqual(result['need_coverage'],[.9,.9])
        self.assertLessEqual(len(result['context']),70)
        self.assertIn('c'*20,result['context'])
        self.assertNotIn('b'*20,result['context'])

    def test_source_order_and_exact_budget_are_preserved(self):
        result=select_evidence(self.doc,self.chunks,[.8,.5,.9],[[.8],[.5],[.9]],len,budget=70)
        self.assertEqual(result['selected_ids'][0],'2')
        self.assertLess(result['context'].index('a'*20),result['context'].index('c'*20))
        exact=len(result['context'])
        self.assertEqual(select_evidence(self.doc,self.chunks,[.8,.5,.9],[[.8],[.5],[.9]],len,budget=exact)['context'],result['context'])

    def test_invalid_scores_and_source_cannot_enter_context(self):
        for relevance in ([.1,.2], [.1,float('nan'),.3], [.1,True,.3], [.1,-1,.3]):
            with self.assertRaises(ValueError):
                select_evidence(self.doc,self.chunks,relevance,[[.1]]*3,len)
        with self.assertRaisesRegex(ValueError,'differs from source'):
            select_evidence(self.doc,[{**self.chunks[0],'text':'invented'}],[.9],[[.9]],len)
        with self.assertRaisesRegex(ValueError,'disjoint'):
            select_evidence(self.doc,[self.chunks[0],self.chunks[0]],[.9,.9],[[.9],[.9]],len)

    def test_zero_evidence_does_not_fill_context_and_title_budget_is_checked(self):
        result=select_evidence(self.doc,self.chunks,[0]*3,[[0]]*3,len)
        self.assertEqual(result['spans'],[])
        with self.assertRaisesRegex(ValueError,'title alone'):
            select_evidence(self.doc,[],[],[],len,budget=1)

    def test_plans_are_small_distinct_search_needs(self):
        self.assertEqual(validate_plan({'needs':[' First event ','Final outcome']}),['First event','Final outcome'])
        for value in ({'needs':[]},{'needs':['a']*2},{'needs':['x'*301]},{'needs':[True]}):
            with self.assertRaises(ValueError):
                validate_plan(value)

    def test_need_scoring_batches_shared_source_and_preserves_matrix_order(self):
        class Fake(PlannedJev):
            def __init__(self):
                self._cache={}
                self.batch_size=64
                self.max_state_chars=100000
                self.max_concurrency=1
                self.payloads=[]
            def _request(self,state,questions):
                self.payloads.append((state,questions))
                return {k:(i+1)/10 for i,k in enumerate(questions)}
        scorer=Fake()
        cards=[{'node_id':str(i),'text':c,'heading_path':['H']} for i,c in enumerate(['first','last'])]
        self.assertEqual(scorer.needs('q',['start','finish'],cards),[[.1,.2],[.3,.4]])
        state,questions=scorer.payloads[0]
        self.assertEqual((len(state['candidates']),len(state['needs']),len(questions)),(2,2,4))
        scorer.needs('q',['start','finish'],cards)
        self.assertEqual(len(scorer.payloads),1)


class StructuredClientTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        with patch.dict(os.environ,{'CODEX_EVAL_BINARY':'fixture'}):
            self.client=StructuredCodex(self.root,Budget(self.root/'budget.json'))
        self.request=[{'id':'q','question':'Q','context':'C'}]

    def process(self,value,tool=False):
        def call(args,**kwargs):
            path=Path(args[args.index('--output-last-message')+1])
            path.write_text(json.dumps(value),encoding='utf-8')
            events=[{'type':'turn.completed','usage':{'input_tokens':100,'output_tokens':10}}]
            if tool:
                events.insert(0,{'type':'item.completed','item':{'type':'command_execution'}})
            return types.SimpleNamespace(returncode=0,stdout='\n'.join(json.dumps(e) for e in events))
        return call

    def test_rejected_response_usage_is_retained_and_resume_does_not_reset_retry_cap(self):
        with patch('scripts.experiment_clients.subprocess.run',side_effect=self.process({'answers':[]})) as run:
            with self.assertRaises(ValueError):
                self.client.call(self.request,'reader')
            with self.assertRaisesRegex(ValueError,'retry limit'):
                self.client.call(self.request,'reader')
            self.assertEqual(run.call_count,2)
        rows=[json.loads(x) for x in (self.root/'codex-audit.jsonl').read_text().splitlines()]
        self.assertEqual([r['usage']['input_tokens'] for r in rows],[100,100])
        self.assertEqual(len(list((self.root/'codex-responses').glob('*.json'))),2)
        self.assertEqual(self.client.budget.value['codex_calls_reserved'],2)

    def test_reader_reuses_exact_cache_and_reserves_only_real_attempt(self):
        value={'answers':[{'id':'q','answer':'A'}]}
        with patch('scripts.experiment_clients.subprocess.run',side_effect=self.process(value)) as run:
            self.assertEqual(self.client.call(self.request,'reader'),value)
            self.assertEqual(self.client.call(self.request,'reader'),value)
            self.assertEqual(run.call_count,1)
        self.assertEqual(self.client.budget.value['codex_calls_reserved'],1)

    def test_any_tool_use_rejects_prediction_without_cache(self):
        with patch('scripts.experiment_clients.subprocess.run',side_effect=self.process({'answers':[{'id':'q','answer':'A'}]},True)):
            with self.assertRaisesRegex(RuntimeError,'used tools'):
                self.client.call(self.request,'reader')
        self.assertEqual(list(self.client.cache.glob('*.json')),[])


if __name__=='__main__':
    unittest.main()
