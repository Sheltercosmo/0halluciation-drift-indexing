from copy import deepcopy
from io import BytesIO
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zero_index import RetrievalAdapter, document_from_text
from zero_index.application import adapter_from_config, default_application
from zero_index.providers import (CallableDecisions, CallableEmbeddings, DecisionModel,
    EmbeddingAPI, OllamaEmbeddings, OllamaDecisions, SystemOneAPI, JevAPI, SentenceTransformerEmbeddings)


def response(value):
    return BytesIO(json.dumps(value).encode())


class ProviderTests(unittest.TestCase):
    def test_embedding_api_restores_order_and_batches_deduplicated_inputs(self):
        calls = []
        def serve(request, **kwargs):
            body = json.loads(request.data)
            calls.append(body)
            return response({"data": [{"index": i, "embedding": [len(text), 1]}
                                       for i, text in reversed(list(enumerate(body["input"])))]})
        embed = EmbeddingAPI("custom", endpoint="http://localhost:9000/v1/embeddings", batch_size=2)
        with patch("zero_index.providers.urlopen", serve):
            self.assertEqual(embed.embed(["a", "bbb", "a", "cc"], "splitting"), [[1,1],[3,1],[1,1],[2,1]])
            embed.embed(["a"], "splitting")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["encoding_format"], "float")

    def test_bad_embedding_indices_dimensions_and_nonfinite_vectors_fail(self):
        for result in ({"data": [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}]},
                       {"data": [{"index": 0, "embedding": [1]}, {"index": 1, "embedding": [2, 3]}]},
                       {"data": [{"index": 0, "embedding": [float('nan')]}, {"index": 1, "embedding": [2]}]}):
            with self.subTest(result=result), patch("zero_index.providers.urlopen", return_value=response(result)), self.assertRaises(ValueError):
                EmbeddingAPI("custom", endpoint="http://localhost/embeddings").embed(["a","b"],"x")

    def test_ollama_embedding_uses_native_endpoint_without_truncation_or_key(self):
        def serve(request, **kwargs):
            self.assertEqual(request.full_url, "http://localhost:11434/api/embed")
            self.assertNotIn("Authorization", request.headers)
            self.assertFalse(json.loads(request.data)["truncate"])
            return response({"embeddings": [[1,2]]})
        with patch("zero_index.providers.urlopen", serve):
            self.assertEqual(OllamaEmbeddings("local").embed(["text"],"central-sentences"), [[1,2]])

    def test_local_and_api_decision_contract_checks_exact_ids_and_probabilities(self):
        questions={"q0":{"type":"noul","instructions":"Is it useful?"}}
        for value in ({}, {"q0":True}, {"q0":float('nan')}, {"q0":2}, {"q0":"0.9"}, {"q0":.5,"extra":.5}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CallableDecisions(lambda *a:value,model="local").decide({}, questions)
        api=OllamaDecisions("nimble")
        def serve(request, **kwargs):
            self.assertEqual(request.full_url,"http://localhost:11434/v1/systemone")
            self.assertEqual(json.loads(request.data)["model"],"nimble")
            return response({"answers":{"q0":{"type":"noul","noul":.8}}})
        with patch("zero_index.providers.urlopen",serve):
            self.assertEqual(api.decide({"text":"source"},questions),{"q0":.8})

    def test_wire_budget_counts_structured_question_source_and_never_truncates(self):
        questions={f"q{i}":{"type":"noul","instructions":{"text":"日"*100}} for i in range(4)}
        api=SystemOneAPI("custom",endpoint="http://localhost/v1/systemone",max_request_bytes=800)
        calls=[]
        def serve(request,**kwargs):
            self.assertLessEqual(len(request.data),800)
            body=json.loads(request.data);calls.append(body)
            for q in body["questions"].values():self.assertEqual(q["instructions"]["text"],"日"*100)
            return response({"answers":{k:{"type":"noul","noul":.9} for k in body["questions"]}})
        with patch("zero_index.providers.urlopen",serve):
            self.assertEqual(len(api.decide({"question":"q"},questions)),4)
        self.assertGreater(len(calls),1)
        too_small=SystemOneAPI("m",endpoint="http://localhost/v1/systemone",max_request_bytes=10)
        with self.assertRaises(ValueError):too_small.decide({},questions)
        self.assertEqual(too_small.http.requests,0)

    def test_api_auth_and_request_caps_are_explicit(self):
        with patch.dict(os.environ,{"TEST_JEV_KEY":"test-only"}):
            api=JevAPI(api_key_env="TEST_JEV_KEY",max_requests=1)
        def serve(request,**kwargs):
            self.assertEqual(request.headers["Authorization"],"Bearer test-only")
            return response({"answers":{"q":{"type":"noul","noul":.2}}})
        with patch("zero_index.providers.urlopen",serve):
            api.decide({}, {"q":{}})
            with self.assertRaises(RuntimeError):api.decide({}, {"q":{}})
        with patch.dict(os.environ,{},clear=True), self.assertRaises(ValueError):JevAPI()

    def test_local_decision_question_count_limit_splits_large_batches(self):
        api=OllamaDecisions("nimble")
        sizes=[]
        def serve(request,**kwargs):
            questions=json.loads(request.data)['questions'];sizes.append(len(questions))
            return response({'answers':{k:{'type':'noul','noul':.5} for k in questions}})
        with patch('zero_index.providers.urlopen',serve):
            result=api.decide({'text':'source'}, {f'q{i}':{'type':'noul','instructions':'Useful?'} for i in range(65)})
        self.assertEqual(len(result),65)
        self.assertEqual(sizes,[64,1])

    def test_backend_scopes_and_cache_match_existing_measured_clients(self):
        from tests.test_jev_scoped_client import JevInputScopeRegression
        from scripts.jev_scoped_client import ScopedEvidenceJev
        from scripts.jev_evidence_comparison import EvidenceComparisonJev
        from scripts.jev_joint_evidence import SharedSetJev
        harness=JevInputScopeRegression();self.addCleanup(harness.doCleanups)
        cards=[{"node_id":"p0","kind":"paragraph","text":"Exact evidence"},
               {"node_id":"p1","kind":"paragraph","text":"Other source"}]
        packet={"target_ids":["p0","p1"],"passages":cards,"selection_size":1}
        for cls,method,args in ((ScopedEvidenceJev,"route_content",("q","need",cards)),
                               (EvidenceComparisonJev,"compare",("q",[(cards[0],cards[1])])),
                               (SharedSetJev,"score_pool",("q",packet,["p0","p1"]))):
            with self.subTest(method=method):
                calls=[]
                def decide(state,questions):
                    calls.append(deepcopy((state,questions)))
                    return {k:.73 for k in questions}
                new=DecisionModel(CallableDecisions(decide,model="local"),batch_size=64)
                old=harness.harness(cls)
                self.assertEqual(getattr(new,method)(*args),getattr(old,method)(*args))
                self.assertEqual(calls,old.requests)
                getattr(new,method)(*args)
                self.assertEqual(len(calls),1)

    def test_in_process_models_run_full_eej_and_hybrid_without_http(self):
        embed=CallableEmbeddings(lambda texts,purpose:[[1,len(t)] for t in texts],model="local-vector")
        decision=CallableDecisions(lambda state,qs:{k:.95 for k in qs},model="local-decision")
        adapter=RetrievalAdapter.from_models(embed,decision)
        doc=document_from_text("# Title\r\n\r\nFirst paragraph.\r\n\r\n## Detail\r\n\r\nSecond paragraph.")
        with patch("zero_index.providers.urlopen",side_effect=AssertionError("Unexpected network call")):
            index=adapter.build_index(doc)
            self.assertEqual(decision.requests,0)
            result=adapter.retrieve(doc,index,"Find evidence",dense_ranking=adapter.direct_embedding_ranking(doc,"Find evidence"))
        self.assertTrue(result["paragraphs"])
        self.assertEqual(result["mode"],"hybrid")
        for p in result["paragraphs"]:self.assertEqual(p["text"],doc["text"][p["start"]:p["end"]])

    def test_config_and_ee_indexing_do_not_require_a_decision_api_key(self):
        value=default_application();value["decision"]={"provider":"typesafe"}
        with patch.dict(os.environ,{},clear=True):
            adapter=adapter_from_config(value,indexing_only=True)
            self.assertEqual(adapter.config.variant,"EEJ")
            with self.assertRaises(ValueError):adapter_from_config(value)
        value["embedding"]["api_key"]="do-not-save"
        with self.assertRaises(ValueError):adapter_from_config(value)

    def test_optional_local_encoder_honors_offline_and_rejects_overflow(self):
        calls=[]
        class Encoder:
            max_seq_length=3
            def __init__(self,*args,**kwargs):calls.append(kwargs)
            def tokenizer(self,texts,**kwargs):return {"input_ids":[list(range(len(t))) for t in texts]}
            def encode(self,texts,**kwargs):return SimpleNamespace(tolist=lambda:[[1,2] for _ in texts])
        with patch.dict('sys.modules', {'sentence_transformers':SimpleNamespace(SentenceTransformer=Encoder)}):
            local=SentenceTransformerEmbeddings('local/path',local_files_only=True)
            self.assertEqual(local.embed(['ok'],'x'),[[1,2]])
            with self.assertRaises(ValueError):local.embed(['too long'],'x')
        self.assertTrue(calls[0]['local_files_only'])
        self.assertFalse(calls[0]['trust_remote_code'])


if __name__ == '__main__':unittest.main()
