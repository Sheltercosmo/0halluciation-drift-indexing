import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,Mock
from scripts.bounded_clients import save,signature
from scripts.bounded_eval import sha
from scripts.retrieval_v3_gate import validate_predictions,open_test


class RetrievalGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)
        self.case={'id':'q','doc_id':'d'}
        save(self.out/'cases.json',[self.case])
        save(self.out/'documents.json',{'d':{'text':'abc','units':[{'paragraph':0,'start':0,'end':3}]}})
        save(self.out/'manifest.json',{'questions':1,'input_hashes':{n:sha(self.out/n) for n in ('cases.json','documents.json')}})
        self.pack={str(b):{'paragraphs':['p0'],'skipped':[],'source_tokens':3} for b in (512,1024,2048)}
        self.methods=['tree','qwen','bge']
        for folder,name in zip(('retrieval','baselines','bge'),self.methods):
            save(self.out/folder/(signature('q')+'.json'),{'id':'q','doc_id':'d',
                'manifest_sha256':sha(self.out/'manifest.json'),
                'methods':{name:{'ranking':['p0'],'status':'complete','packed':self.pack}}})
        fake=Mock();fake.encode.side_effect=lambda text,**kw:list(text)
        self.patcher=patch('scripts.bounded_eval.tokenizer',return_value=fake);self.patcher.start();self.addCleanup(self.patcher.stop)

    def check(self):validate_predictions(self.out,self.methods,[self.case])

    def test_complete_whole_paragraph_records_pass(self):self.check()

    def test_missing_method_cannot_pass(self):
        (self.out/'bge'/(signature('q')+'.json')).unlink()
        with self.assertRaisesRegex(ValueError,'Missing prediction'):self.check()

    def test_misbound_question_cannot_pass(self):
        with self.assertRaisesRegex(ValueError,'registry'):
            validate_predictions(self.out,self.methods,[{'id':'another','doc_id':'d'}])

    def test_truncated_paragraph_cannot_pass(self):
        self.pack['512']['source_tokens']=1
        save(self.out/'bge'/(signature('q')+'.json'),{'id':'q','methods':{'bge':{
            'ranking':['p0'],'status':'complete','packed':self.pack}}})
        with self.assertRaisesRegex(ValueError,'packing'):self.check()

    def test_test_access_stops_before_labels_if_integrity_fails(self):
        with patch('scripts.retrieval_v3_gate.checked',side_effect=ValueError('Frozen source changed')), \
             patch('scripts.retrieval_v3_gate.analyze') as analyze:
            with self.assertRaisesRegex(ValueError,'Frozen'):open_test()
            analyze.assert_not_called()


if __name__=='__main__':unittest.main()
