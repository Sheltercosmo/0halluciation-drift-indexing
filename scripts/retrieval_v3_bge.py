"""Published BGE-M3 dense retrieval plus BGE reranker v2 M3, whole paragraphs."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bounded_clients import save,signature
from scripts.bounded_eval import read_json,tokenizer
from scripts.retrieval_v3 import load_run,index_path,CACHE
from zero_index.index import DocumentIndex
from zero_index.evidence_search import pack_paragraphs

REVISIONS={'embedding':'5617a9f61b028005a4858fdac845db406aefb181',
           'reranker':'953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'}


class BGE:
    def __init__(self,mode):
        import torch
        from transformers import AutoModel,AutoModelForSequenceClassification,AutoTokenizer
        self.torch,self.mode=torch,mode
        path=ROOT/'output/retrieval-v3-models'/('bge-embed' if mode=='embedding' else 'bge-rerank')
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        cls=AutoModel if mode=='embedding' else AutoModelForSequenceClassification
        self.model=cls.from_pretrained(str(path),torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()

    def evaluate(self,texts,query=None):
        values=[]
        for i in range(0,len(texts),8):
            part=texts[i:i+8]
            tokens=self.tokenizer(part if query is None else [[query,s] for s in part],padding=True,truncation=False,return_tensors='pt')
            if tokens['input_ids'].shape[1]>8192:raise ValueError('BGE context exceeds 8192; explicit chunking required')
            tokens={k:v.to('cuda') for k,v in tokens.items()}
            with self.torch.inference_mode():
                output=self.model(**tokens)
                value=self.torch.nn.functional.normalize(output.last_hidden_state[:,0].float(),p=2,dim=1) if self.mode=='embedding' else self.torch.sigmoid(output.logits.view(-1).float())
            values.extend(value.cpu().tolist())
        return values

    def close(self):
        del self.model;self.torch.cuda.empty_cache()


def run(out):
    import numpy as np
    _,cases,docs=load_run(out)
    model=BGE('embedding')
    cache=CACHE/'bge-vectors'
    def embed(texts):
        paths=[cache/(signature([REVISIONS['embedding'],s])+'.json') for s in texts]
        missing=list(dict.fromkeys((p,s) for p,s in zip(paths,texts) if not p.exists()))
        if missing:
            vals=model.evaluate([s for p,s in missing])
            for (p,_),v in zip(missing,vals):save(p,v)
        return np.asarray([read_json(p) for p in paths])
    for i,case in enumerate(cases,1):
        path=out/'bge-pools'/(signature(case['id'])+'.json')
        if path.exists():continue
        doc=docs[case['doc_id']]
        texts=[u['heading']+'\n'+doc['text'][u['start']:u['end']] for u in doc['units']]
        vectors=embed(texts);q=embed([case['question']])[0];scores=vectors@q
        ranking=[f'p{j}' for j in sorted(range(len(texts)),key=lambda j:(-scores[j],j))]
        save(path,{'id':case['id'],'ranking':ranking})
        print(json.dumps({'stage':'bge-dense','done':i,'total':len(cases)}),flush=True)
    model.close();model=BGE('reranker')
    enc=tokenizer();count=lambda s:len(enc.encode(s,disallowed_special=()))
    for i,case in enumerate(cases,1):
        path=out/'bge'/(signature(case['id'])+'.json')
        if path.exists():continue
        doc=docs[case['doc_id']];dense=read_json(out/'bge-pools'/(signature(case['id'])+'.json'))['ranking'];pool=dense[:30]
        texts=[doc['units'][int(p[1:])]['heading']+'\n'+doc['text'][doc['units'][int(p[1:])]['start']:doc['units'][int(p[1:])]['end']] for p in pool]
        key=signature([REVISIONS['reranker'],case['question'],texts]);cache=CACHE/'bge-reranks'/(key+'.json')
        if cache.exists():scores=read_json(cache)
        else:scores=model.evaluate(texts,case['question']);save(cache,scores)
        rank=[pool[j] for j in sorted(range(len(pool)),key=lambda j:(-scores[j],j))]
        index=DocumentIndex.from_dict(read_json(index_path(doc['id'],'E','E')))
        methods={name:{'ranking':ranking,'status':'complete','packed':{str(b):pack_paragraphs(index,ranking,count,b) for b in (512,1024,2048)}}
            for name,ranking in [('bge_dense',dense),('bge_rerank',rank)]}
        save(path,{'id':case['id'],'methods':methods,'models':REVISIONS})
        print(json.dumps({'stage':'bge-rerank','done':i,'total':len(cases)}),flush=True)
    model.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();run(a.output)
