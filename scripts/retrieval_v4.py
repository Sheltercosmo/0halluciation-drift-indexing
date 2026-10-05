"""Resumable repaired retrieval experiment; source-only inference, common reranker.

Prepare development first. Test preparation freezes implementation and inputs;
scoring is a separate command and requires complete, validated predictions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bounded_clients import save,signature
from scripts.bounded_eval import read_json,sha,tokenizer,bm25
from scripts.retrieval_v3 import ARMS,CACHE,OLD,index_path,clients,source_hashes
from scripts.retrieval_v4_clients import EvidenceJevV4
from scripts.retrieval_v3_planner import SharedPlanner
from scripts.retrieval_v4_structure import restore_heading_hierarchy
from scripts.retrieval_v4_search import navigation_card,navigation_text,search_tree,final_paragraph_rerank
from zero_index.index import DocumentIndex
from zero_index.evidence_search import EvidenceSearchConfig,global_embedding_search,pack_paragraphs,fuse_ids

METHODS=ARMS+['gemini_dense','bm25_dense_rrf','hybrid','qwen_dense','qwen_rerank',
              'bge_dense','bge_rerank','qwen_jev_rerank','bge_jev_rerank','gemini_jev_rerank']
V4=ROOT/'output/retrieval-v4'


def prepare(partition):
    out=V4/partition
    if (out/'manifest.json').exists():raise ValueError('Manifest is immutable; use run to resume')
    source=ROOT/('output/retrieval-v3-pilot-c' if partition=='development' else 'output/retrieval-v3-confirmatory/test')
    if partition=='test':
        validate(V4/'development')
        if not (V4/'development/aligned-summary.json').exists():raise ValueError('Score development before freezing test')
        if any((source/n).exists() for n in ('aligned-scores.json','aligned-summary.json','statistics.json')):
            raise ValueError('Previous test outcomes already exist; cannot claim unopened outcomes')
    base=read_json(source/'manifest.json');cases=read_json(source/'cases.json');docs=read_json(source/'documents.json')
    artifacts={}
    for doc_id,doc in docs.items():
        for s in 'EJ':
            for r in 'EJ':
                p=index_path(doc_id,s,r)
                idx=DocumentIndex.from_dict(read_json(p))
                if idx.source!=doc['text']:raise ValueError('Index/source mismatch')
                artifacts[p.relative_to(ROOT).as_posix()]=sha(p)
    for case in cases:
        p=CACHE/'plans'/(signature([SharedPlanner.model,case['id']])+'.json')
        plan=read_json(p)
        if plan['id']!=case['id'] or plan['model']!=SharedPlanner.model:raise ValueError('Invalid plan provenance')
        artifacts[p.relative_to(ROOT).as_posix()]=sha(p)
        for folder in ('baselines','bge'):
            p=source/folder/(signature(case['id'])+'.json')
            if read_json(p)['id']!=case['id']:raise ValueError('Invalid baseline provenance')
            artifacts[p.relative_to(ROOT).as_posix()]=sha(p)
    hashes=source_hashes()
    for name in ('retrieval_v4.py','retrieval_v4_search.py','retrieval_v4_structure.py','retrieval_v4_evaluate.py','retrieval_v4_clients.py',
                 'retrieval_v3_statistics.py'):
        hashes['scripts/'+name]=sha(ROOT/'scripts'/name)
    out.mkdir(parents=True,exist_ok=True)
    save(out/'cases.json',cases);save(out/'documents.json',docs)
    manifest={'version':'tree-retrieval-v4','created_at':datetime.now(timezone.utc).isoformat(),
        'partition':partition,'questions':len(cases),'documents':len(docs),'methods':METHODS,
        'config':base['config'],'factor_order':['split','central_sentence','search'],
        'output_unit':'whole_original_paragraph','final_rerank':'same Jev original-question top-30 for all factorial arms, hybrid and Jev-reranked dense controls',
        'final_score_weight':.25,'final_score_formula':'0.25 * Jev direct-evidence probability + 0.75 * (1 - zero-based candidate rank / max(1, pool size - 1))',
        'development_selection':{'questions':16,'eligible':15,'ranking_policies_tried':['original-question useful evidence','original-question direct evidence','maximum across shared needs'],
            'direct_evidence_weights':[0,.25,.5,.75,1],'criterion':'highest JJJ development recall@5; selected weight 0.25',
            'grid_sha256':sha(V4/'development-final-ranking-grid.json')},
        'needs_policy':'shared frozen LLM evidence needs plus literal original question; deduplicate exact strings',
        'primary':'aligned paragraph recall@5','selected_system':'JJJ',
        'selection':'fixed from v3 validation; v4 engineering repairs checked on development before test',
        'changes':['restore native nested headings','show immediate subheadings in navigation',
            'retain all scored terminal paragraphs before final pool selection','include original-question search intention',
            'common original-question full-paragraph Jev final ranking'],
        'planner_model':SharedPlanner.model,'embedding_model':base['embedding_model'],'embedding_dimensions':768,
        'jev_model':'jev-1.13.0','representative_candidates':8,'representative_stop':None,
        'split_settings':base['split_settings'],'baseline_models':base.get('baseline_models',{}),
        'native_baseline_source':source.relative_to(ROOT).as_posix(),
        'sources':hashes,'artifacts':artifacts,'input_hashes':{n:sha(out/n) for n in ('cases.json','documents.json')},
        'statistics':{'draws':10000,'seed':20261005,'cluster':'document','families':[12,10,2]},
        'limitations':['Within supplied paper, not full-corpus retrieval.','Qwen 0.6B and BGE dense are compact published baselines.',
            'V3 validation informed repairs; test inputs and some v3 predictions were processed but test labels/outcomes were unopened.',
            'Common final reranking changes the comparison; do not substitute v3 validation scores for v4 test scores.']}
    save(out/'manifest.json',manifest)
    print(json.dumps({'prepared':partition,'questions':len(cases),'documents':len(docs),'manifest_sha256':sha(out/'manifest.json')}),flush=True)


def load_run(out):
    m=read_json(out/'manifest.json')
    for paths,base in ((m['sources'],ROOT),(m['artifacts'],ROOT),(m['input_hashes'],out)):
        for name,expected in paths.items():
            if sha(base/name)!=expected:raise ValueError('Frozen input changed: '+name)
    return m,read_json(out/'cases.json'),read_json(out/'documents.json')


def run(out,workers):
    manifest,cases,docs=load_run(out);config=EvidenceSearchConfig(**manifest['config'])
    budget,embed,unused=clients();enc=tokenizer();stop=threading.Event()
    count=lambda s:len(enc.encode(s,disallowed_special=()))
    source=ROOT/manifest['native_baseline_source'];manifest_hash=sha(out/'manifest.json')
    def work(case):
        if stop.is_set():return
        path=out/'retrieval'/(signature(case['id'])+'.json')
        artifact=read_json(path) if path.exists() else {'id':case['id'],'doc_id':case['doc_id'],
            'manifest_sha256':manifest_hash,'methods':{},'status':'running'}
        if artifact['manifest_sha256']!=manifest_hash:raise ValueError('Prediction manifest mismatch')
        if artifact['status']=='complete':return
        doc=docs[case['doc_id']];methods=artifact['methods']
        planned=read_json(CACHE/'plans'/(signature([SharedPlanner.model,case['id']])+'.json'))['needs']
        needs=list(dict.fromkeys([*planned,case['question']]))
        artifact['needs']=needs
        indexes={s+r:restore_heading_hierarchy(DocumentIndex.from_dict(read_json(index_path(doc['id'],s,r))))
                 for s in 'EJ' for r in 'EJ'}
        if any(i.source!=doc['text'] for i in indexes.values()):raise ValueError('Source mismatch')
        j=EvidenceJevV4(CACHE,budget);j.max_state_chars=250000
        def decide(question,need,cards):
            if stop.is_set():raise RuntimeError('Evaluation stopped')
            return j.route_content(question,need,cards)
        def write(name,result):
            result['packed']={str(b):pack_paragraphs(indexes['JJ'],result['ranking'],count,b) for b in (512,1024,2048)}
            methods[name]=result;save(path,artifact)
        def finish(name,raw):
            if stop.is_set():raise RuntimeError('Evaluation stopped')
            result=final_paragraph_rerank(doc,case['question'],raw['ranking'],j.rank_paragraphs,score_weight=manifest['final_score_weight'])
            result['retrieval']=raw;result['status']=raw['status']
            write(name,result)
        queries=[f'Question: {case["question"]}\nEvidence need: {n}' for n in needs]
        qvec=embed.embed(['task: question answering | query: '+q for q in queries],'v4-search-query')
        qlookup=dict(zip(needs,qvec))
        for sr,index in indexes.items():
            if sr+'E' not in methods:
                cards=[navigation_card(index,n,config) for n in index.root.walk() if n.kind!='document']
                texts=list(dict.fromkeys(navigation_text(c) for c in cards))
                vectors=embed.embed(['task: retrieval | document: '+t for t in texts],'v4-search-nodes')
                lookup=dict(zip(texts,vectors));by_id={c['node_id']:lookup[navigation_text(c)] for c in cards}
                raw=global_embedding_search(index,needs,lambda need,rows:[float(qlookup[need]@by_id[c['node_id']]) for c in rows],config)
                finish(sr+'E',raw)
            if sr+'J' not in methods:
                raw=search_tree(index,case['question'],needs,decide,config)
                finish(sr+'J',raw)
        if 'gemini_dense' not in methods:
            texts=[u['heading']+'\n'+doc['text'][u['start']:u['end']] for u in doc['units']]
            pv=embed.embed(['task: retrieval | document: '+t for t in texts],'v4-direct-paragraphs')
            ranks=[[f'p{i}' for i in sorted(range(len(texts)),key=lambda i:(-float(pv[i]@v),i))] for v in qvec]
            write('gemini_dense',{'ranking':fuse_ids(ranks),'status':'complete'})
        dense=methods['gemini_dense']['ranking']
        if 'bm25_dense_rrf' not in methods:
            texts=[u['heading']+'\n'+doc['text'][u['start']:u['end']] for u in doc['units']]
            values=bm25(texts,case['question'])
            lexical=[f'p{i}' for i in sorted(range(len(texts)),key=lambda i:(-values[i],i))]
            write('bm25_dense_rrf',{'ranking':fuse_ids([dense,lexical]),'status':'complete'})
        if 'gemini_jev_rerank' not in methods:
            finish('gemini_jev_rerank',{'ranking':dense,'status':'complete'})
        if 'hybrid' not in methods:
            # Fuse independent candidate routes before the common final model.
            raw={'ranking':fuse_ids([methods['JJJ']['retrieval']['ranking'],dense]),'status':methods['JJJ']['status']}
            finish('hybrid',raw)
        for folder,prefix in (('baselines','qwen'),('bge','bge')):
            native=read_json(source/folder/(signature(case['id'])+'.json'))
            for suffix in ('dense','rerank'):
                name=prefix+'_'+suffix
                if name not in methods:write(name,dict(native['methods'][name]))
            name=prefix+'_jev_rerank'
            if name not in methods:finish(name,{'ranking':methods[prefix+'_dense']['ranking'],'status':'complete'})
        if set(methods)!=set(METHODS):raise ValueError('Method coverage mismatch')
        artifact['status']='complete';save(path,artifact)
    missing=[c for c in cases if not (out/'retrieval'/(signature(c['id'])+'.json')).exists() or
        read_json(out/'retrieval'/(signature(c['id'])+'.json')).get('status')!='complete']
    print(json.dumps({'stage':'retrieve','remaining':len(missing),'total':len(cases)}),flush=True)
    pending={};iterator=iter(missing);done=len(cases)-len(missing)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            def fill():
                while len(pending)<workers and not stop.is_set():
                    item=next(iterator,None)
                    if item is None:break
                    pending[pool.submit(work,item)]=item
            fill()
            while pending:
                completed,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in completed:
                    case=pending.pop(future)
                    try:future.result()
                    except Exception:
                        stop.set()
                        for job in pending:job.cancel()
                        raise
                    done+=1;print(json.dumps({'stage':'retrieve','done':done,'total':len(cases)}),flush=True)
                fill()
    finally:embed.pool.shutdown()


def validate(out):
    manifest,cases,docs=load_run(out);expected=set(manifest['methods']);enc=tokenizer()
    count=lambda s:len(enc.encode(s,disallowed_special=()))
    traces=0
    for case in cases:
        a=read_json(out/'retrieval'/(signature(case['id'])+'.json'));doc=docs[case['doc_id']]
        if a['id']!=case['id'] or a['doc_id']!=case['doc_id'] or a['manifest_sha256']!=sha(out/'manifest.json'):
            raise ValueError('Prediction identity/provenance mismatch')
        if a['status']!='complete' or set(a['methods'])!=expected:raise ValueError('Incomplete prediction')
        index=DocumentIndex.from_dict(read_json(index_path(doc['id'],'J','J')))
        valid={f'p{u["paragraph"]}' for u in doc['units']}
        for name,result in a['methods'].items():
            ranking=result['ranking']
            if result['status']!='complete' or len(ranking)!=len(set(ranking)) or not set(ranking)<=valid:
                raise ValueError('Invalid ranking/status')
            for b in (512,1024,2048):
                if result['packed'][str(b)]!=pack_paragraphs(index,ranking,count,b):raise ValueError('Packing mismatch')
            if name in ARMS or name in ('hybrid','gemini_jev_rerank','qwen_jev_rerank','bge_jev_rerank'):
                pool=list(dict.fromkeys(result['retrieval']['ranking']))[:30]
                if pool!=result['candidate_pool'] or set(pool)!=set(ranking) or len(result['scores'])!=len(pool):
                    raise ValueError('Final pool changed')
                if result['question']!=case['question']:raise ValueError('Final reranker used narrowed request')
                weight=manifest['final_score_weight']
                recomputed=[weight*p+(1-weight)*(1-i/max(1,len(pool)-1)) for i,p in enumerate(result['decision_scores'])]
                if result['scores']!=recomputed or result['final_score_weight']!=weight:raise ValueError('Final score formula mismatch')
                order=sorted(range(len(pool)),key=lambda i:(-result['scores'][i],i))
                if ranking!=[pool[i] for i in order]:raise ValueError('Incorrect final score ordering')
            if name in ARMS and name.endswith('J'):
                repaired=restore_heading_hierarchy(DocumentIndex.from_dict(read_json(index_path(doc['id'],*name[:2]))))
                frontiers={i:{repaired.root.node_id} for i in range(len(a['needs']))}
                for step in result['retrieval']['trace']:
                    ni=step['need_index']
                    if set(step['parents'])!=frontiers[ni]:raise ValueError('Traversal jumped branches')
                    children=[c.node_id for p in step['parents'] for c in repaired._node(p).children if c.kind!='sentence']
                    if children!=step['candidates']:raise ValueError('Unvisited candidate access')
                    frontiers[ni]=set(step['selected_internal'])
                traces+=1
    receipt={'questions':len(cases),'methods':len(expected),'predictions':len(cases)*len(expected),'verified_traces':traces,
             'manifest_sha256':sha(out/'manifest.json'),'status':'complete'}
    save(out/'conformance.json',receipt);print(json.dumps(receipt),flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','validate']);p.add_argument('partition',choices=['development','test'])
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if args.stage=='prepare':prepare(args.partition)
    elif args.stage=='run':run(V4/args.partition,args.workers)
    else:validate(V4/args.partition)
