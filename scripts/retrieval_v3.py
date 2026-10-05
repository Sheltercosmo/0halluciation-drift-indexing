"""Run registered paragraph-evidence comparisons; tuning uses development only."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
import importlib.util
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bounded_clients import save,signature
from scripts.bounded_eval import read_json,sha,tokenizer,sections,bm25
from scripts.live_tree_eval import tree,cached_partitions
from scripts.live_tree_clients import LiveEmbeddings
from scripts.retrieval_v3_clients import EvidenceJev,RetrievalBudget,QwenModels
from scripts.retrieval_v3_planner import SharedPlanner
from zero_index.index import DocumentIndex,reselect_representatives
from zero_index.embeddings import CentroidRepresentatives
from zero_index.evidence_search import (EvidenceSearchConfig,source_card,card_text,
    global_embedding_search,promising_search,pack_paragraphs,fuse_ids)
from zero_index.segment import segment_runs
from zero_index.parse import Block
from zero_index.model import Config

ARMS=[''.join(p) for p in itertools.product('EJ',repeat=3)]
METHODS=ARMS+['gemini_dense','bm25_dense_rrf','hybrid','qwen_dense','qwen_rerank']
CONFIGS={'reference':asdict(EvidenceSearchConfig()),
         'b3_a0':asdict(EvidenceSearchConfig(acceptance=0)),
         'b3_a4':asdict(EvidenceSearchConfig(acceptance=.4)),
         'b5_a2':asdict(EvidenceSearchConfig(beam=5)),
         'g10':asdict(EvidenceSearchConfig(global_hits=10)),
         'g100':asdict(EvidenceSearchConfig(global_hits=100))}
CACHE=ROOT/'output/retrieval-v3-cache'
SPLITS=ROOT/'output/improvement-v1'
OLD=ROOT/'output/bounded-v1'


def source_hashes():
    names=['scripts/retrieval_v3.py','scripts/retrieval_v3_clients.py','scripts/bounded_clients.py',
           'scripts/bounded_eval.py','scripts/live_tree_clients.py','scripts/live_tree_eval.py',
           'scripts/retrieval_v3_planner.py','scripts/experiment_clients.py',
           'scripts/retrieval_v3_metrics.py','scripts/analyze_retrieval_v3.py','scripts/retrieval_v3_bge.py']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'zero_index').glob('*.py')]
    return {n:sha(ROOT/n) for n in names}


def prepare(out,partition,limit,config_name):
    if (out/'manifest.json').exists():raise ValueError('Existing registration is immutable')
    if partition!='development':
        raise ValueError('Held-out preparation requires the separate retrieval freeze gate')
    all_cases=read_json(SPLITS/partition/'cases.json')
    all_cases=[c for c in all_cases if c['dataset']=='qasper']
    old_ids={c['id'] for c in read_json(OLD/'cases.json') if c['dataset']=='qasper'}
    if limit:
        # Small pilot selected by ID hash, with no outcome/evidence filtering.
        pool=sorted([c for c in all_cases if c['id'] in old_ids],key=lambda c:signature(['v3-pilot',c['doc_id']]))
        selected=pool[:limit]
    else:selected=all_cases
    docs=read_json(SPLITS/partition/'documents.json')
    docs={d:docs[d] for d in sorted({c['doc_id'] for c in selected})}
    out.mkdir(parents=True,exist_ok=True)
    save(out/'cases.json',selected);save(out/'documents.json',docs)
    config=CONFIGS[config_name]
    manifest={'version':'tree-retrieval-v3','status':'registered_before_inference','partition':partition,
      'questions':len(selected),'documents':len(docs),'methods':METHODS,'config_name':config_name,'config':config,
      'factor_order':['split','central_sentence','search'],'output_unit':'whole_original_paragraph',
      'primary':'maximum aligned-paragraph evidence recall over acceptable references, first five original paragraphs',
      'secondary':['precision/recall/complete@1,3,5,10','whole paragraphs at 512/1024/2048 source tokens'],
      'planner_model':SharedPlanner.model,'embedding_model':LiveEmbeddings.model,'embedding_dimensions':768,
      'jev_model':'jev-1.13.0','representative_candidates':8,'representative_stop':None,
      'split_settings':{'embedding_adjacent_distance_quantile':.85,'jev_prior':.7,'cutoff':.5,'drop':.2},
      'sources':source_hashes(),'input_hashes':{n:sha(out/n) for n in ('cases.json','documents.json')},
      'split_registry_sha256':sha(ROOT/'evals/iterations/v1/registry.json'),
      'baseline_models':{'qwen_embedding':QwenModels.embedding_revision,'qwen_reranker':QwenModels.reranker_revision},
      'limitations':['Within-document retrieval; paper is supplied equally to all systems.',
        'Compact published Qwen 0.6B checkpoints; not the 8B models or a frontier claim.',
        'Text evidence; figures/table pixels are not supplied.',
        'Bounded additional source cues can still miss evidence during branch selection.']}
    save(out/'manifest.json',manifest)
    print(json.dumps({'stage':'prepared','questions':len(selected),'documents':len(docs),'config':config_name}),flush=True)


def load_run(out):
    manifest=read_json(out/'manifest.json')
    for name,expected in manifest['input_hashes'].items():
        if sha(out/name)!=expected:raise ValueError('Changed run input: '+name)
    for name,expected in manifest['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Changed frozen implementation: '+name)
    return manifest,read_json(out/'cases.json'),read_json(out/'documents.json')


def parallel(fn,items,workers,stage):
    failures=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fn,item):item for item in items}
        for i,future in enumerate(as_completed(futures),1):
            try:future.result()
            except Exception as e:failures.append(type(e).__name__+': '+str(e)[:160])
            print(json.dumps({'stage':stage,'done':i,'total':len(items),'failures':len(failures),
                'last_error':failures[-1] if failures else None}),flush=True)
    if failures:raise RuntimeError(str(Counter(failures)))


def clients():
    CACHE.mkdir(parents=True,exist_ok=True)
    budget=RetrievalBudget(ROOT/'output/research-budget.json')
    embed=LiveEmbeddings(CACHE,budget)
    embed.cache=OLD/'embedding-cache'
    planner=SharedPlanner(CACHE,budget)
    return budget,embed,planner


def index_path(doc_id,split,central):
    return CACHE/'indexes'/(signature(doc_id)+'-'+split+central+'.json')


def run_index(out,workers):
    manifest,cases,docs=load_run(out)
    budget,embed,_=clients()
    def work(doc_id):
        doc=docs[doc_id]
        if all(index_path(doc_id,s,r).exists() for s in 'EJ' for r in 'EJ'):return
        j=EvidenceJev(CACHE,budget);j.max_state_chars=250000
        old_paths={s+r:ROOT/'output/live-tree-v1/indexes'/(signature(doc_id)+'-'+s+r+'.json') for s in 'EJ' for r in 'EJ'}
        if all(p.exists() for p in old_paths.values()):
            for sr,p in old_paths.items():
                idx=DocumentIndex.from_dict(read_json(p))
                if idx.source!=doc['text']:raise ValueError('Historical source mismatch')
                save(index_path(doc_id,*sr),idx.to_dict())
            return
        native=sections(doc)
        texts=[doc['text'][u['start']:u['end']] for u in doc['units']]
        import numpy as np
        vectors=embed.embed(['task: sentence similarity | query: '+s for s in texts],'splitting')
        lookup={u['start']:v for u,v in zip(doc['units'],vectors)}
        e_groups=[]
        for group in native:
            distances=[1-float(lookup[a['start']]@lookup[b['start']]) for a,b in zip(group,group[1:])]
            threshold=float(np.quantile(distances,.85)) if distances else 2
            current=[group[0]]
            for distance,u in zip(distances,group[1:]):
                if distance>threshold:e_groups.append(current);current=[]
                current.append(u)
            e_groups.append(current)
        runs=[[Block('paragraph',u['start'],u['end']) for u in group] for group in native]
        segmented=segment_runs(doc['text'],runs,j,Config())
        by_start={u['start']:u for u in doc['units']}
        j_groups=[[by_start[b.start] for b in group] for groups,_ in segmented for group in groups]
        bases={s:tree(doc,groups,s) for s,groups in [('E',e_groups),('J',j_groups)]}
        sentences=list(dict.fromkeys(n.title for n in bases['E'].root.walk() if n.kind=='sentence'))
        vectors=embed.embed(sentences,'central-sentences');lookup=dict(zip(sentences,vectors))
        scorer=CentroidRepresentatives(lookup.__getitem__,model_name=embed.model)
        for s in 'EJ':
            for r,central in [('E',scorer),('J',j)]:
                idx=reselect_representatives(bases[s],central,sentence_budget=8)
                save(index_path(doc_id,s,r),idx.to_dict())
    try:parallel(work,list(docs),workers,'index')
    finally:embed.pool.shutdown()


def run_plans(out,workers):
    _,cases,docs=load_run(out)
    budget,embed,planner=clients();embed.pool.shutdown()
    missing=[c for c in cases if not (CACHE/'plans'/(signature([SharedPlanner.model,c['id']])+'.json')).exists()]
    def work(batch):
        values=planner.plan_batch(batch,docs)
        for case in batch:
            path=CACHE/'plans'/(signature([SharedPlanner.model,case['id']])+'.json')
            save(path,{'id':case['id'],'model':SharedPlanner.model,'needs':values[case['id']]})
    parallel(work,[missing[i:i+8] for i in range(0,len(missing),8)],workers,'plans')


def run_retrieve(out,workers):
    manifest,cases,docs=load_run(out)
    config=EvidenceSearchConfig(**manifest['config'])
    budget,embed,_=clients()
    enc=tokenizer();count=lambda s:len(enc.encode(s,disallowed_special=()))
    def work(case):
        path=out/'retrieval'/(signature(case['id'])+'.json')
        if path.exists():return
        began=time.perf_counter();doc=docs[case['doc_id']]
        needs=read_json(CACHE/'plans'/(signature([SharedPlanner.model,case['id']])+'.json'))['needs']
        indexes={s+r:DocumentIndex.from_dict(read_json(index_path(doc['id'],s,r))) for s in 'EJ' for r in 'EJ'}
        for idx in indexes.values():
            if idx.source!=doc['text']:raise ValueError('Index does not match source')
        j=EvidenceJev(CACHE,budget);j.max_state_chars=250000
        queries=[f'Question: {case["question"]}\nEvidence need: {need}' for need in needs]
        qvec=embed.embed(['task: question answering | query: '+s for s in queries],'search-query')
        qlookup=dict(zip(needs,qvec))
        methods={}
        for sr,index in indexes.items():
            cards=[source_card(index,n,config) for n in index.root.walk() if n.kind!='document']
            texts=list(dict.fromkeys(card_text(c) for c in cards))
            vec=embed.embed(['task: retrieval | document: '+s for s in texts],'search-nodes')
            lookup=dict(zip(texts,vec))
            def score(need,rows):return [float(qlookup[need]@lookup[card_text(c)]) for c in rows]
            for route in 'EJ':
                started=time.perf_counter()
                result=global_embedding_search(index,needs,score,config) if route=='E' else promising_search(index,case['question'],needs,j.route_content,config)
                result['seconds']=time.perf_counter()-started
                result['packed']={str(b):pack_paragraphs(index,result['ranking'],count,b) for b in (512,1024,2048)}
                methods[sr+route]=result
        # Independent full-paragraph dense path; no tree visit restriction.
        texts=[u['heading']+'\n'+doc['text'][u['start']:u['end']] for u in doc['units']]
        pv=embed.embed(['task: retrieval | document: '+s for s in texts],'direct-paragraphs')
        ranks=[[f'p{i}' for i in sorted(range(len(texts)),key=lambda i:(-float(pv[i]@v),i))] for v in qvec]
        dense=fuse_ids(ranks)
        lexical=bm25(texts,case['question'])
        bm=[f'p{i}' for i in sorted(range(len(texts)),key=lambda i:(-lexical[i],i))]
        for name,rank in [('gemini_dense',dense),('bm25_dense_rrf',fuse_ids([dense,bm])),
                          ('hybrid',fuse_ids([methods['JJJ']['ranking'],dense]))]:
            methods[name]={'ranking':rank,'status':'complete','packed':{
                str(b):pack_paragraphs(indexes['JJ'],rank,count,b) for b in (512,1024,2048)}}
        save(path,{'id':case['id'],'doc_id':case['doc_id'],'needs':needs,'methods':methods,
            'elapsed_seconds':time.perf_counter()-began,'manifest_sha256':sha(out/'manifest.json')})
    try:parallel(work,cases,workers,'retrieve')
    finally:embed.pool.shutdown()


def run_qwen(out):
    import numpy as np
    _,cases,docs=load_run(out)
    model=QwenModels(ROOT/'output/retrieval-v3-models','embedding')
    vector_dir=CACHE/'qwen-vectors';vector_dir.mkdir(exist_ok=True)
    def vectors(texts,query):
        paths=[vector_dir/(signature([model.embedding_revision,model.instruction,query,s])+'.json') for s in texts]
        missing=list(dict.fromkeys((p,s) for p,s in zip(paths,texts) if not p.exists()))
        if missing:
            values=model.embed([s for p,s in missing],query)
            for (p,_),v in zip(missing,values):save(p,v)
        return np.asarray([read_json(p) for p in paths])
    for number,case in enumerate(cases,1):
        path=out/'qwen-pools'/(signature(case['id'])+'.json')
        if path.exists():continue
        started=time.perf_counter();doc=docs[case['doc_id']]
        texts=[u['heading']+'\n'+doc['text'][u['start']:u['end']] for u in doc['units']]
        v=vectors(texts,False);q=vectors([case['question']],True)[0]
        sims=v@q;order=sorted(range(len(texts)),key=lambda i:(-sims[i],i))
        save(path,{'id':case['id'],'ranking':[f'p{i}' for i in order], 'seconds':time.perf_counter()-started})
        print(json.dumps({'stage':'qwen-dense','done':number,'total':len(cases)}),flush=True)
    model.close()
    model=QwenModels(ROOT/'output/retrieval-v3-models','reranker')
    enc=tokenizer();count=lambda s:len(enc.encode(s,disallowed_special=()))
    for number,case in enumerate(cases,1):
        path=out/'baselines'/(signature(case['id'])+'.json')
        if path.exists():continue
        started=time.perf_counter();doc=docs[case['doc_id']]
        dense=read_json(out/'qwen-pools'/(signature(case['id'])+'.json'))['ranking']
        pool=dense[:30]
        texts=[doc['units'][int(p[1:])]['heading']+'\n'+doc['text'][doc['units'][int(p[1:])]['start']:doc['units'][int(p[1:])]['end']] for p in pool]
        key=signature([model.reranker_revision,model.instruction,case['question'],texts])
        cached=CACHE/'qwen-reranks'/(key+'.json')
        if cached.exists():scores=read_json(cached)
        else:scores=model.rerank(case['question'],texts);save(cached,scores)
        ranking=[pool[i] for i in sorted(range(len(pool)),key=lambda i:(-scores[i],i))]
        index=DocumentIndex.from_dict(read_json(index_path(doc['id'],'E','E')))
        methods={name:{'ranking':rank,'status':'complete','packed':{str(b):pack_paragraphs(index,rank,count,b) for b in (512,1024,2048)}}
                 for name,rank in [('qwen_dense',dense),('qwen_rerank',ranking)]}
        save(path,{'id':case['id'],'methods':methods,'seconds':time.perf_counter()-started})
        print(json.dumps({'stage':'qwen-rerank','done':number,'total':len(cases)}),flush=True)
    model.close()


def evidence_metrics(predicted,references,official):
    eligible=[r for r in references if r]
    if not eligible:return {'f1':None,'precision':None,'recall':None,'complete':None}
    # Every metric takes its best acceptable annotation, as the official F1 does.
    p=[len(set(predicted)&set(r))/len(predicted) if predicted else 0 for r in eligible]
    r=[len(set(predicted)&set(ref))/len(ref) for ref in eligible]
    return {'f1':max(official.paragraph_f1_score(predicted,ref) for ref in eligible),
            'precision':max(p),'recall':max(r),'complete':int(any(set(ref)<=set(predicted) for ref in eligible))}


def score_run(out):
    manifest,cases,docs=load_run(out)
    gold=read_json(SPLITS/manifest['partition']/'gold.json')
    spec=importlib.util.spec_from_file_location('official_qasper',SPLITS/manifest['partition']/'qasper_evaluator.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    rows=[];unmapped=Counter();empty=0
    for case in cases:
        document=docs[case['doc_id']]
        references=[[] if a['unanswerable'] else [s for s in a['evidence'] if 'FLOAT SELECTED' not in s]
                    for a in gold[case['id']]['answers']]
        empty+=not any(references)
        source_texts={document['text'][u['start']:u['end']] for u in document['units']}
        for ref in references:
            for text in ref:
                if text not in source_texts:unmapped[case['id']]+=1
        retrieval=read_json(out/'retrieval'/(signature(case['id'])+'.json'))
        baseline=read_json(out/'baselines'/(signature(case['id'])+'.json'))
        methods={**retrieval['methods'],**baseline['methods']}
        if set(methods)!=set(METHODS):raise ValueError('Incomplete method coverage')
        def text_of(ids):
            if len(ids)!=len(set(ids)):raise ValueError('Duplicate ranked paragraphs')
            values=[]
            for node_id in ids:
                if not node_id.startswith('p') or not node_id[1:].isdigit():raise ValueError('Non-paragraph output')
                unit=document['units'][int(node_id[1:])]
                values.append(document['text'][unit['start']:unit['end']])
            return values
        for name,result in methods.items():
            row={'id':case['id'],'doc_id':case['doc_id'],'method':name,'status':result['status'],'ranking':result['ranking']}
            for k in (1,3,5,10):
                values=evidence_metrics(text_of(result['ranking'][:k]),references,official)
                row.update({f'{metric}@{k}':v for metric,v in values.items()})
            for limit,pack in result['packed'].items():
                values=evidence_metrics(text_of(pack['paragraphs']),references,official)
                row.update({f'{metric}@tokens{limit}':v for metric,v in values.items()})
                row['source_tokens@'+limit]=pack['source_tokens']
            rows.append(row)
    if unmapped:save(out/'unmapped-evidence.json',dict(unmapped))
    metrics=[key for key in rows[0] if '@' in key]
    summary={'questions':len(cases),'documents':len(docs),'evidence_bearing':len(cases)-empty,'empty_text_evidence':empty,
        'unmapped_evidence_questions':len(unmapped),'methods':{},'partition':manifest['partition'],
        'output_unit':'whole original paragraphs','primary':'f1@5','status':'scored_complete'}
    for method in METHODS:
        subset=[r for r in rows if r['method']==method]
        summary['methods'][method]={key:statistics.mean([r[key] for r in subset if r[key] is not None]) for key in metrics}
        summary['methods'][method]['truncated']=sum(r['status']=='truncated' for r in subset)
    save(out/'scores.json',rows);save(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','index','plans','retrieve','qwen','score'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=16)
    p.add_argument('--partition',default='development');p.add_argument('--config',choices=list(CONFIGS),default='reference')
    p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.stage=='prepare':prepare(a.output,a.partition,a.limit,a.config)
    elif a.stage=='index':run_index(a.output,a.workers)
    elif a.stage=='plans':run_plans(a.output,a.workers)
    elif a.stage=='retrieve':run_retrieve(a.output,a.workers)
    elif a.stage=='qwen':run_qwen(a.output)
    else:score_run(a.output)
