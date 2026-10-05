"""Pre-outcome amendment: Jev reranks the exact native dense candidate pools."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bounded_clients import save,signature
from scripts.bounded_eval import read_json,sha,tokenizer
from scripts.retrieval_v3 import load_run,CACHE,SPLITS,index_path,parallel
from scripts.retrieval_v3_clients import EvidenceJev,RetrievalBudget
from scripts.retrieval_v3_gate import RUN,checked as original_checked,validate_predictions
from scripts.validation_gate import write_once
from scripts.analyze_retrieval_v3 import analyze
from zero_index.index import DocumentIndex
from zero_index.evidence_search import fuse_ids,pack_paragraphs,validate_scores

CONTROLS={'qwen_jev_rerank':('baselines','qwen_dense'),
          'bge_jev_rerank':('bge','bge_dense'),
          'gemini_jev_rerank':('retrieval','gemini_dense')}
AMENDMENT=RUN/'jev-rerank-amendment.json'


def rerank_pool(doc,question,needs,pool,decide):
    if len(pool)>30 or len(pool)!=len(set(pool)):raise ValueError('Invalid top-30 pool')
    cards=[]
    for node_id in pool:
        u=doc['units'][int(node_id[1:])]
        cards.append({'node_id':node_id,'kind':'paragraph','heading_path':[u['heading']],
            'central_sentence':'','excerpts':[{'start':u['start'],'end':u['end'],
            'text':doc['text'][u['start']:u['end']],'role':'full_source'}]})
    rankings,scores=[],[]
    for need in needs:
        values=validate_scores(decide(question,need,cards),len(cards));scores.append(values)
        rankings.append([pool[i] for i in sorted(range(len(pool)),key=lambda i:(-values[i],i))])
    return {'ranking':fuse_ids(rankings),'candidate_pool':pool,'need_scores':scores,
            'source_cards_sha256':signature(cards),'needs':needs,'status':'complete'}


def amend():
    reg,_=original_checked()
    if (SPLITS/'test-opened.json').exists():raise ValueError('Test already exposed')
    if any((RUN/p/n).exists() for p in ('validation','test') for n in
           ('aligned-scores.json','aligned-summary.json','scores.json','summary.json','statistics.json')):
        raise ValueError('Amendment requires no held-out outcome inspection')
    out=RUN/'validation'
    if (out/'retrieval').exists():raise ValueError('Amend before factorial validation predictions')
    manifest,_,_=load_run(out)
    sources={name:sha(ROOT/name) for name in ('scripts/retrieval_v3_jev_controls.py','scripts/retrieval_v3_statistics.py')}
    record={'version':'tree-retrieval-v3-jev-rerank-controls','created_at':datetime.now(timezone.utc).isoformat(),
        'base_registration_sha256':sha(RUN/'registration.json'),'stage':'validation indexing/planning, no held-out scores inspected',
        'reason':'User requested Jev reranking as a same-pool model control.',
        'additional_methods':list(CONTROLS),'candidate_count':30,'sources':sources,
        'query_policy':{'qwen_jev_rerank':'original question, matching native Qwen baseline',
                        'bge_jev_rerank':'original question, matching native BGE baseline',
                        'gemini_jev_rerank':'same shared evidence needs, fused with RRF60'},
        'input_policy':'same dense top-30 IDs, native heading plus complete source paragraphs, no central-sentence restriction',
        'output_policy':'whole original paragraphs, common rank and source-token limits',
        'paired_reranker_contrasts':[['qwen_rerank','qwen_jev_rerank'],['bge_rerank','bge_jev_rerank']],
        'final_system_contrasts_added':[['selected',m] for m in CONTROLS],
        'statistics':'Holm family of 10 final-system contrasts; separate Holm family of 2 native same-pool reranker contrasts; original 12 component contrasts unchanged',
        'unchanged':'original eight arms, JJJ/hybrid selection, primary metric, data assignment and hyperparameters'}
    write_once(AMENDMENT,record)
    write_once(ROOT/'evals/registrations/tree-retrieval-v3-jev-rerank-amendment.json',record)
    write_once(out/'manifest-before-jev-controls.json',manifest)
    amended={**manifest,'methods':reg['methods']+list(CONTROLS),
             'sources':{**manifest['sources'],**sources},'amendment_sha256':sha(AMENDMENT)}
    save(out/'manifest.json',amended)
    print({'status':'amended_before_outcomes','methods':len(amended['methods']),'amendment_sha256':sha(AMENDMENT)})


def checked():
    reg,registry=original_checked();amendment=read_json(AMENDMENT)
    if amendment['base_registration_sha256']!=sha(RUN/'registration.json'):raise ValueError('Changed base registration')
    for name,expected in amendment['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Changed amended source: '+name)
    reg={**reg,'methods':reg['methods']+amendment['additional_methods'],
        'sources':{**reg['sources'],**amendment['sources']},
        'final_system_contrasts':reg['final_system_contrasts']+amendment['final_system_contrasts_added'],
        'paired_reranker_contrasts':amendment['paired_reranker_contrasts'],'amendment_sha256':sha(AMENDMENT)}
    return reg,registry


def run(out,workers):
    reg,_=checked();manifest,cases,docs=load_run(out)
    if manifest['amendment_sha256']!=reg['amendment_sha256']:raise ValueError('Wrong amendment')
    budget=RetrievalBudget(ROOT/'output/research-budget.json')
    enc=tokenizer();count=lambda s:len(enc.encode(s,disallowed_special=()))
    def work(case):
        doc=docs[case['doc_id']];index=DocumentIndex.from_dict(read_json(index_path(doc['id'],'E','E')))
        if index.source!=doc['text']:raise ValueError('Source mismatch')
        j=EvidenceJev(CACHE,budget);j.max_state_chars=250000
        for name,(folder,dense_name) in CONTROLS.items():
            path=out/folder/(signature(case['id'])+'.json');artifact=read_json(path)
            if artifact['id']!=case['id']:raise ValueError('Candidate identity mismatch')
            if name in artifact['methods']:continue
            pool=artifact['methods'][dense_name]['ranking'][:30]
            needs=artifact['needs'] if folder=='retrieval' else [case['question']]
            result=rerank_pool(doc,case['question'],needs,pool,j.route_content)
            result['packed']={str(b):pack_paragraphs(index,result['ranking'],count,b) for b in (512,1024,2048)}
            artifact['methods'][name]=result;artifact['amendment_sha256']=sha(AMENDMENT);save(path,artifact)
    parallel(work,cases,workers,'same-pool-jev-rerank')


def verify_pools(out):
    _,cases,_=load_run(out)
    for case in cases:
        for name,(folder,dense_name) in CONTROLS.items():
            artifact=read_json(out/folder/(signature(case['id'])+'.json'))
            result=artifact['methods'][name];pool=artifact['methods'][dense_name]['ranking'][:30]
            if result['candidate_pool']!=pool or set(result['ranking'])!=set(pool):
                raise ValueError('Candidate pool changed')
            if artifact['amendment_sha256']!=sha(AMENDMENT):raise ValueError('Missing amendment provenance')


def open_test():
    reg,registry=checked();out=RUN/'validation'
    expected=[c for c in registry if c['partition']=='validation' and c['dataset']=='qasper']
    validate_predictions(out,reg['methods'],expected);verify_pools(out)
    summary=analyze(out)
    winner=sorted(reg['system_candidates'],key=lambda m:(-summary['methods'][m]['recall@5'],
        -summary['methods'][m]['complete@5'],-summary['methods'][m]['f1@5'],m))[0]
    frozen={'version':'retrieval-v3-amended','registration_sha256':sha(RUN/'registration.json'),
        'amendment_sha256':sha(AMENDMENT),'winner':winner,'sources':reg['sources'],
        'validation_scores_sha256':sha(out/'aligned-scores.json')}
    write_once(RUN/'frozen-winner.json',frozen)
    write_once(SPLITS/'test-opened.json',{'version':'retrieval-v3-amended','status':'test_opened_once','frozen':frozen})
    test=RUN/'test';cases=[c for c in read_json(SPLITS/'test/cases.json') if c['dataset']=='qasper']
    all_docs=read_json(SPLITS/'test/documents.json');docs={d:all_docs[d] for d in sorted({c['doc_id'] for c in cases})}
    write_once(test/'cases.json',cases);write_once(test/'documents.json',docs)
    manifest={**read_json(out/'manifest.json'),'partition':'test','questions':len(cases),'documents':len(docs),
              'input_hashes':{n:sha(test/n) for n in ('cases.json','documents.json')}}
    write_once(test/'manifest.json',manifest)
    print({'selected':winner,'test_questions':len(cases),'test_documents':len(docs)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['amend','run','open-test'])
    p.add_argument('--output',type=Path,default=RUN/'validation');p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.command=='amend':amend()
    elif a.command=='open-test':open_test()
    else:run(a.output,a.workers)
