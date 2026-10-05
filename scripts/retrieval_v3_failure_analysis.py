"""Supplemental descriptive loss locations; never replace registered scores."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def best_reference(prediction,references):
    """Match primary recall's best-reference policy; ties keep annotation order."""
    return max(references,key=lambda r:len(set(prediction)&set(r))/len(r))


def tree_loss(paragraph,parents,kinds,root,result):
    if paragraph in result['ranking'][:5]:raise ValueError('Paragraph is not missed')
    if paragraph in result['ranking']:return 'reached_but_ranked_below_5'
    lineage=[];node=paragraph
    while node!=root:
        lineage.append(node);node=parents[node]
    lineage.reverse()
    by_need=defaultdict(dict)
    for step in result['trace']:
        for node,score in zip(step['candidates'],step['scores']):
            by_need[step['need_index']][node]=(node in step['selected'],score)
    barriers=[]
    for need in range(len(result['need_rankings'])):
        events=by_need[need]
        for depth,node in enumerate(lineage):
            if node not in events:
                if result['status']!='truncated':raise ValueError('Unexplained missing traversal event')
                barriers.append((depth,'decision_budget'));break
            selected,score=events[node]
            if not selected:
                reason='below_threshold' if score<result['config']['acceptance'] else 'outside_beam'
                barriers.append((depth,kinds[node]+':'+reason));break
        else:
            raise ValueError('Selected paragraph missing from ranking')
    if not barriers:raise ValueError('No search needs')
    # Attribute to the furthest progress made by any need, not a less relevant request.
    return max(barriers,key=lambda row:row[0])[1]


def analyze(out,public_dir=None):
    from scripts.bounded_eval import read_json,sha
    from scripts.bounded_clients import save,signature
    from scripts.retrieval_v3 import load_run,index_path
    from scripts.retrieval_v3_gate import validate_predictions
    from scripts.retrieval_v3_jev_controls import checked,verify_pools
    from scripts.retrieval_v3_metrics import align_references
    from zero_index.index import DocumentIndex
    reg,_=checked();manifest,cases,docs=load_run(out)
    if manifest['partition']!='test':raise ValueError('Use only the completed frozen test')
    if not (out/'statistics.json').exists():raise ValueError('Complete primary analysis first')
    validate_predictions(out,reg['methods']);verify_pools(out)
    gold=read_json(ROOT/'output/improvement-v1/test/gold.json')
    models=('JJJ','gemini_jev_rerank');counts={m:Counter() for m in models}
    questions={m:Counter() for m in models};details=[];indexes={}
    for case in cases:
        doc=docs[case['doc_id']];refs,_,_=align_references(doc,gold[case['id']]['answers'])
        if not refs:continue
        artifact=read_json(out/'retrieval'/(signature(case['id'])+'.json'))
        if case['doc_id'] not in indexes:
            index=DocumentIndex.from_dict(read_json(index_path(case['doc_id'],'J','J')))
            indexes[case['doc_id']]=(index._parents,{n.node_id:n.kind for n in index.root.walk()},index.root.node_id)
        parents,kinds,root=indexes[case['doc_id']]
        for model in models:
            result=artifact['methods'][model];prediction=result['ranking'][:5]
            ref=best_reference(prediction,refs);missed=[p for p in ref if p not in prediction]
            questions[model]['eligible_questions']+=1
            questions[model]['incomplete_top5_questions']+=bool(missed)
            questions[model]['chosen_references_larger_than_5']+=len(ref)>5
            questions[model]['minimum_misses_for_chosen_reference_at_k5']+=max(0,len(ref)-5)
            for paragraph in missed:
                if model=='JJJ':reason=tree_loss(paragraph,parents,kinds,root,result)
                else:reason=('candidate_reranked_below_5' if paragraph in result['candidate_pool'] else 'outside_dense_top30')
                counts[model][reason]+=1
                details.append({'id':case['id'],'doc_id':case['doc_id'],'method':model,
                    'paragraph':paragraph,'reason':reason})
    statistics=read_json(out/'statistics.json')
    for model in models:
        n=questions[model]['eligible_questions']
        assert n==statistics['metrics']['recall@5']['questions']
        expected_incomplete=round(n*(1-statistics['metrics']['complete@5']['methods'][model]['mean']))
        assert questions[model]['incomplete_top5_questions']==expected_incomplete
    report={'analysis':'supplemental descriptive analysis added after validation; separate from registered hypothesis tests',
        'partition':'test','questions':len(cases),'statistics_sha256':sha(out/'statistics.json'),
        'analysis_source_sha256':sha(Path(__file__)),
        'reference_policy':'Per method, choose the fully aligned reference with highest recall@5; ties keep annotation order. Count missed paragraph instances in that reference.',
        'tree_attribution':'For a paragraph never retrieved, use the deepest evidence-path node reached by any shared search need. This locates the loss; it does not establish its cause.',
        'methods':{m:{**questions[m],'missed_reference_paragraph_instances':sum(counts[m].values()),
            'loss_locations':dict(sorted(counts[m].items()))} for m in models}}
    save(out/'failure-analysis.json',report);save(out/'failure-analysis-cases.json',details)
    if public_dir:
        save(public_dir/'failure-analysis.json',report)
        catalog=read_json(public_dir/'catalog.json');path=public_dir/'failure-analysis.json'
        catalog['files'][path.name]={'sha256':sha(path),'bytes':path.stat().st_size}
        catalog['supplemental_analysis']='Descriptive loss locations added after validation; no additional hypothesis tests.'
        save(public_dir/'catalog.json',catalog)
    print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--public-dir',type=Path)
    a=p.parse_args();analyze(a.output,a.public_dir)
