"""Offline validation diagnostics. Frozen predictions and test stay untouched."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bounded_eval import read_json,sha
from scripts.bounded_clients import save,signature
from scripts.retrieval_v3 import load_run,index_path
from scripts.retrieval_v3_gate import validate_predictions
from scripts.retrieval_v3_jev_controls import checked,verify_pools
from scripts.retrieval_v3_metrics import align_references,paragraph_metrics
from scripts.retrieval_v3_failure_analysis import best_reference,tree_loss
from zero_index.index import DocumentIndex
from zero_index.evidence_search import source_card,EvidenceSearchConfig,fuse_ids


def barrier_details(index,paragraph,result):
    """Return every need's first barrier and the deepest observed one."""
    lineage=[];node=paragraph
    while node!=index.root.node_id:
        lineage.append(node);node=index._parents[node]
    lineage.reverse();events=defaultdict(dict)
    for step in result['trace']:
        for node,score in zip(step['candidates'],step['scores']):
            events[step['need_index']][node]=(step,score)
    barriers=[]
    for need in range(len(result['need_rankings'])):
        for depth,node in enumerate(lineage):
            if node not in events[need]:
                if result['status']!='truncated':raise ValueError('Missing event without budget stop')
                barriers.append({'depth':depth,'need':need,'node':node,'reason':'decision_budget'});break
            step,score=events[need][node]
            if node in step['selected']:continue
            kind=index._node(node).kind
            reason='below_threshold' if score<result['config']['acceptance'] else 'outside_beam'
            scores=dict(zip(step['candidates'],step['scores']))
            barriers.append({'depth':depth,'need':need,'node':node,'kind':kind,'score':score,
                'reason':kind+':'+reason,'refined':node in step['refined_ids'],
                'selected_count':len(step['selected']),
                'accepted_candidates':sum(s>=result['config']['acceptance'] for s in step['scores']),
                'selected_at_same_score':sum(scores[n]==score for n in step['selected']),
                'selected_scores':[scores[n] for n in step['selected']]});break
    return barriers,max(barriers,key=lambda row:row['depth']) if barriers else None


def scored_paragraph_replay(index,result):
    """Re-rank already scored terminal paragraphs; do not invent unseen scores.

    Preserve original per-need sort and RRF, but remove terminal acceptance/beam
    filtering. This cannot recover paragraphs below unvisited internal nodes.
    """
    per_need=[{} for _ in result['need_rankings']]
    for step in result['trace']:
        for node,score in zip(step['candidates'],step['scores']):
            if index._node(node).kind=='paragraph':per_need[step['need_index']][node]=score
    ranks=[[node for node,score in sorted(rows.items(),key=lambda item:(-item[1],item[0]))]
           for rows in per_need]
    return fuse_ids(ranks)


def run():
    out=ROOT/'output/retrieval-v3-confirmatory/validation'
    reg,_=checked();manifest,cases,docs=load_run(out)
    assert manifest['partition']=='validation'
    validate_predictions(out,reg['methods']);verify_pools(out)
    gold=read_json(ROOT/'output/improvement-v1/validation/gold.json')
    counts=Counter();locations=Counter();recoveries=defaultdict(Counter)
    details=[];cache={};replays=[]
    for case in cases:
        doc=docs[case['doc_id']];refs,_,_=align_references(doc,gold[case['id']]['answers'])
        if not refs:continue
        record=read_json(out/'retrieval'/(signature(case['id'])+'.json'))
        methods=record['methods'];result=methods['JJJ'];prediction=result['ranking'][:5]
        if case['doc_id'] not in cache:
            cache[case['doc_id']]=DocumentIndex.from_dict(read_json(index_path(case['doc_id'],'J','J')))
        index=cache[case['doc_id']]
        ref=best_reference(prediction,refs)
        missed=[p for p in ref if p not in prediction]
        counts['eligible_questions']+=1;counts['incomplete_top5_questions']+=bool(missed)
        replay=scored_paragraph_replay(index,result)
        replays.append({'id':case['id'],'doc_id':case['doc_id'],
            'original_recall5':paragraph_metrics(prediction,refs)['recall'],
            'replay_recall5':paragraph_metrics(replay[:5],refs)['recall'],
            'original_recall10':paragraph_metrics(result['ranking'][:10],refs)['recall'],
            'replay_recall10':paragraph_metrics(replay[:10],refs)['recall']})
        for paragraph in missed:
            counts['missed_reference_paragraph_instances']+=1
            reason=tree_loss(paragraph,index._parents,{n.node_id:n.kind for n in index.root.walk()},index.root.node_id,result)
            locations[reason]+=1
            recovered={m:paragraph in methods[m]['ranking'][:5] for m in ('EJJ','JEJ','JJE','gemini_jev_rerank','hybrid')}
            for method,yes in recovered.items():recoveries[reason][method]+=yes
            item={'id':case['id'],'doc_id':case['doc_id'],'paragraph':paragraph,'reason':reason,
                  'question':case['question'],'needs':record['needs'],'recovered_at5':recovered,
                  'source':index.source[index._node(paragraph).start:index._node(paragraph).end]}
            if paragraph not in result['ranking']:
                barriers,deepest=barrier_details(index,paragraph,result)
                item['barriers']=barriers;item['deepest']=deepest
                if deepest and deepest['reason']!='decision_budget':
                    node=index._node(deepest['node']);p=index._node(paragraph)
                    card=source_card(index,node,EvidenceSearchConfig(**result['config']),detail=deepest['refined'])
                    visible=any(x['start']<p.end and x['end']>p.start for x in card['excerpts'])
                    item['barrier_card']=card;item['evidence_paragraph_visible_in_card']=visible
                    item['paragraph_central']=p.central
                    item['topic_central']=index._node(index._parents[paragraph]).central
                    if node.kind!='paragraph':
                        counts['internal_barrier_instances']+=1
                        counts['internal_barrier_without_evidence_paragraph_excerpt']+=not visible
                        counts['internal_barrier_empty_excerpts']+=not card['excerpts']
                    if deepest['reason'].endswith(':outside_beam'):
                        counts['beam_barrier_instances']+=1
                        counts['beam_barrier_tied_with_selected']+=deepest['selected_at_same_score']>0
                    if deepest['reason'].endswith(':below_threshold'):
                        score=deepest['score']
                        counts['threshold_score_zero']+=score==0
                        counts['threshold_score_below_005']+=score<.05
                        counts['threshold_score_005_to_02']+=.05<=score<.2
            details.append(item)
    fanout=[];nested_headings=0;papers_with_nested_paths=0
    for doc_id in docs:
        index=cache.get(doc_id) or DocumentIndex.from_dict(read_json(index_path(doc_id,'J','J')))
        fanout.append(len(index.root.children))
        count=sum(' ::: ' in n.title for n in index.root.children)
        nested_headings+=count;papers_with_nested_paths+=bool(count)
    summary={'partition':'validation','analysis':'post-outcome offline engineering diagnostics; no API calls or test access',
        'source_sha256':sha(Path(__file__)),'statistics_sha256':sha(out/'statistics.json'),
        'reference_policy':'JJJ best recall@5 reference; ties use annotation order. Every recovery refers to the same missed original paragraph.',
        'counts':dict(counts),'loss_locations':dict(locations),
        'native_heading_structure':{'papers':len(docs),'papers_with_nested_heading_paths':papers_with_nested_paths,
            'nested_heading_paths_attached_to_root':nested_headings,
            'root_children_median':statistics.median(fanout),'root_children_max':max(fanout)},
        'same_paragraph_recovered_at5':{reason:dict(values) for reason,values in recoveries.items()},
        'terminal_filter_replay':{'policy':'Remove only terminal acceptance/beam filtering for already scored paragraphs, retain observed internal traversal and RRF. Offline diagnostic, not a new live-system result.',
            **{key:sum(r[key] for r in replays)/len(replays) for key in ('original_recall5','replay_recall5','original_recall10','replay_recall10')},
            'questions_improved_at5':sum(r['replay_recall5']>r['original_recall5'] for r in replays),
            'questions_worsened_at5':sum(r['replay_recall5']<r['original_recall5'] for r in replays)}}
    dest=ROOT/'output/retrieval-v3-validation-diagnostics'
    save(dest/'summary.json',summary);save(dest/'cases.json',details);save(dest/'replay.json',replays)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
