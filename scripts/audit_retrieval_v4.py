"""Independent no-gold audit of traversal, unchanged controls and source splits."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bounded_clients import save,signature
from scripts.bounded_eval import read_json,sha
from scripts.retrieval_v4 import load_run,V4,ARMS
from scripts.retrieval_v3 import index_path
from scripts.retrieval_v4_structure import restore_heading_hierarchy
from zero_index.index import DocumentIndex
from zero_index.evidence_search import fuse_ids


def audit(partial=False):
    out=V4/'test';m,cases,docs=load_run(out);source=ROOT/m['native_baseline_source']
    # Verify source allocation against the original pre-outcome manifest.
    old=read_json(source/'manifest.json')
    for name,expected in old['input_hashes'].items():
        if sha(out/name)!=expected:raise ValueError('Original frozen allocation changed')
    test_docs=set(docs)
    for partition in ('development','validation'):
        previous=read_json(ROOT/'output/improvement-v1'/partition/'documents.json')
        if test_docs & set(previous):raise ValueError('Document leakage across partitions')
    copied=0;traces=0;audited=0
    for case in cases:
        path=out/'retrieval'/(signature(case['id'])+'.json')
        if partial and not path.exists():continue
        a=read_json(path)
        if partial and a.get('status')!='complete':continue
        if a.get('status')!='complete':raise ValueError('Incomplete case')
        audited+=1
        for folder,prefix in [('baselines','qwen'),('bge','bge')]:
            native=read_json(source/folder/(signature(case['id'])+'.json'))
            for suffix in ('dense','rerank'):
                name=prefix+'_'+suffix
                if a['methods'][name]!=native['methods'][name]:raise ValueError('Unchanged control was altered: '+name)
                copied+=1
            name=prefix+'_jev_rerank'
            if a['methods'][name]['candidate_pool']!=native['methods'][prefix+'_dense']['ranking'][:30]:
                raise ValueError('Native dense pool changed')
        for arm in ARMS:
            if not arm.endswith('J'):continue
            idx=restore_heading_hierarchy(DocumentIndex.from_dict(read_json(index_path(case['doc_id'],*arm[:2]))))
            raw=a['methods'][arm]['retrieval'];config=raw['config'];decisions=0
            ranked=[[] for _ in a['needs']]
            for step in raw['trace']:
                nodes=[idx._node(pid) for pid in step['candidates']];values=step['scores']
                if len(values)!=len(nodes):raise ValueError('Score count mismatch')
                allowed=sorted([i for i,n in enumerate(nodes) if n.kind!='paragraph' and values[i]>=config['acceptance']],
                    key=lambda i:(-values[i],nodes[i].start,nodes[i].node_id))[:config['beam']]
                if step['selected_internal']!=[nodes[i].node_id for i in allowed]:raise ValueError('Incorrect beam/threshold')
                paragraphs=[i for i,n in enumerate(nodes) if n.kind=='paragraph']
                if step['scored_paragraphs']!=[nodes[i].node_id for i in paragraphs]:raise ValueError('Paragraph was dropped before pool ranking')
                ranked[step['need_index']].extend((values[i],nodes[i].node_id) for i in paragraphs)
                decisions+=len(nodes)+len(step['refined_ids'])
            by_need=[[pid for p,pid in sorted(items,key=lambda row:(-row[0],row[1]))] for items in ranked]
            if raw['ranking']!=fuse_ids(by_need) or raw['need_rankings']!=by_need:raise ValueError('Candidate aggregation mismatch')
            if decisions!=raw['decisions'] or decisions>config['max_decisions']:raise ValueError('Decision budget mismatch')
            traces+=1
    receipt={'questions':audited,'expected_questions':len(cases),'status':'complete' if audited==len(cases) else 'partial',
        'unchanged_native_predictions':copied,'verified_tree_traces':traces,
        'document_disjoint':True,'original_allocation_preserved':True,'gold_read':False,
        'manifest_sha256':sha(out/'manifest.json')}
    save(out/('independent-audit-partial.json' if partial else 'independent-audit.json'),receipt);print(receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');audit(parser.parse_args().partial)
