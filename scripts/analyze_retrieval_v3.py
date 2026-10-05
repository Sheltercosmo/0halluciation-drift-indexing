"""Analyze fixed paragraph rankings; keep official-string and aligned metrics distinct."""
import argparse
from collections import Counter,defaultdict
import importlib.util
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bounded_clients import save,signature
from scripts.bounded_eval import read_json,sha
from scripts.retrieval_v3_metrics import align_references,paragraph_metrics,oracle_at_k


def analyze(out):
    manifest=read_json(out/'manifest.json');cases=read_json(out/'cases.json');docs=read_json(out/'documents.json')
    for name,expected in manifest['input_hashes'].items():
        if sha(out/name)!=expected:raise ValueError('Changed inputs')
    partition=manifest['partition']
    source=ROOT/'output/improvement-v1'/partition
    gold=read_json(source/'gold.json')
    spec=importlib.util.spec_from_file_location('official_qasper_v3',source/'qasper_evaluator.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    rows=[];issues=Counter();population=Counter();oracles=[];expected=set(manifest['methods'])
    for case in cases:
        doc=docs[case['doc_id']]
        refs,diagnostics,raw=align_references(doc,gold[case['id']]['answers'])
        population['all']+=1;population['aligned_paragraph_evidence']+=bool(refs)
        population['raw_text_evidence']+=any(raw)
        population['unmapped_only']+=any(raw) and not refs
        for d in diagnostics:issues.update(d['issues'])
        if refs:oracles.append(oracle_at_k(refs,5))
        methods={}
        for folder in ('retrieval','baselines','bge'):
            path=out/folder/(signature(case['id'])+'.json')
            if path.exists():methods.update(read_json(path)['methods'])
        if not expected<=methods.keys():raise ValueError('Missing methods for '+case['id'])
        for method,result in methods.items():
            rank=result['ranking']
            if len(rank)!=len(set(rank)):raise ValueError('Duplicate outputs')
            row={'id':case['id'],'doc_id':case['doc_id'],'method':method,'status':result['status']}
            for k in (1,3,5,10):
                pred=rank[:k];values=paragraph_metrics(pred,refs)
                row.update({f'{m}@{k}':v for m,v in values.items()})
            texts=[]
            for p in rank[:5]:
                if not p.startswith('p') or not p[1:].isdigit():raise ValueError('Non-paragraph output')
                unit=doc['units'][int(p[1:])];texts.append(doc['text'][unit['start']:unit['end']])
            row['raw_official_f1@5']=max(official.paragraph_f1_score(texts,r) for r in raw)
            for budget,packed in result['packed'].items():
                row.update({f'{m}@tokens{budget}':v for m,v in paragraph_metrics(packed['paragraphs'],refs).items()})
                row['tokens@'+budget]=packed['source_tokens']
            row['returned_paragraphs@5']=min(5,len(rank));row['candidate_paragraphs']=len(rank)
            rows.append(row)
    summary={'population':dict(population),'annotation_issues':dict(issues),'primary':'aligned paragraph recall@5',
        'oracle_f1@5':statistics.mean(oracles) if oracles else None,'methods':{},'partition':partition,
        'output':'whole_original_paragraphs','metric_source_sha256':sha(ROOT/'scripts/retrieval_v3_metrics.py')}
    for method in sorted({r['method'] for r in rows}):
        subset=[r for r in rows if r['method']==method]
        if len(subset)!=len(cases):raise ValueError('Incomplete optional baseline: '+method)
        summary['methods'][method]={key:statistics.mean([r[key] for r in subset if r[key] is not None])
                                   for key in subset[0] if '@' in key or key=='candidate_paragraphs'}
        summary['methods'][method]['n']=len(subset)
        summary['methods'][method]['truncated']=sum(r['status']=='truncated' for r in subset)
    save(out/'aligned-scores.json',rows);save(out/'aligned-summary.json',summary)
    print(json.dumps({'population':summary['population'],'annotation_issues':dict(issues),
        'methods':{m:{k:round(v,4) for k,v in r.items() if k in ('f1@5','recall@5','complete@5','f1@tokens2048','candidate_paragraphs')}
                   for m,r in summary['methods'].items()}},indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();analyze(a.output)
