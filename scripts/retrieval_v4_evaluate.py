"""Complete-run evidence scoring and paired paper-cluster comparisons for v4."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bounded_eval import read_json,sha
from scripts.bounded_clients import save,signature
from scripts.retrieval_v4 import V4,ARMS,METHODS,validate
from scripts.analyze_retrieval_v3 import analyze
from scripts.retrieval_v3_statistics import cluster_summary,contrast,holm


def evaluate(out):
    validate(out)
    manifest=read_json(out/'manifest.json')
    if manifest['partition']=='test' and not (out/'outcomes-opened.json').exists():
        save(out/'outcomes-opened.json',{'opened_at':datetime.now(timezone.utc).isoformat(),
            'manifest_sha256':sha(out/'manifest.json'),'conformance_sha256':sha(out/'conformance.json')})
    summary=analyze(out);rows=read_json(out/'aligned-scores.json')
    import numpy as np
    result={'version':'tree-retrieval-v4','partition':manifest['partition'],'selected_system':'JJJ',
        'manifest_sha256':sha(out/'manifest.json'),'scores_sha256':sha(out/'aligned-scores.json'),
        'draws':10000,'seed':20261005,'cluster':'document','mean':'question-weighted',
        'population':summary['population'],'metrics':{}}
    for metric in ('recall@5','complete@5','f1@5','precision@5','recall@1','recall@3','recall@10',
                   'recall@tokens512','recall@tokens1024','recall@tokens2048'):
        means,boot,sums,counts=cluster_summary(rows,METHODS,metric)
        result['metrics'][metric]={'questions':int(counts.sum()),'documents':len(counts),'methods':{
            m:{'mean':float(means[i]),'ci95':np.quantile(boot[:,i],[.025,.975]).tolist()} for i,m in enumerate(METHODS)}}
        if metric!='recall@5':continue
        component_pairs=[]
        for position in range(3):
            for a in ARMS:
                if a[position]=='E':component_pairs.append((a,a[:position]+'J'+a[position+1:]))
        families={'components':component_pairs,
            'systems':[(m,'JJJ') for m in METHODS if m not in ARMS],
            'rerankers':[('qwen_rerank','qwen_jev_rerank'),('bge_rerank','bge_jev_rerank')]}
        for name,pairs in families.items():
            records=[]
            for a,b in pairs:
                coeff=[int(m==b)-int(m==a) for m in METHODS]
                records.append({'negative':a,'positive':b,'comparison':a+' -> '+b,**contrast(sums,counts,boot,coeff)})
            for row,p in zip(records,holm([r['p_two_sided'] for r in records])):
                row['p_holm']=p
            result[name]=records
        result['factor_effects']={}
        for i,factor in enumerate(manifest['factor_order']):
            coeff=[(1 if m[i]=='J' else -1)/4 if m in ARMS else 0 for m in METHODS]
            row=contrast(sums,counts,boot,coeff);row.pop('p_two_sided')
            result['factor_effects'][factor]=row
    save(out/'statistics.json',result)
    print({'statistics':str(out/'statistics.json'),'eligible':result['metrics']['recall@5']['questions']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('partition',choices=['development','test']);a=p.parse_args();evaluate(V4/a.partition)
