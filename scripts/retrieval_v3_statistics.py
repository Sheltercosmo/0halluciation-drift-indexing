"""Paired, document-cluster inference for the preregistered retrieval comparison."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bounded_eval import read_json,sha
from scripts.bounded_clients import save


def holm(values):
    order=sorted(range(len(values)),key=values.__getitem__)
    adjusted=[0.]*len(values);previous=0.
    for position,i in enumerate(order):
        previous=max(previous,min(1.,values[i]*(len(values)-position)))
        adjusted[i]=previous
    return adjusted


def cluster_summary(rows,methods,metric,draws=10000,seed=20261005):
    import numpy as np
    by_id=defaultdict(dict)
    for row in rows:by_id[(row['doc_id'],row['id'])][row['method']]=row.get(metric)
    docs=sorted({d for d,q in by_id});doc_lookup={d:i for i,d in enumerate(docs)}
    sums=np.zeros((len(docs),len(methods)));counts=np.zeros(len(docs),dtype=int)
    for (doc,q),results in by_id.items():
        if set(results)!=set(methods):raise ValueError('Incomplete method coverage')
        values=[results[m] for m in methods]
        if all(v is None for v in values):continue
        if any(v is None for v in values):raise ValueError('Different metric denominators')
        index=doc_lookup[doc];sums[index]+=np.asarray(values);counts[index]+=1
    active=counts>0;sums=sums[active];counts=counts[active]
    if not len(counts):raise ValueError('Empty evidence population')
    rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(counts),np.repeat(1/len(counts),len(counts)),size=draws)
    boot=(weights@sums)/(weights@counts)[:,None]
    means=sums.sum(axis=0)/counts.sum()
    return means,boot,sums,counts


def contrast(sums,counts,boot,coefficients,seed=20261005,draws=10000):
    import numpy as np
    coefficients=np.asarray(coefficients);cluster_deltas=sums@coefficients
    delta=float(cluster_deltas.sum()/counts.sum())
    rng=np.random.default_rng(seed)
    signs=rng.choice([-1,1],size=(draws,len(counts)))
    null=(signs@cluster_deltas)/counts.sum()
    # Two-sided randomization with +1 correction; whole documents change sign.
    p=float((1+np.sum(np.abs(null)>=abs(delta)-1e-12))/(draws+1))
    interval=np.quantile(boot@coefficients,[.025,.975]).tolist()
    return {'delta':delta,'ci95':interval,'p_two_sided':p}


def run(out):
    from scripts.retrieval_v3_gate import checked,validate_predictions,RUN
    reg,_=checked();manifest=read_json(out/'manifest.json')
    if manifest['partition'] not in ('validation','test'):raise ValueError('Confirmatory partitions only')
    validate_predictions(out,reg['methods'])
    rows=read_json(out/'aligned-scores.json');methods=reg['methods']
    summary=read_json(out/'aligned-summary.json')
    winner=read_json(RUN/'frozen-winner.json')['winner'] if manifest['partition']=='test' else None
    result={'partition':manifest['partition'],'registration_sha256':sha(RUN/'registration.json'),
        'scores_sha256':sha(out/'aligned-scores.json'),'statistics_source_sha256':sha(Path(__file__)),
        'draws':10000,'seed':reg['statistics_seed'],'cluster':'document','mean':'question-weighted',
        'interval':'95% marginal percentile bootstrap; not simultaneous intervals',
        'population':summary['population'],'selected_system':winner,'metrics':{}}
    for metric in ('recall@5','complete@5','f1@5','precision@5','recall@1','recall@3','recall@10','recall@tokens512','recall@tokens1024','recall@tokens2048'):
        means,boot,sums,counts=cluster_summary(rows,methods,metric)
        result['metrics'][metric]={'questions':int(counts.sum()),'documents':len(counts),'methods':{
            m:{'mean':float(means[i]),'ci95':__import__('numpy').quantile(boot[:,i],[.025,.975]).tolist()}
            for i,m in enumerate(methods)}}
        if metric!='recall@5':continue
        families={'components':[(a+' -> '+b,a,b) for a,b in reg['component_contrasts']]}
        if winner:families['system_contrasts']=[(winner+' vs '+b,b,winner) for a,b in reg['final_system_contrasts']]
        for family,pairs in families.items():
            records=[]
            for label,a,b in pairs:
                c=[int(m==b)-int(m==a) for m in methods]
                records.append({'comparison':label,'negative':a,'positive':b,**contrast(sums,counts,boot,c)})
            adjusted=holm([r['p_two_sided'] for r in records])
            for r,p in zip(records,adjusted):r['p_holm']=p;r['significant_holm_05']=p<.05
            result[family]=records
        # Factor averages are descriptive summaries, not additional unadjusted tests.
        factors={}
        for factor in range(3):
            c=[0.]*len(methods)
            for arm in reg['pilot_template']['methods'][:8]:
                c[methods.index(arm)]=(1 if arm[factor]=='J' else -1)/4
            record=contrast(sums,counts,boot,c);record.pop('p_two_sided')
            factors[reg['pilot_template']['factor_order'][factor]]=record
        result['descriptive_factor_effects']=factors
    save(out/'statistics.json',result)
    print(json.dumps({'partition':result['partition'],'primary':result['metrics']['recall@5'],
                      'selected_system':winner,'components':result['components'],
                      'system_contrasts':result.get('system_contrasts',[])}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();run(a.output)
