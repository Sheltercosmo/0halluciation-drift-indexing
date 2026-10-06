"""Offline integrity, policy, selection and score checks for the standard."""
import gzip,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from copy import deepcopy
from scripts.jev_joint_evidence import joint_select
from scripts.retrieval_standard import STANDARD_POLICY
from zero_index.configuration import RetrievalConfig, MEASURED_JJJ_CONFIG
DEST=ROOT/'evals/results/shared-context-standard/test'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f]
def check(value,message):
    if not value:raise ValueError(message)

def replay():
    catalog=read(DEST/'catalog.json');manifest=read(DEST/'manifest.json');stats=read(DEST/'statistics.json')
    for name,h in catalog['files'].items():check(hashlib.sha256((DEST/name).read_bytes()).hexdigest()==h,'Changed archive: '+name)
    check(STANDARD_POLICY==manifest['policy'],'Standard policy drifted from retained measured version')
    check(RetrievalConfig.load(ROOT/'configs/retrieval-jjj-measured.json')==MEASURED_JJJ_CONFIG,
          'Historical JJJ configuration changed')
    check(RetrievalConfig.load(ROOT/'configs/retrieval-standard.json')==RetrievalConfig(),
          'Incorrect EEJ deployment default')
    check(stats['manifest_sha256']==hashlib.sha256((DEST/'manifest.json').read_bytes()).hexdigest(),'Statistics provenance mismatch')
    source_rows=rows(DEST/'runtime-sources.jsonl.gz')
    for source in source_rows:
        s=source['source'];variants=(s.encode(),s.replace('\n','\r\n').encode())
        check(any(hashlib.sha256(v).hexdigest()==source['sha256'] for v in variants),'Source snapshot mismatch')
    check({s['path']:s['sha256'] for s in source_rows}==manifest['sources'],'Incomplete sources')
    predictions=rows(DEST/'predictions.jsonl.gz');scores=rows(DEST/'scores.jsonl.gz')
    upstream={r['id']:r for r in rows(DEST/'upstream-predictions.jsonl.gz')}
    check(len(predictions)==len(upstream)==728 and len({p['id'] for p in predictions})==728,'Incomplete questions')
    lookup={(r['id'],r['method']):r for r in scores};check(len(lookup)==len(scores)==728*22,'Incomplete scores')
    historical=rows(ROOT/'evals/results/tree-retrieval-v4/test/scores.jsonl.gz')
    for r in historical:check(lookup[r['id'],r['method']]==r,'Historical baseline changed')
    matched=0
    for a in predictions:
        check(a['status']=='complete' and set(a['arms'])=={'Jev','hybrid'},'Incomplete system predictions')
        for arm,record in a['arms'].items():
            packet=record['method']['packet'];result=record['method']['result'];ranking=record['baseline']
            check(ranking==upstream[a['id']]['pairwise'][arm]['ranking'],'Wrong upstream pairwise seed')
            check(set(packet['target_ids'])==set(ranking[:12]),'Changed standard target pool')
            check(not packet['expanded_targets'],'Experimental target expansion activated')
            def decide(q,p,ids):
                check(q==result['question'],'Changed question');expected=deepcopy(packet);expected['selection_size']=5
                if p==expected:key='scores_source_order'
                else:
                    for field in ('passages','target_ids','links'):expected[field].reverse()
                    check(p==expected,'Changed shared context');key='scores_reverse_order'
                return [result[key][pid] for pid in ids]
            check(joint_select(result['question'],packet,ranking,decide,k=5)==result,'Selection replay mismatch')
            method=('JJJ' if arm=='Jev' else 'hybrid')+'_shared'
            check(lookup[a['id'],method]['prediction']==result['selected'],'Scored different output')
            matched+=1
    for metric,group in stats['metrics'].items():
        for method,value in group['methods'].items():
            selected=[r[metric] for r in scores if r['method']==method and r[metric] is not None]
            check(len(selected)==640 and abs(statistics.mean(selected)-value['mean'])<1e-12,'Metric mismatch')
    check(read(DEST/'standard-replay.json')['matched']==matched==1456,'Pipeline replay incomplete')
    print({'standard':STANDARD_POLICY['name'],'questions':728,'selections_replayed':matched,
           'baseline_rows_unchanged':len(historical),'score_rows':len(scores),'new_model_calls':0,'passed':True})

if __name__=='__main__':replay()
