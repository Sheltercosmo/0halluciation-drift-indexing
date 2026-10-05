"""Export complete, verified retrieval artifacts without private provider ledgers."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bounded_clients import save,signature
from scripts.bounded_eval import read_json,sha
from scripts.retrieval_v3 import load_run
from scripts.retrieval_v3_gate import validate_predictions,RUN
from scripts.retrieval_v3_jev_controls import checked,verify_pools,AMENDMENT


def export(out,destination):
    reg,registry=checked();manifest,cases,docs=load_run(out)
    expected=[c for c in registry if c['partition']==manifest['partition'] and c['dataset']=='qasper']
    validate_predictions(out,reg['methods'],expected);verify_pools(out)
    summary=read_json(out/'aligned-summary.json')
    if summary['population']['all']!=len(cases):raise ValueError('Incomplete score summary')
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('manifest.json','aligned-summary.json','statistics.json'):
        shutil.copyfile(out/name,destination/name)
    with (destination/'predictions.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:
            for case in cases:
                methods={};needs=None
                for folder in ('retrieval','baselines','bge'):
                    artifact=read_json(out/folder/(signature(case['id'])+'.json'))
                    methods.update(artifact['methods'])
                    if folder=='retrieval':needs=artifact['needs']
                record={'id':case['id'],'doc_id':case['doc_id'],'shared_needs':needs,
                        'methods':{name:methods[name] for name in reg['methods']}}
                stream.write((json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n').encode())
    with (destination/'scores.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:
            for row in read_json(out/'aligned-scores.json'):
                stream.write((json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode())
    catalog={'partition':manifest['partition'],'questions':len(cases),'documents':len(docs),
        'methods':reg['methods'],'question_records':len(cases),'method_predictions':len(cases)*len(reg['methods']),
        'registration_sha256':sha(RUN/'registration.json'),'amendment_sha256':sha(AMENDMENT),
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(destination.iterdir()) if p.name!='catalog.json'},
        'scope':'whole original paragraphs inside the supplied QASPER paper',
        'source_inputs':'prepared document-disjoint allocation; input hashes and source code are in manifest.json'}
    save(destination/'catalog.json',catalog)
    print(json.dumps({'output':str(destination),'questions':len(cases),'method_predictions':catalog['method_predictions'],
                      'files':catalog['files']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();export(a.output,a.destination)
