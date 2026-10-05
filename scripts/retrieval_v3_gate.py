"""Freeze paragraph retrieval settings before validation and one-time test access."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bounded_clients import save,signature
from scripts.bounded_eval import read_json,sha
from scripts.retrieval_v3 import source_hashes,CONFIGS,METHODS,SPLITS
from scripts.retrieval_v3_bge import REVISIONS
from scripts.validation_gate import check_data,write_once
from scripts.analyze_retrieval_v3 import analyze

RUN=ROOT/'output/retrieval-v3-confirmatory'


def validate_predictions(out,methods,expected_cases=None):
    """Reject missing/misbound results before any labels are loaded."""
    from scripts.bounded_eval import tokenizer
    manifest=read_json(out/'manifest.json');cases=read_json(out/'cases.json');docs=read_json(out/'documents.json')
    for name,expected in manifest['input_hashes'].items():
        if sha(out/name)!=expected:raise ValueError('Changed inputs')
    identities=[(c['id'],c['doc_id']) for c in cases]
    if len(identities)!=len(set(identities)) or len(cases)!=manifest['questions']:
        raise ValueError('Duplicated or incomplete question list')
    if expected_cases is not None and set(identities)!={(c['id'],c['doc_id']) for c in expected_cases}:
        raise ValueError('Question identities do not match registry')
    enc=tokenizer()
    for case in cases:
        doc=docs[case['doc_id']];valid={'p'+str(i) for i in range(len(doc['units']))};seen={}
        for folder in ('retrieval','baselines','bge'):
            path=out/folder/(signature(case['id'])+'.json')
            if not path.exists():raise ValueError('Missing prediction: '+folder)
            artifact=read_json(path)
            if artifact['id']!=case['id'] or artifact.get('doc_id',case['doc_id'])!=case['doc_id']:
                raise ValueError('Prediction identity mismatch')
            if folder=='retrieval' and artifact['manifest_sha256']!=sha(out/'manifest.json'):
                raise ValueError('Prediction registration mismatch')
            if seen.keys() & artifact['methods'].keys():raise ValueError('Duplicate method output')
            seen.update(artifact['methods'])
        if set(seen)!=set(methods):raise ValueError('Incomplete or unexpected method coverage')
        sizes={}
        for i,u in enumerate(doc['units']):
            if u['paragraph']!=i:raise ValueError('Paragraph identity mismatch')
            sizes['p'+str(i)]=len(enc.encode(doc['text'][u['start']:u['end']],disallowed_special=()))
        for result in seen.values():
            rank=result['ranking']
            if result['status'] not in ('complete','truncated'):raise ValueError('Failed inference')
            if len(rank)!=len(set(rank)) or not set(rank)<=valid:raise ValueError('Invalid paragraph ranking')
            if set(result['packed'])!={'512','1024','2048'}:raise ValueError('Missing output budget')
            for limit,packed in result['packed'].items():
                chosen,skipped,total=[],[],0
                for node_id in rank:
                    if total+sizes[node_id]<=int(limit):chosen.append(node_id);total+=sizes[node_id]
                    else:skipped.append(node_id)
                if packed!={'paragraphs':chosen,'skipped':skipped,'source_tokens':total}:
                    raise ValueError('Whole paragraph packing mismatch')


def register(pilot):
    if (SPLITS/'validation-opened.json').exists() or (SPLITS/'test-opened.json').exists():
        raise ValueError('Existing holdout access marker; do not silently reuse')
    split,registry=check_data(SPLITS)
    pilot_manifest=read_json(pilot/'manifest.json')
    if pilot_manifest['partition']!='development':raise ValueError('Tuning must use development only')
    for name,expected in pilot_manifest['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Pilot differs from frozen implementation')
    validate_predictions(pilot,METHODS+['bge_dense','bge_rerank'])
    pilot_summary=analyze(pilot)
    if pilot_summary['population']['all']!=16:raise ValueError('Unexpected calibration set')
    sources=source_hashes();sources['scripts/retrieval_v3_gate.py']=sha(Path(__file__))
    registration={'version':'tree-retrieval-v3-confirmatory','status':'frozen_before_validation',
        'registry_sha256':split['registry_sha256'],'sources':sources,
        'config':CONFIGS['b5_a2'],'methods':METHODS+['bge_dense','bge_rerank'],
        'system_candidates':['JJJ','hybrid'],
        'primary':'aligned paragraph recall@5','secondary':['complete@5','f1@5','recall@tokens2048'],
        'selection':'recall@5 then complete@5 then f1@5 then method id',
        'output_unit':'whole_original_paragraph','pilot_manifest_sha256':sha(pilot/'manifest.json'),
        'pilot_summary_sha256':sha(pilot/'aligned-summary.json'),
        'annotation_policy':'exact or unambiguous whitespace-normalized source paragraph match; require one fully mapped nonempty reference; report excluded and raw official metric separately',
        'baseline_models':{**pilot_manifest['baseline_models'],'bge':REVISIONS},
        'component_contrasts':[['EEE','JEE'],['EJE','JJE'],['EEJ','JEJ'],['EJJ','JJJ'],
             ['EEE','EJE'],['EEJ','EJJ'],['JEE','JJE'],['JEJ','JJJ'],
             ['EEE','EEJ'],['EJE','EJJ'],['JEE','JEJ'],['JJE','JJJ']],
        'final_system_contrasts':[['selected',m] for m in ['EEE','gemini_dense','bm25_dense_rrf','qwen_dense','qwen_rerank','bge_dense','bge_rerank']],
        'statistics_seed':20261005,'statistical_unit':'document cluster, question-weighted macro mean',
        'statistics':'paired document bootstrap 10000 draws; paired document sign randomization; Holm across 12 component contrasts and separately all registered final-system contrasts',
        'scope':'within-document QASPER paragraph retrieval; compact Qwen and BGE baselines, no corpus/frontier claim',
        'pilot_model':'gpt-6.1-sol','pilot_settings_selection':'Gemini 3.8 sweep selected beam 5/acceptance .2/global hits 30; repeated on development with final shared Codex planner',
        'practical_gain_target':.02,'pilot_template':pilot_manifest}
    RUN.mkdir(parents=True,exist_ok=True)
    write_once(RUN/'registration.json',registration)
    public=ROOT/'evals/registrations/tree-retrieval-v3.json'
    write_once(public,registration)
    print(json.dumps({'status':'registered','registration_sha256':sha(RUN/'registration.json')}))


def checked():
    reg=read_json(RUN/'registration.json');split,registry=check_data(SPLITS)
    if reg['registry_sha256']!=split['registry_sha256']:raise ValueError('Registry changed')
    for name,expected in reg['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen source changed: '+name)
    return reg,registry


def prepare_partition(partition,reg):
    out=RUN/partition
    cases=[c for c in read_json(SPLITS/partition/'cases.json') if c['dataset']=='qasper']
    docs=read_json(SPLITS/partition/'documents.json');docs={d:docs[d] for d in sorted({c['doc_id'] for c in cases})}
    write_once(out/'cases.json',cases);write_once(out/'documents.json',docs)
    manifest={**reg['pilot_template'],'partition':partition,'questions':len(cases),'documents':len(docs),
        'methods':reg['methods'],'config':reg['config'],'config_name':'frozen_b5_a2_g30',
        'sources':reg['sources'],'input_hashes':{n:sha(out/n) for n in ('cases.json','documents.json')},
        'registration_sha256':sha(RUN/'registration.json'),'primary':reg['primary'],'baseline_models':reg['baseline_models']}
    write_once(out/'manifest.json',manifest)
    print(json.dumps({'partition':partition,'questions':len(cases),'documents':len(docs),'output':str(out)}))


def open_validation():
    reg,_=checked()
    if (SPLITS/'test-opened.json').exists():raise ValueError('Test already opened')
    write_once(SPLITS/'validation-opened.json',{'version':'retrieval-v3','registration_sha256':sha(RUN/'registration.json'),
        'status':'validation_opened_no_new_candidates','registration':reg})
    prepare_partition('validation',reg)


def open_test():
    reg,_=checked()
    marker=read_json(SPLITS/'validation-opened.json')
    if marker['registration_sha256']!=sha(RUN/'registration.json'):raise ValueError('Registration mismatch')
    # Recompute from complete question/method records, not submitted aggregate scores.
    expected=[c for c in read_json(SPLITS/'registry.json') if c['partition']=='validation' and c['dataset']=='qasper']
    validate_predictions(RUN/'validation',reg['methods'],expected)
    summary=analyze(RUN/'validation')
    required=sum(r['partition']=='validation' and r['dataset']=='qasper' for r in read_json(SPLITS/'registry.json'))
    if summary['population']['all']!=required:raise ValueError('Incomplete validation questions')
    winner=sorted(reg['system_candidates'],key=lambda m:(-summary['methods'][m]['recall@5'],
        -summary['methods'][m]['complete@5'],-summary['methods'][m]['f1@5'],m))[0]
    frozen={'version':'retrieval-v3','registration_sha256':sha(RUN/'registration.json'),'winner':winner,
        'validation_scores_sha256':sha(RUN/'validation/aligned-scores.json'),'sources':reg['sources']}
    write_once(RUN/'frozen-winner.json',frozen)
    write_once(SPLITS/'test-opened.json',{'version':'retrieval-v3','status':'test_opened_once','frozen':frozen})
    prepare_partition('test',reg)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','open-validation','open-test'])
    p.add_argument('--pilot',type=Path,default=ROOT/'output/retrieval-v3-pilot-c');a=p.parse_args()
    if a.command=='register':register(a.pilot)
    elif a.command=='open-validation':open_validation()
    else:open_test()
