"""Generate a deterministic development search proposal; makes no model calls."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random


def proposal(seed=20261005, sampled=8):
    # The first five isolate each main option around a shared reference.
    anchors = [(None,None,2,None), (None,.9,2,None), (.3,None,2,None),
               (None,None,2,.5), (.3,.9,2,None)]
    grid=list(itertools.product((None,.15,.3,.5),(None,.85,.9,.95),(2,4),(None,.3,.5,.7)))
    remaining=[row for row in grid if row not in anchors]
    if type(sampled) is not int or not 0 <= sampled <= len(remaining):
        raise ValueError('sampled must fit the remaining finite search space')
    rows=anchors+random.Random(seed).sample(remaining,sampled)
    configs=[]
    for number,(separation,stop,beam,route) in enumerate(rows,1):
        settings={'split_separation_threshold':separation,'sentence_stop_threshold':stop,
                  'beam_width':beam,'route_acceptance_threshold':route,
                  'minimum_parent_probability':.05,'sentence_budget':8,
                  'max_node_scores':256,'max_preview_tokens':8192,'reader_tokens':2048}
        digest=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
        configs.append({'id':f'dev-{number:02d}','source':'anchor' if number <= len(anchors) else 'seeded_sample',
                        'configuration_sha256':digest,**settings})
    return {'status':'proposed_development_search_not_run','seed':seed,'configurations':configs,
            'model_calls':0,'held_out_access':False,
            'separation_rule_status':'Score helper implemented; group probability definitions and split adapter must be frozen before live use.',
            'selection':'On document-grouped development folds, retain nondominated QA/evidence versus usage configurations; at most three fixed candidates may enter validation.',
            'live_gate':'Publish source/model/data hashes, comparable group judgments, complete-run plan and budget reservation before inference.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261005)
    parser.add_argument('--sampled',type=int,default=8)
    parser.add_argument('--output',type=Path,default=Path('output/threshold-search-plan.json'))
    args=parser.parse_args()
    result=proposal(args.seed,args.sampled)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f"Prepared {len(result['configurations'])} development configurations; zero model calls; {args.output}")
