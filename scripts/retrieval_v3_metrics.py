"""Paragraph-identity metrics with auditable QASPER annotation alignment."""
from collections import defaultdict
import re


def normalize(text):
    return ' '.join(text.split())


def align_references(doc,answers):
    by_exact,by_normal=defaultdict(list),defaultdict(list)
    for unit in doc['units']:
        text=doc['text'][unit['start']:unit['end']]
        key='p'+str(unit['paragraph'])
        by_exact[text].append(key);by_normal[normalize(text)].append(key)
    headings={normalize(u['heading']) for u in doc['units'] if u['heading']}
    valid,diagnostics,raw=[],[],[]
    for annotation in answers:
        evidence=[] if annotation['unanswerable'] else [s for s in annotation['evidence'] if 'FLOAT SELECTED' not in s]
        raw.append(evidence)
        mapped,issues=[],[]
        for text in evidence:
            matches=by_exact.get(text,[]) or by_normal.get(normalize(text),[])
            if len(matches)==1:mapped.append(matches[0])
            elif len(matches)>1:issues.append('ambiguous_duplicate_paragraph')
            elif normalize(text) in headings:issues.append('heading_annotation')
            else:issues.append('unmapped_annotation')
        if mapped and not issues:
            valid.append(list(dict.fromkeys(mapped)))
        diagnostics.append({'raw_text_items':len(evidence),'mapped_items':len(mapped),'issues':issues,
                            'valid_paragraph_reference':bool(mapped) and not issues})
    return valid,diagnostics,raw


def paragraph_metrics(prediction,references):
    if len(prediction)!=len(set(prediction)):raise ValueError('Duplicate paragraph predictions')
    if not references:return {'f1':None,'precision':None,'recall':None,'complete':None}
    values=[]
    for ref in references:
        common=len(set(prediction)&set(ref))
        p=common/len(prediction) if prediction else 0
        r=common/len(ref)
        values.append({'f1':2*p*r/(p+r) if p+r else 0,'precision':p,'recall':r,
                       'complete':int(set(ref)<=set(prediction))})
    return {key:max(row[key] for row in values) for key in values[0]}


def oracle_at_k(references,k):
    if not references:return None
    return max(2*min(k,len(r))/(min(k,len(r))+len(r)) for r in references)
