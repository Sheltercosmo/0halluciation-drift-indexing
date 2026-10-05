"""Experimental source selection for an explicit set of query evidence needs."""
import math


def select_evidence(doc, candidates, relevance, need_scores, token_count, *,
                    budget=2048, coverage_weight=.6, length_power=.5):
    """Greedily trade off new need coverage, relevance and source-token cost.

    Need scores are candidate-major. Coverage is the maximum decision score
    reached for each need, not a calibrated probability of answer correctness.
    Only complete, disjoint source passages are returned, in document order.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError('budget must be a positive integer')
    for name,value in [('coverage_weight',coverage_weight),('length_power',length_power)]:
        if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(name+' must be finite and in [0,1]')
    if len(candidates)!=len(relevance) or len(candidates)!=len(need_scores):
        raise ValueError('Scores must cover every candidate')
    width = len(need_scores[0]) if need_scores else 0
    if candidates and not width:
        raise ValueError('At least one evidence need is required')
    for row in need_scores:
        if len(row)!=width:
            raise ValueError('Inconsistent evidence need count')
    for value in [*relevance, *(v for row in need_scores for v in row)]:
        if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Decision scores must be finite and in [0,1]')
    previous_end=-1
    for chunk in sorted(candidates,key=lambda c:c['start']):
        if not 0 <= chunk['start'] < chunk['end'] <= len(doc['text']) or chunk['start'] < previous_end:
            raise ValueError('Candidate spans must be valid and disjoint')
        if chunk['text'] != doc['text'][chunk['start']:chunk['end']]:
            raise ValueError('Candidate text differs from source')
        previous_end=chunk['end']
    if len({c['id'] for c in candidates})!=len(candidates):
        raise ValueError('Duplicate candidate IDs')
    def count(text):
        value=token_count(text)
        if type(value) is not int or value < 0:
            raise ValueError('token_count must return a nonnegative integer')
        return value
    def render(indices):
        ordered=sorted((candidates[i] for i in indices),key=lambda c:c['start'])
        return 'Title: '+doc['title']+'\n'+'\n\n'.join(f"[{i+1}] {c['heading']}\n{c['text']}" for i,c in enumerate(ordered))
    if count(render([])) > budget:
        raise ValueError('Document title alone exceeds the context budget')
    costs=[max(1,count(c['text'])) for c in candidates]
    selected,trace,covered=[],[],[0.]*width
    remaining=set(range(len(candidates)))
    while remaining:
        possible=[]
        for i in sorted(remaining):
            if count(render(selected+[i])) > budget:
                continue
            gain=sum(max(covered[j],need_scores[i][j])-covered[j] for j in range(width))/width
            utility=(coverage_weight*gain+(1-coverage_weight)*relevance[i])/(costs[i]**length_power)
            possible.append((utility,relevance[i],-i,i,gain))
        if not possible:
            break
        utility,_,_,i,gain=max(possible)
        if utility <= 0:
            break
        selected.append(i)
        remaining.remove(i)
        covered=[max(v,need_scores[i][j]) for j,v in enumerate(covered)]
        trace.append({'candidate_id':candidates[i]['id'],'utility':utility,
                      'coverage_gain':gain,'coverage_after':covered[:]})
    context=render(selected)
    return {'context':context,'context_tokens':count(context),
            'spans':[(candidates[i]['start'],candidates[i]['end']) for i in selected],
            'selected_ids':[candidates[i]['id'] for i in selected], 'selection_trace':trace,
            'need_coverage':covered}
